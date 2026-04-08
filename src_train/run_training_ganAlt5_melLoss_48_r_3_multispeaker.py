"""
GAN Alt5 training with mel-band reconstruction loss — 48 kHz, r=3, multispeaker.

Changes vs run_training_ganAlt5_48_r_3_multispeaker.py:
  - WithLoss_G_melLoss adds a masked mel-spectrogram L1 loss on top of RMSE + adversarial.
  - Mel loss is applied only to bins above the LR Nyquist (sr // r / sr * n_mels),
    i.e. the generated band, so it does not conflict with the LR content.
  - Multi-scale mel: two STFT resolutions (n_fft=512 and n_fft=2048) are averaged.
    At 48 kHz a single n_fft cannot simultaneously give good frequency resolution
    and enough time frames; multi-scale avoids this trade-off.
  - New CLI arg --mel_loss_weight (default 0.1) controls contribution.
  - Log file gains a mel_loss column.
"""

import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
import numpy as np

from models.gan import Generator, Discriminator, BCEWithSquareLoss
from models.calculate_snr_lsd import get_lsd, get_snr
from torch.utils.data import Dataset

import h5py


class H5Dataset(Dataset):
    """Lazy HDF5 loader — reads patches on demand to avoid loading full dataset into RAM."""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.length = f['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            x = torch.tensor(f['data'][idx],  dtype=torch.float32)
            y = torch.tensor(f['label'][idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
class WithLoss_D(nn.Module):

    def __init__(self, D_net, G_net, loss_fn):
        super().__init__()
        self.D_net    = D_net
        self.G_net    = G_net
        self.loss_fn  = loss_fn
        self.sigmoid  = nn.Sigmoid()

    def forward(self, hr, fake_patches):
        logits_fake, _ = self.D_net(fake_patches)
        logits_real, _ = self.D_net(hr)

        d_loss1 = self.loss_fn(logits_real, torch.ones_like(logits_real))
        d_loss2 = self.loss_fn(logits_fake, torch.zeros_like(logits_fake))
        d_loss  = d_loss1 + d_loss2

        d_real = self.sigmoid(logits_real).squeeze()
        d_fake = self.sigmoid(logits_fake).squeeze()
        d_acc  = torch.mean(torch.cat(
            (torch.ge(d_real, 0.5).float(), torch.le(d_fake, 0.5).float()), 0))

        return d_loss, d_loss1, d_loss2, d_acc


# ---------------------------------------------------------------------------
class WithLoss_G_melLoss(nn.Module):
    """Generator loss = RMSE + adv + multi-scale mel-band L1."""

    # Two STFT scales: small n_fft for temporal resolution,
    # large n_fft for spectral resolution (important at 48 kHz).
    MEL_SCALES = [
        dict(n_fft=1024, hop_length=256,  n_mels=128),
        dict(n_fft=2048, hop_length=512,  n_mels=128),
    ]

    def __init__(self, D_net, G_net, loss_fn1, loss_fn2,
                 sample_rate, upscale_factor):
        super().__init__()
        self.D_net    = D_net
        self.G_net    = G_net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **cfg)
            for cfg in self.MEL_SCALES
        ])
        # first mel bin in the generated (SR) band — same for all scales since n_mels=128
        n_mels = self.MEL_SCALES[0]['n_mels']
        self.lr_nyquist_bin = int(n_mels * (sample_rate // upscale_factor) / sample_rate)

    def forward(self, hr, fake_patches, adv_weight=0.001, mel_weight=0.1):
        logits_fake, _ = self.D_net(fake_patches)
        logits_real, _ = self.D_net(hr)

        g_gan_loss = self.loss_fn1(logits_fake, torch.ones_like(logits_fake))

        mse        = self.loss_fn2(fake_patches, hr)
        mse_loss   = torch.mean(mse, dim=[1, 2])
        sqrt_l2    = torch.sqrt(mse_loss)
        avg_sqrt_l2 = torch.mean(sqrt_l2)
        avg_l2      = torch.mean(mse_loss)

        # --- multi-scale mel-band loss ---
        device   = hr.device
        hr_wav   = hr.squeeze(2)          # (B, T)
        fake_wav = fake_patches.squeeze(2)
        mel_loss = 0.0
        for mel_tf in self.mel_transforms:
            mel_tf  = mel_tf.to(device)
            m_real  = mel_tf(hr_wav)      # (B, n_mels, frames)
            m_fake  = mel_tf(fake_wav)
            mel_loss += F.l1_loss(
                m_fake[:, self.lr_nyquist_bin:, :],
                m_real[:, self.lr_nyquist_bin:, :],
            )
        mel_loss = mel_loss / len(self.mel_transforms)

        g_loss = avg_sqrt_l2 + adv_weight * g_gan_loss + mel_weight * mel_loss

        return g_loss, g_gan_loss, avg_sqrt_l2, avg_l2, mel_loss


# ---------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    train_p = subparsers.add_parser('train')
    train_p.set_defaults(func=train)
    train_p.add_argument('--model',           default='gan')
    train_p.add_argument('--train',           required=True)
    train_p.add_argument('--val',             required=True)
    train_p.add_argument('-e', '--epochs',    type=int, default=100)
    train_p.add_argument('--batch_size',      type=int, default=128)
    train_p.add_argument('--logname',         default='tmp-run')
    train_p.add_argument('--layers',          type=int, default=4)
    train_p.add_argument('--alg',             default='adam')
    train_p.add_argument('--lr',              type=float, default=1e-3)
    train_p.add_argument('--r',               type=int, default=3)
    train_p.add_argument('--speaker',         default='multi',
                         choices=('single', 'multi'))
    train_p.add_argument('--pool_size',       type=int, default=4)
    train_p.add_argument('--strides',         type=int, default=4)
    train_p.add_argument('--full',            default='false',
                         choices=('true', 'false'))
    train_p.add_argument('--sr',              type=int, default=48000)
    train_p.add_argument('--patch_size',      type=int, default=8192)
    train_p.add_argument('--mel_loss_weight', type=float, default=0.1,
                         help='weight of the mel-band L1 loss term')
    train_p.add_argument('--resume_epoch',   type=int, default=0,
                         help='resume from this epoch (loads checkpoint for that epoch)')

    return parser


# ---------------------------------------------------------------------------
def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    b1, b2 = 0.9, 0.999
    lr         = args.lr
    batch_size = args.batch_size
    epochs     = args.epochs
    time_dim   = 8192

    # ---- models ----
    discriminator = Discriminator(layers=5, time_dim=time_dim).to(device)
    generator     = Generator(layers=5).to(device)
    generator.train()
    discriminator.train()

    optimizer_G = torch.optim.Adam(generator.parameters(),    lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=lr)

    net_D = WithLoss_D(discriminator, generator, BCEWithSquareLoss)
    net_G = WithLoss_G_melLoss(
        discriminator, generator,
        BCEWithSquareLoss, nn.MSELoss(reduction='none'),
        sample_rate=args.sr, upscale_factor=args.r,
    )

    # ---- data (lazy HDF5 — avoids loading full dataset into RAM) ----
    train_loader = torch.utils.data.DataLoader(
        H5Dataset(args.train),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        H5Dataset(args.val),
        batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    # ---- log paths ----
    save_dir = '../logs'
    os.makedirs(save_dir, exist_ok=True)
    b_str  = f'.b{batch_size}'
    prefix = (f"{args.logname}.r_{args.r}.{args.model}{b_str}"
              f".sr_{args.sr}.melLoss_w{args.mel_loss_weight}")

    loss_file       = os.path.join(save_dir, prefix + '_loss_gan.txt')
    val_loss_file   = os.path.join(save_dir, prefix + '_loss_val_gan.txt')
    train_loss_file = os.path.join(save_dir, prefix + '_loss_train_gan.txt')

    adv_weight = 0.001
    mel_weight = args.mel_loss_weight

    # ---- resume from checkpoint ----
    start_epoch = args.resume_epoch + 1
    if args.resume_epoch > 0:
        gen_path    = os.path.join(save_dir, prefix + f'.generator_gan.epoch_{args.resume_epoch}.pth')
        disc_path   = os.path.join(save_dir, prefix + f'.discriminator_gan.epoch_{args.resume_epoch}.pth')
        opt_g_path  = os.path.join(save_dir, prefix + f'.optimizer_G.epoch_{args.resume_epoch}.pth')
        opt_d_path  = os.path.join(save_dir, prefix + f'.optimizer_D.epoch_{args.resume_epoch}.pth')
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        if os.path.exists(opt_g_path):
            optimizer_G.load_state_dict(torch.load(opt_g_path, map_location=device))
        if os.path.exists(opt_d_path):
            optimizer_D.load_state_dict(torch.load(opt_d_path, map_location=device))
        print(f"Resumed from epoch {args.resume_epoch}")

    gen_ckpt_prev = None

    for epoch_idx in range(start_epoch, epochs + 1):

        # ---- validation ----
        generator.eval()
        discriminator.eval()

        # upsample_wav_train skipped — wav file not available on all systems

        with torch.no_grad():
            def _eval_loop(loader):
                g_loss_acc = g_gan_acc = d1_acc = d2_acc = 0.0
                mse_acc = sqrt_acc = mel_acc = 0.0
                Ys, Ps = [], []
                for lr_s, hr_s in loader:
                    hr_s = hr_s.to(device).float()
                    lr_s = lr_s.to(device).float()
                    fake = generator(lr_s).detach()
                    Ps.append(fake.cpu().numpy().flatten())
                    Ys.append(hr_s.cpu().numpy().flatten())
                    g_loss, g_gan, sqrt_l2, l2, mel_l = net_G(hr_s, fake, adv_weight, mel_weight)
                    loss_d, loss_d1, loss_d2, _        = net_D(hr_s, fake)
                    n = len(loader)
                    g_loss_acc += g_loss.item() / n
                    g_gan_acc  += g_gan / n
                    d1_acc     += loss_d1.item() / n
                    d2_acc     += loss_d2.item() / n
                    mse_acc    += l2 / n
                    sqrt_acc   += sqrt_l2 / n
                    mel_acc    += mel_l.item() / n
                Y_np = np.concatenate(Ys); P_np = np.concatenate(Ps)
                lsd  = get_lsd(P_np, Y_np, n_fft=2048)
                snr  = get_snr(P_np, Y_np)
                return d2_acc, d1_acc, g_loss_acc, g_gan_acc, mse_acc, sqrt_acc, mel_acc, lsd, snr

            val_stats = _eval_loop(val_loader)

        d2, d1, g, gg, mse, sq, mel, lsd, snr = val_stats
        with open(val_loss_file, 'a') as f:
            f.write(f"{epoch_idx}, {d2}, {d1}, {g}, {gg}, {mse}, {sq}, {mel}, {lsd}, {snr}\n")

        # ---- training loop ----
        generator.train()
        discriminator.train()

        for i, (lr_s, hr_s) in enumerate(train_loader):
            hr_s = hr_s.to(device).float()
            lr_s = lr_s.to(device).float()
            fake = generator(lr_s)

            loss_d, loss_d1, loss_d2, d_acc = net_D(hr_s, fake.detach())

            if i % 5 == 0:
                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss, g_gan, sqrt_l2, l2, mel_l = net_G(hr_s, fake, adv_weight, mel_weight)
            g_loss.backward()
            optimizer_G.step()

            with open(loss_file, 'a') as f:
                f.write(f"{epoch_idx}, {loss_d.item()}, {loss_d1.item()}, "
                        f"{loss_d2.item()}, {d_acc.item()}, "
                        f"{g_gan}, {g_loss.item()}, "
                        f"{sqrt_l2}, {l2}, {mel_l.item()}\n")

        scheduler_G.step()
        scheduler_D.step()

        if epoch_idx % 5 == 0:
            gen_path   = os.path.join(save_dir, prefix + f'.generator_gan.epoch_{epoch_idx}.pth')
            disc_path  = os.path.join(save_dir, prefix + f'.discriminator_gan.epoch_{epoch_idx}.pth')
            opt_g_path = os.path.join(save_dir, prefix + f'.optimizer_G.epoch_{epoch_idx}.pth')
            opt_d_path = os.path.join(save_dir, prefix + f'.optimizer_D.epoch_{epoch_idx}.pth')
            torch.save(generator.state_dict(),     gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            torch.save(optimizer_G.state_dict(),   opt_g_path)
            torch.save(optimizer_D.state_dict(),   opt_d_path)
            if gen_ckpt_prev:
                os.remove(gen_ckpt_prev)
                os.remove(gen_ckpt_prev.replace('.generator_gan.', '.discriminator_gan.'))
                for opt_prev in [gen_ckpt_prev.replace('.generator_gan.', '.optimizer_G.'),
                                 gen_ckpt_prev.replace('.generator_gan.', '.optimizer_D.')]:
                    if os.path.exists(opt_prev):
                        os.remove(opt_prev)
            gen_ckpt_prev = gen_path


# ---------------------------------------------------------------------------
def main():
    gc.collect()
    parser = make_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
