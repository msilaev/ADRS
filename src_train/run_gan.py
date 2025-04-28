import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

from models.audiounet import AudioUNet

from models.gan import Generator, Discriminator, BCEWithSquareLoss
#from calculate_snr_lsd import get_lsd, get_snr

import torch
import torch.nn as nn
from dataset_batch_norm import BatchData
from models.io import load_h5, upsample_wav
import torch.optim as optim
from torchinfo import summary
import numpy as np

class WithLoss_D(nn.Module):

    def __init__(self, D_net, G_net, loss_fn):
        super(WithLoss_D, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn
        self.sigmoid = nn.Sigmoid()

    def forward (self, hr, fake_patches ):

        logits_fake, fmap_fake = self.D_net(fake_patches)
        logits_real, fmap_real = self.D_net(hr)

        d_loss1 =  self.loss_fn(logits_real, torch.ones_like(logits_real))
        d_loss2 = self.loss_fn(logits_fake, torch.zeros_like(logits_fake))

        d_loss = (d_loss1 + d_loss2)

        d_real = self.sigmoid(logits_real)
        d_fake = self.sigmoid(logits_fake)

        d_real = d_real.squeeze()
        d_fake = d_fake.squeeze()

        d_real_acc = torch.ge(d_real, 0.5).float()
        d_fake_acc = torch.le(d_fake, 0.5).float()

        d_acc = torch.mean( torch.cat( (d_real_acc, d_fake_acc), 0 ) )

        return d_loss, d_loss1, d_loss2, d_acc

###################################################
class WithLoss_G_new(nn.Module):

    def __init__(self, D_net, G_net, loss_fn1, loss_fn2,  autoencoder = None):

        super(WithLoss_G_new, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.autoencoder = autoencoder
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def feature_loss(self, fmap_r, fmap_g):

        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


    def loss_artifacts(self, lr, hr, n_fft=2048):
        sr = 16000
        # Create a time array from 0 to the length of lr
        time = np.arange(len(lr))  # Correctly initialize the time array

        # Create the harmonic tensor
        harm_4000 = torch.tensor(np.sin(2 * np.pi * 4000 * time / sr), dtype=torch.float32, device=lr.device)

        harm_3000 = torch.tensor(np.sin(2 * np.pi * 3000 * time / sr), dtype=torch.float32, device=lr.device)

        harm_5000 = torch.tensor(np.sin(2 * np.pi * 5000 * time / sr), dtype=torch.float32, device=lr.device)

        harm_6000 = torch.tensor(np.sin(2 * np.pi * 6000 * time / sr), dtype=torch.float32, device=lr.device)

        harm_3500 = torch.tensor(np.sin(2 * np.pi * 3500 * time / sr), dtype=torch.float32, device=lr.device)

        # Compute the spectral representations
        #S_lr = torch.abs(harm_3000* lr)
        #S_hr = torch.abs(harm_3000 * hr)

        # Calculate MSE loss without reduction
        loss_3000 = nn.MSELoss(reduction='none')(torch.abs(harm_3000* lr), torch.abs(harm_3000 * hr))
        mse_3000 = torch.mean(torch.sqrt(torch.mean(loss_3000, dim=[1, 2])), dim=0)

        # Calculate MSE loss without reduction
        loss_4000 = nn.MSELoss(reduction='none')(torch.abs(harm_4000 * lr), torch.abs(harm_3000 * hr))
        mse_4000 = torch.mean(torch.sqrt(torch.mean(loss_4000, dim=[1, 2])), dim=0)

        loss_5000 = nn.MSELoss(reduction='none')(torch.abs(harm_5000* lr), torch.abs(harm_3000 * hr))
        mse_5000 = torch.mean(torch.sqrt(torch.mean(loss_5000, dim=[1, 2])), dim=0)

        loss_6000 = nn.MSELoss(reduction='none')(torch.abs(harm_6000* lr), torch.abs(harm_3000 * hr))
        mse_6000 = torch.mean(torch.sqrt(torch.mean(loss_6000, dim=[1, 2])), dim=0)

        loss_3500 = nn.MSELoss(reduction='none')(torch.abs(harm_3500* lr), torch.abs(harm_3000 * hr))
        mse_3500 = torch.mean(torch.sqrt(torch.mean(loss_3500, dim=[1, 2])), dim=0)
        # Return the mean loss
        return (mse_3500 + mse_3000 + mse_4000 + mse_5000 + mse_6000 )/5

    # Example usage
    # lr = torch.randn(1, 16000)  # Simulated low-resolution signal
    # hr = torch.randn(1, 16000)  # Simulated high-resolution signal
    # loss_value = loss_artifacts(lr, hr)
    # print("Loss value:", loss_value.item())

    def forward(self, hr, fake_patches, adv_weight = 0.001):

        logits_fake, fmap_fake = self.D_net(fake_patches)

        g_gan_loss = self.loss_fn1(logits_fake, torch.ones_like(logits_fake))

        mse = self.loss_fn2(fake_patches, hr)

        mse_loss = torch.mean(mse, dim=[1,2])

        sqrt_l2_loss = torch.sqrt(mse_loss)

        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        avg_l2_loss = torch.mean(mse_loss, dim=0)

        autoencoder_loss = None

        #feature_loss = 0.1* calculate_feature_loss(fake_patches, hr)
        #feature_loss = 0.005 * calculate_mel_loss(fake_patches, hr)
        feature_loss = self.loss_artifacts(fake_patches, hr)

        if self.autoencoder is not None:

            feature_real = self.autoencoder(hr)
            feature_fake = self.autoencoder(fake_patches)

            autoencoder_loss = \
                torch.mean(torch.sqrt( torch.mean(self.loss_fn2(feature_fake, feature_real), dim =[1,2])), dim =0)

            g_loss = avg_sqrt_l2_loss + adv_weight * g_gan_loss + 10*autoencoder_loss

        else:
            g_loss = avg_sqrt_l2_loss + adv_weight * g_gan_loss + feature_loss #+ spectral_loss

        return g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, feature_loss

# -----------------------------------------------------------------
def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--model', default='gan',
    choices=('gan', 'gan_simple'), help='model to train')
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  train_parser.add_argument('--batch_size', type=int, default=128,
    help='training batch size')
  train_parser.add_argument('--logname', default='tmp-run',
    help='folder where logs will be stored')
  train_parser.add_argument('--layers', default=4, type=int,
    help='number of layers in each of the D and U halves of the network')
  train_parser.add_argument('--alg', default='adam',
    help='optimization algorithm')
  train_parser.add_argument('--lr', default=1e-3, type=float,
    help='learning rate')
  train_parser.add_argument('--r', type=int, default=4,
                            help='upscaling factor')
  train_parser.add_argument('--speaker', default='single',
                            choices=('single', 'multi'),
    help='number of speakers being trained on')
  train_parser.add_argument('--pool_size', type=int, default=4,
                            help='size of pooling window')
  train_parser.add_argument('--strides', type=int, default=4,
                            help='pooling stide')
  train_parser.add_argument('--full', default='false',
                            choices=('true', 'false'))
  train_parser.add_argument('--sr', help='high-res sampling rate',
                           type=int, default=48000)
  train_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')

  # eval
  eval_parser = subparsers.add_parser('eval')
  eval_parser.add_argument('--batch_size', type=int, default=16,
    help='training batch size')
  eval_parser.set_defaults(func=eval)
  eval_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  eval_parser.add_argument('--logname', required=True,
    help='path to training checkpoint')
  eval_parser.add_argument('--out_label', default='',
    help='append label to output samples')
  eval_parser.add_argument('--layers', default= 4,
                           help='number of layers')
  eval_parser.add_argument('--wav_file_list',
    help='list of audio files for evaluation')
  eval_parser.add_argument('--r', help='upscaling factor',
                           default = 4, type=int)
  eval_parser.add_argument('--sr', help='high-res sampling rate',
                                   type=int, default=48000)
  eval_parser.add_argument('--model', default='audiounet')
   # choices=('gan', 'gan_multispeaker', 'gan_multiD', 'audiounet_multispeaker',
   #          'gan_audiounet', 'audiounet', 'gan_multispeaker_bs_128'), help='model to train')
  eval_parser.add_argument('--speaker', default='single', choices=('single', 'multi'),
    help='number of speakers being trained on')
  eval_parser.add_argument('--pool_size', type=int, default=4,
                           help='size of pooling window')
  eval_parser.add_argument('--strides', type=int, default=4,
                           help='pooling stide')
  eval_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')

  return parser

# --------
# training gan
# --------
def train( args):
    torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #--------
    # learning parameters
    #--------
    b1 = 0.9
    b2 = 0.999
    lr = args.lr

    batch_size = args.batch_size
    epochs = args.epochs
    epochs_init = 10

    time_dim = 8192

    #d_loss_thresh = 0.6

    # --------
    # Initialize the model
    # --------
    num_layers_disc = 5
    discriminator = Discriminator(layers = num_layers_disc,
                                  time_dim = time_dim).to(device)

    model_path = os.path.join("logs", "multispeaker", "sr16000", "logsGAN",
                              "singlespeaker.r_4.gan.b128.sr_16000.discriminator_autoencoder_gan.epoch_180.pth")

    summary(discriminator, input_size=(16, 8192, 1))
    input()

    #state_dict = torch.load(model_path, map_location=device)
    #discriminator.load_state_dict(state_dict)

    num_layers_gen = 5
    #generator = GeneratorDilConv(layers=num_layers_gen).to(device)
    #generator = Generator(layers=num_layers_gen).to(device)

    #summary(generator, input_size=(16, 8192, 1))
    #input()

    #model_path = os.path.join("logs", "multispeaker", "sr16000", "logsGAN",
    #                          "singlespeaker.r_4.gan.b128.sr_16000.generator_autoencoder_gan.epoch_180.pth")

    #state_dict = torch.load(model_path, map_location=device)
    #enerator.load_state_dict(state_dict)

    generator = AudioUNet(layers = 4).to(device)
    summary(generator, input_size=(16, 8192, 1))
    input()

    #example_input = torch.randn(1, 1, 8192).to(device)  # e.g., for a mono audio signal of 1-second at 16kHz
    #autoencoder = GeneratorFeatures(layers=num_layers_gen).to(device)
    #model_path = os.path.join("logs", "feature_gen_180.pth")
    #state_dict = torch.load(model_path, map_location=device)
    #autoencoder.load_state_dict(state_dict)

    generator.train()
    discriminator.train()
    #autoencoder.eval()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas=(b1, b2))

    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = lr, momentum=0.9)

    scheduler_G = \
        optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
                                             T_max = epochs, eta_min = lr)

    scheduler_D = \
        optim.lr_scheduler.CosineAnnealingLR(optimizer_D,
                                                       T_max = epochs, eta_min = lr)

    net_with_loss_D = \
        WithLoss_D(discriminator, generator, BCEWithSquareLoss)

    net_with_loss_G = \
        WithLoss_G_new(discriminator, generator, BCEWithSquareLoss,
                                 nn.MSELoss(reduction='none'), autoencoder = None)

    # --------
    # define data loader
    # --------
    print(args.train)
    print(args.val)
    X_train, Y_train = load_h5(args.train)

    X_val, Y_val = load_h5(args.val)

    lr_mean, lr_std = X_train.mean(), X_train.std()
    hr_mean, hr_std = Y_train.mean(), Y_train.std()

    lr_mean = 0
    lr_std = 1
    hr_mean = 0
    hr_std = 1

    dataset = BatchData(X_train, Y_train,  lr_mean, lr_std, hr_mean, hr_std)

    train_loader = \
        torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True, drop_last = True)

    dataset_val = BatchData(X_val, Y_val, lr_mean, lr_std, hr_mean, hr_std)

    val_loader = \
        torch.utils.data.DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    shuffle=False, drop_last=True)

    print("Train DS", len(train_loader)*batch_size)
    print("Validation DS", len(val_loader) * batch_size)

    #input()

    checkpoint_root_prev = None
    gen_checkpoint_root_prev = None
    det_checkpoint_root_prev = None

    ###########
    save_dir = 'logs'
    os.makedirs(save_dir, exist_ok=True)

    b_str = '.b%d' % int(args.batch_size)
    model_name = args.model
    log_prefix = args.logname

    loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                    + f".sr_{args.sr}" + "_loss_autoencoder_gan" + ".txt"

    loss_filename_init = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                    + f".sr_{args.sr}" + "_init_loss" + ".txt"

    loss_filename_full = os.path.join(save_dir, loss_filename)

    loss_filename_init_full = os.path.join(save_dir, loss_filename_init)

    val_loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                    + f".sr_{args.sr}" + "_loss_val_autoencoder_gan" + ".txt"

    train_loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                        + f".sr_{args.sr}" + "_loss_train_autoencoder_gan" + ".txt"

    val_loss_filename_full = os.path.join(save_dir, val_loss_filename)
    train_loss_filename_full = os.path.join(save_dir, train_loss_filename)

    # --------
    # initialize the generator model
    # --------
    init_checkpoint_filename = log_prefix + '.r_%d.' % args.r + \
                               model_name + b_str \
                               + f".sr_{args.sr}" + f".gen_init.pth"

    init_checkpoint_filename_full = \
        os.path.join(save_dir, init_checkpoint_filename)

    generator.eval()  # Set the model to evaluation mode
    discriminator.eval()

    # -----------------
    # Train GAN model
    # -----------------
    initial_weight = 0.001
    decay_rate = 200
    for epoch_idx in range(45, epochs + 1):

        #adv_weight = initial_weight * np.exp(-epoch_idx / decay_rate)
        
        adv_weight = initial_weight

        # loss_str  = get_validation(generator, discriminator, net_with_loss_G,
        #                           net_with_loss_D, adv_weight, val_loader,
        #                           epoch_idx,  device)

        # with open(val_loss_filename_full, "a") as f_write_loss:

        # f_write_loss.write(loss_str)

        ########################################
        generator.train()
        discriminator.train()

        for i, (lr_sound, hr_sound) in enumerate(train_loader, 0):

            hr_sound, lr_sound = hr_sound.to(device).float(), lr_sound.to(device).float()

            lr_sound_norm = (lr_sound - lr_mean) / lr_std
            hr_sound_norm = (hr_sound - hr_mean) / hr_std

            fake_patches = generator(lr_sound_norm)

            loss_d, loss_d1, loss_d2, d_real_acc = \
                net_with_loss_D(hr_sound_norm, fake_patches.detach())

            if (i%1 == 0):

            # ------------
            # Train Discriminator
            # ------------
                #if d_real_acc < d_loss_thresh:

                    optimizer_D.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10.0)
                    #optimizer_D.step()

                    # Compute the total norm of the gradients before clipping
                    total_norm = 0.0
                    for p in discriminator.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)  # Compute the L2 norm for this parameter
                            total_norm += param_norm.item() ** 2  # Accumulate the squared norm

                    total_norm = total_norm ** 0.5  # Take the square root to get the total L2 norm

                    #print(f"Gradient norm before clipping: {total_norm}")

                    # Now perform gradient clipping
                    #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                    # Perform the optimizer step
                    optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            # No detach for fake_sound here, therefore when
            # calculating backward, parts of the loss function
            # with fake_sound contribute to the update of the Generator weights

            # g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, feature_loss

            #summary(generator, input_data=lr_sound)
            #input()

            optimizer_G.zero_grad()

            g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, feature_loss = \
                net_with_loss_G(hr_sound_norm, fake_patches, adv_weight)

            # g_loss = avg_sqrt_l2_loss + adv_weight * g_gan_loss + feature_loss #+ spectral_loss

            g_loss.backward()
            optimizer_G.step()

            print("gen loss", g_loss.item(), g_gan_loss.item(), feature_loss.item(), avg_sqrt_l2_loss.item())
            print("disc loss", loss_d.item(), loss_d1.item(), loss_d2.item())
            with open(loss_filename_full, "a") as f_write_loss:

                #g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, feature_loss
                #g_loss = avg_sqrt_l2_loss + adv_weight * g_gan_loss + feature_loss

                loss_str = f"{epoch_idx}, {loss_d.item()}, {loss_d1}, {loss_d2},  " \
                           f"{adv_weight*g_gan_loss}, {g_loss.item()}, " \
                           f"{avg_sqrt_l2_loss}, {feature_loss}\n"

                f_write_loss.write(loss_str)

        scheduler_G.step()
        scheduler_D.step()

        gen_checkpoint_filename = \
            log_prefix + '.r_%d.' % args.r + model_name + b_str \
                          + f".sr_{args.sr}" + \
            f".generator_autoencoder_gan.epoch_{epoch_idx}.pth"

        gen_checkpoint_filename_full = os.path.join(save_dir, gen_checkpoint_filename)

        det_checkpoint_filename = \
            log_prefix + '.r_%d.' % args.r + model_name + b_str \
                              + f".sr_{args.sr}" + \
            f".discriminator_autoencoder_gan.epoch_{epoch_idx}.pth"

        det_checkpoint_filename_full = os.path.join(save_dir, det_checkpoint_filename)

        # ------------
        # Writing checkpoint
        # ------------
        if (epoch_idx % 5 == 0):

            torch.save(generator.state_dict(), gen_checkpoint_filename_full)
            if gen_checkpoint_root_prev is not None:
                os.remove(gen_checkpoint_root_prev)
            gen_checkpoint_root_prev = gen_checkpoint_filename_full

            torch.save(discriminator.state_dict(), det_checkpoint_filename_full)
            if det_checkpoint_root_prev is not None:
                os.remove(det_checkpoint_root_prev)
            det_checkpoint_root_prev = det_checkpoint_filename_full

def eval(args):

  #print("eval")

  #input()

  batch_size = args.batch_size

  X_val, Y_val = load_h5(args.val)

  lr_mean = 0
  lr_std = 1
  hr_mean = 0
  hr_std = 1

  dataset_val = BatchData(X_val, Y_val, lr_mean, lr_std, hr_mean, hr_std)

  val_loader = torch.utils.data.DataLoader(dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False, drop_last=True)

  if args.model in ["gan",  "gan_multispeaker",  "gan_multispeaker_bs_128",
                    "gan_singlespeaker", "gan_GenF_multispeaker",
                    "gan_16_Clipping_multispeaker", "gan_WideDil_multispeaker",
                    "gan_WideAlt_multispeaker", "gan_alt_5_multispeaker",
                    "gan_alt_3_multispeaker", "gen_dec", "gen_s_multispeaker"]:

      #input()
      model = Generator(layers=5)

  elif args.model in ["audiounet",  "audiounet_multispeaker", "audiounet_singlespeaker" ]:

      model = AudioUNet(layers=4)

  model.eval()

  checkpoint_root = args.logname

  # uncomment this is calculation of the total SNR is needed. It can take quite long for large dataset.

  #lsd_val_kuleshov, lsd_val, snr_val, lsd_val_spline, snr_val_spline, lsd_val_avg, snr_val_avg = \
  #    eval_snr_lsd(model, val_loader, model_path=checkpoint_root)

  #metrics_filename_full = os.path.join("logs", "metrics_summary.txt")

  #with open(metrics_filename_full, "a") as f_write_metrics:
  #    metrics_str = f"{args.model}, {lsd_val_kuleshov}, {lsd_val}, {snr_val}" \
  #               f"{lsd_val_spline}, {snr_val_spline}, " \
  #               f"{lsd_val_avg}, {snr_val_avg}\n"

#      f_write_metrics.write(metrics_str)

  #upsample_wav(model, '../data/vctk/VCTK-Corpus/wav48/p376/p376_240.wav',
  #             args, epoch=None, model_path=checkpoint_root)

  if args.wav_file_list:
    with open(args.wav_file_list) as f:
      for line in f:

        #print(line)
        #input()

        try:
          if(args.speaker == 'single'):

            upsample_wav(model, '../data/vctk/VCTK-Corpus/wav48/p225/'+line.strip(),
                         args, epoch= None, model_path=checkpoint_root)

          else:

            #u = np.random.uniform()
            #if u < 0.01:
            upsample_wav(model, '../data/vctk'+line.strip().split("..")[1],
                         args, epoch= None, model_path=checkpoint_root)

        except EOFError:
          print('WARNING: Error reading file:', line.strip())

def get_validation(generator, discriminator, net_with_loss_G, net_with_loss_D,
                   adv_weight, val_loader, epoch_idx,  device):
    # ---------------------
    # ----- validation loop
    # ---------------------
    generator.eval()  # Set the model to evaluation mode
    discriminator.eval()

    ######################################3
    with torch.no_grad():  # Disable gradient calculation

        g_loss_val = 0
        g_gan_loss_val = 0
        d_gen_loss_val = 0
        d_discr_loss_val = 0
        mse_loss_val = 0
        sqrt_mse_loss_val = 0
        feature_loss_val = 0

        Y = []
        P = []

        for i, (lr_sound, hr_sound) in enumerate(val_loader, 0):
            hr_sound, lr_sound = hr_sound.to(device).float(), \
                                 lr_sound.to(device).float()

            generator.zerograd()

            fake_patches = generator(lr_sound).detach()

            P.append(fake_patches.cpu().numpy().flatten())
            Y.append(hr_sound.cpu().numpy().flatten())

            g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, feature_loss = \
                net_with_loss_G(hr_sound, fake_patches, adv_weight)

            loss_d, loss_d1, loss_d2, d_real_acc = \
                net_with_loss_D(hr_sound, fake_patches)

            d_gen_loss_val += loss_d2.item() / len(val_loader)
            d_discr_loss_val += loss_d1.item() / len(val_loader)

            g_loss_val += g_loss.item() / len(val_loader)
            g_gan_loss_val += g_gan_loss / len(val_loader)
            mse_loss_val += avg_l2_loss / len(val_loader)
            sqrt_mse_loss_val += avg_sqrt_l2_loss / len(val_loader)
            feature_loss_val += feature_loss.item() / len(val_loader)

            print(g_gan_loss, loss_d1, loss_d2)

        Y = np.concatenate(Y)
        P = np.concatenate(P)

        lsd_val = get_lsd(P, Y, n_fft=2048)
        snr_val = get_snr(P, Y)

        return  f"{epoch_idx}, {d_gen_loss_val}, {d_discr_loss_val}, " \
                       f"{g_loss_val}, {g_gan_loss_val}, " \
                       f"{mse_loss_val}, {sqrt_mse_loss_val}, {lsd_val}, {snr_val}\n"

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  #print("main")
  main()
