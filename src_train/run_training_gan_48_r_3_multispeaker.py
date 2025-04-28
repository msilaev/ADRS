import gc

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

from models.gan import Generator, Discriminator, BCEWithSquareLoss

from models.calculate_snr_lsd import get_lsd, get_snr

import torch
import torch.nn as nn

from dataset_batch_norm import BatchData
from models.io import load_h5, upsample_wav_train
import torch.optim as optim
import numpy as np

class WithLoss_init(nn.Module):

    def __init__(self, G_net, loss_fn):
        super(WithLoss_init, self).__init__()

        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        out = self.G_net(lr)
        loss = self.loss_fn(out, hr)
        rmse = torch.sqrt(loss)  # Take the square root of the MSE
        avg_rmse = torch.mean(rmse)  # Average over the batch dimension
        return avg_rmse

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

class WithLoss_G_new(nn.Module):

    def __init__(self, D_net, G_net, loss_fn1, loss_fn2, autoencoder = None):

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

    def forward(self, hr, fake_patches, adv_weight = 0.001):

        logits_fake, fmap_fake = self.D_net(fake_patches)
        logits_real, fmap_real = self.D_net(hr)

        g_gan_loss = self.loss_fn1(logits_fake, torch.ones_like(logits_fake))

        mse = self.loss_fn2(fake_patches, hr)

        feature = 0.001*self.feature_loss(fmap_fake, fmap_real)

        mse_loss = torch.mean(mse, dim=[1,2])
        feature_loss = torch.mean(feature, dim=[0])

        sqrt_l2_loss = torch.sqrt(mse_loss)

        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        avg_l2_loss = torch.mean(mse_loss, dim=0)

        autoencoder_loss = None

        if self.autoencoder is not None:

            feature_real = self.autoencoder.get_features(hr)
            feature_fake = self.autoencoder.get_features(fake_patches)
            autoencoder_loss = \
                torch.mean(torch.sqrt( torch.mean(self.loss_fn2(feature_fake, feature_real), dim =[1,2])), dim =0)

            g_loss = avg_sqrt_l2_loss + autoencoder_loss + adv_weight * g_gan_loss

        else:
            g_loss = avg_sqrt_l2_loss + adv_weight * g_gan_loss + 0*feature_loss

        return g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, autoencoder_loss, feature_loss

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
  eval_parser.set_defaults(func=eval)

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
  eval_parser.add_argument('--model', default='audiounet',
    choices=('gan', 'gen_init', 'gan_multiD', 'gan_multiD_features',
             'gan_audiounet'), help='model to train')
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

    # --------
    # Initialize the model
    # --------
    num_layers_disc = 5
    discriminator = Discriminator(layers = num_layers_disc,
                                  time_dim = time_dim).to(device)

    #model_path = os.path.join("../logs",
    # "singlespeaker.r_3.gan.b128.sr_48000.discriminator_gan.epoch_495.pth")
     
    #state_dict = torch.load(model_path, map_location=device)
    #discriminator.load_state_dict(state_dict)

    num_layers_gen = 5
    generator = Generator(layers = num_layers_gen).to(device)

    #model_path = os.path.join("../logs",
    # "singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_495.pth")
      
    #state_dict = torch.load(model_path, map_location=device)
    #generator.load_state_dict(state_dict)

    generator.train()
    discriminator.train()

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
    X_train, Y_train = load_h5(args.train)

    X_val, Y_val = load_h5(args.val)

    dataset = BatchData(X_train, Y_train,  lr_mean=0, lr_std=1, hr_mean=0, hr_std=1)

    train_loader = \
        torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True, drop_last = True)

    dataset_val = BatchData(X_val, Y_val,  lr_mean=0, lr_std=1, hr_mean=0, hr_std=1)

    val_loader = \
        torch.utils.data.DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    shuffle=False, drop_last=True)

    gen_checkpoint_root_prev = None
    det_checkpoint_root_prev = None

    ###########
    save_dir = '../logs'
    os.makedirs(save_dir, exist_ok=True)

    b_str = '.b%d' % int(args.batch_size)
    model_name = args.model
    log_prefix = args.logname

    loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                    + f".sr_{args.sr}" + "_loss_gan" + ".txt"

    loss_filename_full = os.path.join(save_dir, loss_filename)

    val_loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                    + f".sr_{args.sr}" + "_loss_val_gan" + ".txt"

    train_loss_filename = log_prefix + '.r_%d.' % args.r + model_name + b_str \
                        + f".sr_{args.sr}" + "_loss_train_gan" + ".txt"

    val_loss_filename_full = os.path.join(save_dir, val_loss_filename)
    train_loss_filename_full = os.path.join(save_dir, train_loss_filename)

    # -----------------
    # Train GAN model
    # -----------------
    initial_weight = 0.001
    for epoch_idx in range(1, epochs + 1):

        adv_weight = initial_weight

        #---------------------
        #----- validation loop
        #---------------------
        generator.eval()  # Set the model to evaluation mode
        discriminator.eval()
        
        line = "p362_147.wav"
        upsample_wav_train(generator,
                     '../data/vctk/VCTK-Corpus/wav48/p362/' + line,
                     args, epoch_idx)

        with torch.no_grad():  # Disable gradient calculation

            g_loss_val = 0
            g_gan_loss_val = 0
            d_gen_loss_val = 0
            d_discr_loss_val = 0
            mse_loss_val = 0
            sqrt_mse_loss_val = 0

            Y = []
            P = []

            for i, (lr_sound, hr_sound) in enumerate(val_loader, 0):

                hr_sound, lr_sound = hr_sound.to(device).float(), \
                                     lr_sound.to(device).float()

                fake_patches = generator(lr_sound).detach()

                P.append(fake_patches.cpu().numpy().flatten())
                Y.append(hr_sound.cpu().numpy().flatten())

                g_loss, g_gan_loss, avg_sqrt_l2_loss, \
                avg_l2_loss, autoencoder_loss, feature_loss = \
                    net_with_loss_G(hr_sound, fake_patches, adv_weight)

                loss_d, loss_d1, loss_d2, d_real_acc = \
                net_with_loss_D(hr_sound, fake_patches)
                
                d_gen_loss_val += loss_d2.item()/len(val_loader)
                d_discr_loss_val += loss_d1.item()/len(val_loader)

                g_loss_val += g_loss.item()/len(val_loader)
                g_gan_loss_val += g_gan_loss/len(val_loader)
                mse_loss_val += avg_l2_loss/len(val_loader)
                sqrt_mse_loss_val += avg_sqrt_l2_loss / len(val_loader)

            Y = np.concatenate(Y)
            P = np.concatenate(P)

            lsd_val = get_lsd(P, Y, n_fft = 2048)
            snr_val = get_snr(P, Y)

            g_loss_train = 0
            g_gan_loss_train = 0
            d_gen_loss_train = 0
            d_discr_loss_train = 0
            mse_loss_train = 0
            sqrt_mse_loss_train = 0

            Y = []
            P = []

            for i, (lr_sound, hr_sound) in enumerate(train_loader, 0):

                hr_sound, lr_sound = hr_sound.to(device).float(), \
                                 lr_sound.to(device).float()

                fake_patches = generator(lr_sound).detach()

                P.append(fake_patches.cpu().numpy().flatten())
                Y.append(hr_sound.cpu().numpy().flatten())

                g_loss, g_gan_loss, avg_sqrt_l2_loss, \
                avg_l2_loss, autoencoder_loss, feature_loss = \
                    net_with_loss_G(hr_sound, fake_patches, adv_weight)

                loss_d, loss_d1, loss_d2, d_real_acc = \
                    net_with_loss_D(hr_sound, fake_patches)

                d_gen_loss_train += loss_d2.item() / len(train_loader)
                d_discr_loss_train += loss_d1.item() / len(train_loader)

                g_loss_train += g_loss.item() / len(train_loader)
                g_gan_loss_train += g_gan_loss / len(train_loader)
                mse_loss_train += avg_l2_loss / len(train_loader)
                sqrt_mse_loss_train += avg_sqrt_l2_loss / len(train_loader)

            Y = np.concatenate(Y)
            P = np.concatenate(P)

            lsd_train = get_lsd(P, Y, n_fft = 2048)
            snr_train = get_snr(P, Y)

        ############################################
        with open(val_loss_filename_full, "a") as f_write_loss:

                loss_str = f"{epoch_idx}, {d_gen_loss_val}, {d_discr_loss_val}, " \
                           f"{g_loss_val}, {g_gan_loss_val}, " \
                           f"{mse_loss_val}, {sqrt_mse_loss_val}, {lsd_val}, {snr_val}\n"

                f_write_loss.write(loss_str)

        ############################################
        with open(train_loss_filename_full, "a") as f_write_loss:

                loss_str = f"{epoch_idx}, {d_gen_loss_train}, {d_discr_loss_train}, " \
                           f"{g_loss_train}, {g_gan_loss_train}, " \
                           f"{mse_loss_train}, {sqrt_mse_loss_train}, {lsd_train}, {snr_train}\n"

                f_write_loss.write(loss_str)

        generator.train()
        discriminator.train()

        for i, (lr_sound, hr_sound) in enumerate(train_loader, 0):

            hr_sound, lr_sound = hr_sound.to(device).float(), lr_sound.to(device).float()

            lr_sound_norm = lr_sound
            hr_sound_norm = hr_sound

            fake_patches = generator(lr_sound_norm)

            loss_d, loss_d1, loss_d2, d_real_acc = \
                net_with_loss_D(hr_sound_norm, fake_patches.detach())


            if (i%1 == 0):

            # ------------
            # Train Discriminator
            # ------------
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            # No detach for fake_sound here, therefore when
            # calculating backward, parts of the loss function
            # with fake_sound contribute to the update of the Generator weights

            optimizer_G.zero_grad()

            g_loss, g_gan_loss, avg_sqrt_l2_loss, avg_l2_loss, autoencoder_loss, feature_loss = \
                net_with_loss_G(hr_sound_norm, fake_patches, adv_weight)

            g_loss.backward()
            optimizer_G.step()

            with open(loss_filename_full, "a") as f_write_loss:

                loss_str = f"{epoch_idx}, {loss_d.item()}, {loss_d1}, " \
                               f"{loss_d2},  {d_real_acc}, " \
                           f"{g_gan_loss}, {g_loss.item()}, " \
                           f"{autoencoder_loss}, {avg_sqrt_l2_loss}, " \
                           f"{avg_l2_loss}, {feature_loss}\n"

                f_write_loss.write(loss_str)

        scheduler_G.step()
        scheduler_D.step()

        gen_checkpoint_filename = \
            log_prefix + '.r_%d.' % args.r + model_name + b_str \
                          + f".sr_{args.sr}" + \
            f".generator_gan.epoch_{epoch_idx}.pth"

        gen_checkpoint_filename_full = os.path.join(save_dir, gen_checkpoint_filename)

        det_checkpoint_filename = \
            log_prefix + '.r_%d.' % args.r + model_name + b_str \
                              + f".sr_{args.sr}" + \
            f".discriminator_gan.epoch_{epoch_idx}.pth"

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

  pass

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
