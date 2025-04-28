import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

import argparse
from models.audiounet import AudioUNet
from models.audiotfilm import AudioTfilm
from models.gan import  BCEWithSquareLoss, MelDiscriminator, STFTDiscriminator, STFTDiscriminator4class

from models.gan import CustomPipeline

import torch.optim as optim
from dataset_batch_norm import BatchData
from torch.utils.data import Dataset, DataLoader
from models.io import load_h5, upsample_wav, H5Dataset
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from torch.utils.data import ConcatDataset

PATCH_DIM = 8192

class WithLoss_D_4class(nn.Module):

    def __init__(self, D_net, loss_fn):
        super(WithLoss_D_4class, self).__init__()
        self.D_net = D_net
        self.loss_fn = loss_fn

    def forward(self, x, y):

        logits, fmap = self.D_net(x)

        # Ensure y has the same shape as logits
        #y = y.view_as(logits)
        y = y.view(-1).long()  # Shape: [batch_size]

        # Compute loss
        d_loss = self.loss_fn(logits, y)

        # Compute accuracy
        predictions = torch.argmax(logits, dim =1)
        d_acc = torch.mean((predictions == y).float())

        return d_loss, d_acc

# -----------------------------------------------------------------
def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--model', default='gan',
    choices=('gan', 'audiounet', 'hifigan'), help='model to train')
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  train_parser.add_argument('--batch_size', type=int, default=32,
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
  train_parser.add_argument('--discriminator')

  return parser

def train( args):

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = args.batch_size
    time_dim = 8192

    if args.discriminator == "conv":
        num_layers_disc = 1
        classificator = Discriminator(layers = num_layers_disc,
                                  time_dim = time_dim).to(device)

    elif args.discriminator == "LR":
        classificator = Discriminator_LR(time_dim = time_dim).to(device)

    elif args.discriminator == "stft":
        classificator = STFTDiscriminator4class(layers = 1).to(device)

    summary(classificator, input_size=(16, time_dim, 1))
    save_dir = '../logs_4class'

    model_path = os.path.join(save_dir,
                              "singlespeaker.r_3.audiounet.b128.sr_48000.discriminator_audiounet_stft.epoch_10.pth")

    state_dict = torch.load(model_path, map_location=device)
    classificator.load_state_dict(state_dict)


    X_val, Y_val = load_h5(args.val)
    print("valdataset loaded", len(X_val))

    dataset_val = BatchData(X_val, Y_val, lr_mean =0, lr_std=1, hr_mean=0, hr_std=1)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)

    dataset1 = H5Dataset(args.train, start_idx=0, end_idx=len(dataset_val)*5)
    dataset2 = H5Dataset(args.train, start_idx=len(dataset_val)*5,
                         end_idx=len(dataset_val)*9)

    combined_dataset = ConcatDataset([dataset1, dataset2])
    save_dir = '../logs_4class'
    os.makedirs(save_dir, exist_ok=True)

    b_str = '.b%d' % int(args.batch_size)
    model_name = args.model
    log_prefix = args.logname

    classificator.eval()
    path = "../data/vctk/listening_test/result/p347"

    with torch.no_grad():  # Disable gradient calculation

        for root, dirs, files in os.walk(path):

            for file in files:
                print(file)

                if "audiounet" in file:
                    print("yes audiounet")
                    input_file_path = os.path.join(root, file)
                    x_audiounet, fs = librosa.load(input_file_path, sr=48000)

                elif "gan" in file and not ("hifigan" in file):
                    print("yes gan")
                    input_file_path = os.path.join(root, file)
                    x_gan, fs = librosa.load(input_file_path, sr=48000)

                elif "hifigan" in file:
                    print("yes hifigan")
                    input_file_path = os.path.join(root, file)
                    x_hifigan, fs = librosa.load(input_file_path, sr=48000)

                else:
                    print("yes real")
                    input_file_path = os.path.join(root, file)
                    x_real, fs = librosa.load(input_file_path, sr=48000)

            min_len = min(len(x_real), len(x_gan), len(x_hifigan), len(x_audiounet), 2*48000)

            x_real = x_real[:min_len]
            x_gan = x_gan[:min_len]
            x_hifigan = x_hifigan[:min_len]
            x_audiounet = x_audiounet[:min_len]

            #############################
            x_real_stft = get_spectrum(x_real, n_fft=3 * 2048)
            pipeline = CustomPipeline(48000)
            pipeline.to(dtype=torch.float32)

            sound_audio_tensor = torch.tensor(x_real).squeeze()
            sound_audio_tensor = sound_audio_tensor.unsqueeze(0)
            x_real_auditorial = pipeline.forward(sound_audio_tensor).squeeze()

            #############################
            x_gan_stft = get_spectrum(x_gan, n_fft=3 * 2048)
            pipeline = CustomPipeline(48000)
            pipeline.to(dtype=torch.float32)

            sound_audio_tensor = torch.tensor(x_gan).squeeze()
            sound_audio_tensor = sound_audio_tensor.unsqueeze(0)
            x_gan_auditorial = pipeline.forward(sound_audio_tensor).squeeze()

            #############################
            x_audiounet_stft = get_spectrum(x_audiounet, n_fft=3 * 2048)
            pipeline = CustomPipeline(48000)
            pipeline.to(dtype=torch.float32)

            sound_audio_tensor = torch.tensor(x_audiounet).squeeze()
            sound_audio_tensor = sound_audio_tensor.unsqueeze(0)
            x_audiounet_auditorial = pipeline.forward(sound_audio_tensor).squeeze()

            #print(x_real_auditorial.shape)
            #print(x_real_stft.shape)
            ####################################################

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 6, 1)
            plt.imshow(x_real_stft, aspect='auto', cmap='viridis')
            plt.subplot(1, 6, 2)
            plt.imshow(x_real_auditorial, aspect='auto', cmap='viridis')

            plt.subplot(1, 6, 3)
            plt.imshow(x_gan_stft, aspect='auto', cmap='viridis')
            plt.subplot(1, 6, 4)
            plt.imshow(x_gan_auditorial, aspect='auto', cmap='viridis')

            plt.subplot(1, 6, 5)
            plt.imshow(x_audiounet_stft, aspect='auto', cmap='viridis')
            plt.subplot(1, 6, 6)
            plt.imshow(x_audiounet_auditorial, aspect='auto', cmap='viridis')
            plt.show()

            #plt.imshow(pipeline.forward(sound_audio_tensor).squeeze(), aspect='auto', cmap='Blues')
            #def moving_average(x, w=5):
            #    return np.convolve(x, np.ones(w) / w, mode='valid')

            #freqs = np.fft.rfftfreq(len(x_real), d=1 / 48000)  # Frequency axis
            #fft_real = np.abs(np.fft.rfft(x_real))
            #fft_gan = np.abs(np.fft.rfft(x_gan))
            #fft_hifigan = np.abs(np.fft.rfft(x_hifigan))
            #fft_audiounet = np.abs(np.fft.rfft(x_audiounet))

            #fft_real_smooth = moving_average(fft_real, w=20)
            #fft_gan_smooth = moving_average(fft_gan, w=20)
            #fft_hifigan_smooth = moving_average(fft_hifigan, w=20)
            #fft_audiounet_smooth = moving_average(fft_audiounet, w=20)

            #freqs_smooth = freqs[:len(fft_real_smooth)]  # Adjust frequency axis

            # Plot
            #plt.figure(figsize=(10, 5))
            #plt.plot(freqs_smooth, fft_real_smooth, label="Real")
            #plt.plot(freqs_smooth, fft_gan_smooth, label="GAN")
            #plt.plot(freqs_smooth, fft_hifigan_smooth, label="HiFi-GAN")
            #plt.plot(freqs_smooth, fft_audiounet_smooth, label="AudioUNet")
            ##plt.xscale("log")  # Log-scale for better visualization
            #plt.yscale("log")  # Optional: log-scale for magnitude
            #plt.xlabel("Frequency (Hz)")
            #plt.ylabel("Magnitude")
            #plt.legend()
            #plt.title("FFT Magnitude Spectrum")
            #plt.grid()
            #plt.show()

        #for i in range(len(x_real)//PATCH_DIM):

        #    x_audiounet_patch = x_audiounet[i : i + PATCH_DIM]
        #    x_gan_patch = x_gan[i : i + PATCH_DIM]
        #    x_hifigan_patch = x_hifigan[i : i + PATCH_DIM]
        #    x_real_patch = x_real[i : i + PATCH_DIM]

        #for i, (sound, label) in enumerate(val_loader, 0):

            #sound, label = sound.to(device).float(), label.to(device).float()
        #    logits_audiounet, fmap_audiounet, linear_weights = classificator(x_audiounet_patch)
        #    logits_gan, fmap_gan, linear_weights = classificator(x_gan_patch)
        #    logits_hifigan, fmap_hifigan, linear_weights = classificator(x_hifigan_patch)
        #    logits_real, fmap_real, linear_weights = classificator(x_real_patch)

        #    # Extract weights and biases
        #    weights = linear_weights.weight.detach().cpu().numpy()  # Shape: (4, 256)
        #    biases = linear_weights.bias.detach().cpu().numpy()  # Shape: (4,)

        #    # Reshape assumption: fmap.shape must be (16, 16) for a correct reshape
        #    fmap_shape = (fmap.shape[2], fmap.shape[3]) # Adjust based on your feature map size

        #    #print(torch.argmax(logits_audiounet, dim =1)[0].item())
        #    print(f"audiounet {logits_audiounet.item()}")
        #    print(f"logits_gan {logits_gan.item()}")
        #    print(f"logits_hifigan {logits_hifigan.item()}")
        #    print(f"logits_real {logits_real.item()}")

        #    #sample = fmap[0, 0].cpu().numpy()
        #    # Shape: (64, 4)

        #    print(fmap_audiounet.shape)

        #    # Plot as a heatmap
        #    sound_stft = sound.cpu().numpy()[0]
        #    sound_stft = np.squeeze(sound_stft)

        #    sound_stft = get_spectrum(sound_stft, n_fft=3 * 2048)

        #    plt.figure(figsize=(12, 6))

        #    for i in range(4):  # 4 output neurons

        #        plt.subplot(1, 4, i + 1)

        #        plt.plot(np.sum( weights[i].reshape(fmap_shape),  axis =1)*np.sum(sample, axis=1))
        #        plt.plot(np.sum(sample, axis=1))
        #        plt.plot(5*np.sum(weights[i].reshape(fmap_shape), axis=1) )
        #        plt.ylim([-30, 30])

#            plt.suptitle("Linear Layer Weight Visualization")
#            plt.tight_layout()
#            plt.show()

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def save_spectrum(S, sr, hop_length, outfile='spectrogram.png', type = "high resolution"):
    # Create a smaller figure with reduced size
    plt.figure(figsize=(5, 5))  # Adjust the figure size for smaller paper size

    # Set font sizes globally
    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    # Plot the spectrogram with larger labels
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr//8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr//8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    #cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    #plt.title('Spectrogram', fontsize=16)
    #plt.title(type, fontsize=15)

    # Calculate the maximum time based on the number of frames in S
    max_time = S.shape[1] * hop_length / sr  # Convert frames to seconds
    plt.xlim(0, max_time)  # Limit x-axis to the range of the data
    plt.xticks(ticks=[i for i in range(int(max_time) + 1) if i <= max_time])

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major')  # Major ticks
    plt.tick_params(axis='both', which='minor')  # Minor ticks

    # Use tight layout for better spacing and save the figure
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')  # High DPI for better quality
    #print(outfile)
    plt.close()


def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
