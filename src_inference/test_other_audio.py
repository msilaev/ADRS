import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import librosa
import matplotlib.pyplot as plt
import numpy as np

from models.gan import Generator
from models.audiounet import AudioUNet
from scipy import interpolate
from models.io import inference_wav
import soundfile as sf
import argparse

# -----------------------------------------------------------------
def make_parser():

  parser = argparse.ArgumentParser()
    # train
  parser.add_argument('--model')
  parser.add_argument('--logname')
  parser.add_argument('--output')
  parser.add_argument('--data')
  parser.add_argument('--sr')
  parser.add_argument('--patch_size', type=int, default=8192,
                            help='Size of patches over which the model operates')
  parser.add_argument('--r', type=int, default=3,
                            help='upscaling factor')
  parser.add_argument('--layers', default=4, type=int,
                            help='number of layers in each of the D and U halves of the network')

  return parser

def eval(args):

  if args.model in [ "gan_multispeaker"]:

    model = Generator(layers=5)

  elif args.model in ["audiounet_multispeaker"]:

    model = AudioUNet(layers=4)

  model.eval()

  checkpoint_root = args.logname
  output_dir = args.output
  directory = args.data

  for root, dirs, files in os.walk(directory):
      for file in files:

          if file.endswith(".flac") or file.endswith(".wav"):

              input_file_path = os.path.join(root, file)

              input_file_path_wav = os.path.join(root, file.split(".")[0])

              relative_path_wav = os.path.relpath(input_file_path_wav, directory)

              output_file_path = os.path.join(output_dir, relative_path_wav)

              output_dir_1 = os.path.dirname(output_file_path)
              os.makedirs(output_dir_1, exist_ok = True)

              #print(input_file_path)
              #print(output_file_path)
              #print(file)

              #input()

              P, Y, X = inference_wav(model, input_file_path,
                                      args, epoch=None, model_path=checkpoint_root)

              x_pr = P.flatten()

              sf.write(output_file_path + '.wav', x_pr, 48000)

def main():
    parser = make_parser()
    args = parser.parse_args()
    eval(args)

# ----------------------------------------------------------------------------
def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def save_spectrum(S, sr, hop_length, outfile='spectrogram.png', type = "high resolution"):
    # Create a smaller figure with reduced size
    plt.figure(figsize=(6, 3))  # Adjust the figure size for smaller paper size

    # Plot the spectrogram with larger labels
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr//8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr//8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    #plt.title('Spectrogram', fontsize=16)
    plt.title(type, fontsize=15)
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Frequency (kHz)', fontsize=15)

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major', labelsize=15)  # Major ticks
    plt.tick_params(axis='both', which='minor', labelsize=15)  # Minor ticks

    # Use tight layout for better spacing and save the figure
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')  # High DPI for better quality
    plt.close()

# Example usage
# save_spectrum(S, sr, hop_length, outfile='spectrogram.png')


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp



if __name__ == "__main__":
    main()