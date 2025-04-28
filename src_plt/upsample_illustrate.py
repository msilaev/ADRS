import numpy as np
import sys
import os
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from scipy.signal import resample
import librosa
from scipy.signal import decimate
from scipy import interpolate
from models.io import  upsample
from models.upsample_4_48 import upsample_wav_4_48_1

from models.gan import Generator

import argparse

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
    #cbar = plt.colorbar(format='%+2.0f dB')
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

# Example usage
# save_spectrum(S, sr, hop_length, outfile='spectrogram.png')

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)
    return x_sp

def make_parser():

       parser = argparse.ArgumentParser()
       parser.add_argument('--log_16', required=True,
                    help='path to training checkpoint')

       parser.add_argument('--log_48', required=True,
                    help='path to training checkpoint')

       parser.add_argument('--out_label', default='',
                    help='append label to output samples')

       parser.add_argument('--wav_file_list',
                    help='list of audio files for evaluation')

       parser.add_argument('--r_16', help='upscaling factor',
                    default=4, type=int)

       parser.add_argument('--r_48', help='upscaling factor',
                    default=4, type=int)

       parser.add_argument('--sr', help='high-res sampling rate',
                    type=int, default=16000)

       parser.add_argument('--model', default='audiounet')

       parser.add_argument('--speaker', default='single', choices=('single', 'multi'),
                    help='number of speakers being trained on')

       parser.add_argument('--patch_size', type=int, default=8192,
                    help='Size of patches over which the model operates')

       return parser

def plt_upsample(args):

       model_1 = Generator(layers=5)
       model_2 = Generator(layers=5)  # AudioUNet(layers=4) # Generator(layers=5)
       model_1.eval()
       model_2.eval()

       wav = '/worktmp/THESES/AudioEnhanceSupervised/' \
              'data/vctk/VCTK-Corpus/wav48/p376/p376_001.wav'


       wav1 = '/worktmp/THESES/AudioEnhanceSupervised/' \
              'data/vctk/VCTK-Corpus/wav48/p361/p361_001.wav'


       r = args.r_48 * args.r_16
       x_hr, fs = librosa.load(wav, sr=48000)

       x_hr_1, fs = librosa.load(wav1, sr=48000)

       pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
       x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

       x_hr, x_lr_4, x_pr_4_48, x_pr_16_48, x_pr_4_16, x_spline_4_16, x_hr_16 = \
              upsample_wav_4_48_1(model_1, model_2, x_hr, args, model_path_1=args.log_16,
                           model_path_2=args.log_48)

       t_hr_48 = np.linspace(0, len(x_hr) / fs, num=len(x_hr))
       t_lr_4 = t_hr_48[0: len(x_lr_4)*r : r]
       x_lr_upsample_4_48 = upsample(x_lr_4, r)
       t_lr_upsample_4_48 = np.linspace(0, len(x_lr_upsample_4_48) / fs, num=len(x_lr_upsample_4_48))
       t_pr = np.linspace(0, len(x_pr_4_48) / fs, num=len(x_pr_4_48))

       #print(x_lr_4.shape, t_lr_4.shape, fs)

       t_hr_16 = np.linspace(0, args.r_48 * len(x_hr_16) / fs, num=len(x_hr_16))

       x_lr_upsample_4_16 = upsample(x_lr_4, args.r_16)

       t_lr_upsample_4_16 = np.linspace(0,  args.r_48*len(x_lr_upsample_4_16) / fs, num=len(x_lr_upsample_4_16))
       t_pr_16 = np.linspace(0, args.r_48 *len(x_pr_4_16) / fs, num=len(x_pr_4_16))

       t_pr_16_48 = np.linspace(0, len(x_pr_16_48) / fs, num=len(x_pr_16_48))

############################################33
       plt.figure(figsize=(12, 5))

       plt.rcParams.update({'font.size': 20})  # General font size
       plt.rcParams.update({'axes.titlesize': 20})  # Title font size
       plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
       plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
       plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
       plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

       plt.figure
       #plt.plot(t_hr, x_hr, color = "blue")
       #plt.plot(t_hr_48, x_hr, marker='o', markersize = 6, linestyle='', label = "hr, 48 KHz", color = "blue" )
       plt.plot(t_hr_48, x_hr, linewidth = "2", linestyle='-', label="WB, 48 KHz", color="blue")

       #plt.plot(t_lr, x_lr, color = "red")
       #plt.plot(t_lr_4, x_lr_4, marker='o', markersize = 10, linestyle='', label = "NB, 4 KHz", color = "red")

       plt.plot(t_hr_48[::3], x_hr[::3],  marker='o', markersize=10, linestyle='',  label = "NB, 16 KHz", color = "red")

       #plt.plot(t_hr, x_lr_upsample, color = "black")
       #plt.plot(t_lr_upsample_4_48, x_lr_upsample_4_48, marker='+', markersize = 6, linestyle='', label = "spline", color = "black")
       #plt.plot(t_lr_upsample_4_48, x_lr_upsample_4_48, linewidth = "2",linestyle='--', label="spline", color="black")
       #plt.plot(t_pr, x_pr_4_48, marker='*', markersize=10, linestyle='', label="GAN16*GAN48", color="magenta")
       #plt.plot(t_pr, x_pr_4_48, linestyle="-", linewidth = "2", label="GAN16*GAN48", color="magenta")
       #plt.plot(t_lr_upsample, x_lr_upsample, marker='+', markersize=6, linestyle='', label="spline, 48 KHz", color="black")

       plt.plot(t_pr_16_48, x_pr_16_48, linestyle="-", linewidth = "2", label="MU-GAN", color="magenta")

       #plt.xlim([1.845, 1.85])

       #plt.title('Original (sr = 48 kHz) and decimated (sr = 4 kHz) Signals')
       plt.xlabel('Time [s]')
       plt.ylabel('Sound amplitude, a.u.')
       plt.grid(True)
       plt.legend()
       plt.tight_layout()

       file_name = wav.split("/")[-1]
       figure_file_name = "../results/illustrations/" + file_name
       plt.savefig(figure_file_name + ".png", format='png')

       #plt.show()

########################################333
       plt.figure(figsize=(6, 5))

       plt.rcParams.update({'font.size': 20})  # General font size
       plt.rcParams.update({'axes.titlesize': 20})  # Title font size
       plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
       plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
       plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
       plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

       plt.plot(t_hr_16, x_hr_16, marker='*', markersize=8, linestyle='--', linewidth = "1", label="WB, 16 KHz", color="blue")

       plt.plot(t_lr_upsample_4_16, x_lr_upsample_4_16, marker='+', markersize=8, linestyle="-", linewidth = "1",color = "black", label="spline")

       #plt.plot(t_pr_16, x_pr_4_16, marker='*', markersize=8, linestyle="--", linewidth = "2", label="MU-GAN", color="magenta")

       plt.plot(t_lr_4, x_lr_4, marker='o', markersize=10, linestyle='--', label="NB, 4 KHz", color="red")

       plt.xlim([1.845, 1.847])

       # plt.title('Original (sr = 48 kHz) and decimated (sr = 4 kHz) Signals')
       plt.xlabel('Time [s]')
       plt.ylabel('Sound amplitude, a.u.')
       plt.grid(True)
       plt.legend()
       plt.tight_layout()

       file_name = wav.split("/")[-1]
       figure_file_name = "../results/illustrations/" + file_name
       plt.savefig(figure_file_name + "_16_spline.png", format='png')

       #plt.show()

       ########################################333
       plt.figure(figsize=(6, 5))

       plt.rcParams.update({'font.size': 20})  # General font size
       plt.rcParams.update({'axes.titlesize': 20})  # Title font size
       plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
       plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
       plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
       plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

       plt.plot(t_hr_16, x_hr_16, marker='*', markersize=8, linestyle='--', linewidth="1", label="WB, 16 KHz",
                color="blue")

       #plt.plot(t_lr_upsample_4_16, x_lr_upsample_4_16, marker='*', markersize=8, linestyle="--", linewidth="2",
       #         color="black", label="spline")

       plt.plot(t_pr_16, x_pr_4_16, marker='+', markersize=8, linestyle="-", linewidth="1", label="MU-GAN",
                color="black")

       plt.plot(t_lr_4, x_lr_4, marker='o', markersize=10, linestyle='-.', label="NB, 4 KHz", color="red")

       plt.xlim([1.845, 1.847])

       # plt.title('Original (sr = 48 kHz) and decimated (sr = 4 kHz) Signals')
       plt.xlabel('Time [s]')
       plt.ylabel('Sound amplitude, a.u.')
       plt.grid(True)
       plt.legend()
       plt.tight_layout()

       file_name = wav.split("/")[-1]
       figure_file_name = "../results/illustrations/" + file_name
       plt.savefig(figure_file_name + "_16.png", format='png')

       #plt.show()

       save_spectrum(get_spectrum(x_hr_1, n_fft=3 * 2048), sr=fs, hop_length=3 * 2048 // 4,
                     outfile= figure_file_name +'_spectrogram_48.png')

       sf.write(figure_file_name + '_spectrogram_48.wav', x_hr_1, fs)

       save_spectrum(get_spectrum(librosa.resample(decimate(x_hr_1,3), orig_sr=16000, target_sr =48000), n_fft=3 * 2048), sr=fs, hop_length=3 * 2048 // 4,
                     outfile=figure_file_name + '_spectrogram_16.png')

       sf.write(figure_file_name + '_spectrogram_16.wav',
                librosa.resample(decimate(x_hr_1,3), orig_sr=16000, target_sr =48000), fs)


       save_spectrum(
           get_spectrum(librosa.resample(decimate(x_hr_1, 12), orig_sr=4000, target_sr=48000), n_fft=3 * 2048), sr=fs,
           hop_length=3 * 2048 // 4,
           outfile=figure_file_name + '_spectrogram_4.png')

       sf.write(figure_file_name + '_spectrogram_4.wav',
                librosa.resample(decimate(x_hr_1, 12), orig_sr=4000, target_sr=48000), fs)


def main():
       parser = make_parser()
       args = parser.parse_args()
       plt_upsample(args)

if __name__ == "__main__":
       main()