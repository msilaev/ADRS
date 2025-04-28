import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import numpy as np
import matplotlib.pyplot as plt
import librosa

PATCH_DIM = 8192

def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='valid')

def get_audio(audio_folder):

    for root, dirs, files in os.walk(audio_folder):

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

            elif "hr" in file:
                print("yes real")
                input_file_path = os.path.join(root, file)
                x_real, fs = librosa.load(input_file_path, sr=48000)

    return x_real, x_audiounet, x_gan, x_hifigan, fs

def eval():

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    audio_folder = "../results/result_listening_test/p347"
    x_real, x_audiounet, x_gan, x_hifigan, fs = get_audio(audio_folder)
    plt_spectrum(x_real, x_audiounet, x_gan, x_hifigan, fs, audio_folder)

    audio_folder = "../results/result_listening_test/p351"
    x_real, x_audiounet, x_gan, x_hifigan, fs = get_audio(audio_folder)
    plt_spectrum(x_real, x_audiounet, x_gan, x_hifigan, fs, audio_folder)

    audio_folder = "../results/result_listening_test/p361"
    x_real, x_audiounet, x_gan, x_hifigan, fs = get_audio(audio_folder)
    plt_spectrum(x_real, x_audiounet, x_gan, x_hifigan, fs, audio_folder)

    audio_folder = "../results/result_listening_test/p376"
    x_real, x_audiounet, x_gan, x_hifigan, fs = get_audio(audio_folder)
    plt_spectrum(x_real, x_audiounet, x_gan, x_hifigan, fs, audio_folder)


def plt_spectrum(x_real, x_audiounet, x_gan, x_hifigan, fs, audio_folder):

    min_len = min(len(x_real), len(x_gan), len(x_hifigan), len(x_audiounet))

    x_real = x_real[:min_len]
    x_gan = x_gan[:min_len]
    x_hifigan = x_hifigan[:min_len]
    x_audiounet = x_audiounet[:min_len]

    freqs = np.fft.rfftfreq(len(x_real), d=1 / fs)  # Frequency axis
    fft_real = np.abs(np.fft.rfft(x_real))
    fft_gan = np.abs(np.fft.rfft(x_gan))
    fft_hifigan = np.abs(np.fft.rfft(x_hifigan))
    fft_audiounet = np.abs(np.fft.rfft(x_audiounet))

    fft_real_smooth = moving_average(fft_real, w=20)
    fft_gan_smooth = moving_average(fft_gan, w=20)
    fft_hifigan_smooth = moving_average(fft_hifigan, w=20)
    fft_audiounet_smooth = moving_average(fft_audiounet, w=20)

    freqs_smooth = freqs[:len(fft_real_smooth)]  # Adjust frequency axis

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_smooth/1000, fft_real_smooth, label="Real")
    plt.plot(freqs_smooth/1000, fft_gan_smooth, label="GAN")
    plt.plot(freqs_smooth/1000, fft_hifigan_smooth, label="HiFi-GAN")
    plt.plot(freqs_smooth/1000, fft_audiounet_smooth, label="AudioUNet")
    #plt.xscale("log")  # Log-scale for better visualization

    plt.rcParams.update({'font.size': 25})  # General font size
    plt.rcParams.update({'axes.titlesize': 25})  # Title font size
    plt.rcParams.update({'axes.labelsize': 25})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 25})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 25})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 25})  # Y tick label font size


    plt.yscale("log")  # Optional: log-scale for magnitude
    plt.xlabel("Frequency (kHz)", fontsize = 25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel("Magnitude", fontsize = 25)

    plt.legend(loc="lower left", fontsize = 20)  # Use 'upper right' instead of 'northeast'

    plt.title("Spectrum, speaker " +  audio_folder.split("/")[-1]  )
    plt.grid()
    plt.tight_layout()
    plt.savefig("../results/result_listening_test/"+audio_folder.split("/")[-1] + ".png")
#    plt.show()

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

    eval()

if __name__ == '__main__':
  main()
