import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gan import Generator

import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
from models.io import upsample
from scipy.signal import decimate

# Helper function to perform FFT and plot the frequency response
def plot_frequency_response(original_signal, processed_signal, sampling_rate):

    original_signal = torch.from_numpy(original_signal)
    processed_signal = torch.from_numpy(processed_signal)


    # Compute FFT for both signals
    original_fft = torch.fft.fft(original_signal).abs().numpy()
    processed_fft = torch.fft.fft(processed_signal).abs().numpy()

    freqs = np.fft.fftfreq(len(original_fft), 1 / sampling_rate)
    freqs1 = np.fft.fftfreq(len(original_fft), 1 / sampling_rate)

    sorted_indices = np.argsort (freqs)
    sorted_indices1 = np.argsort(freqs1)

    original_fft = original_fft[sorted_indices]
    processed_fft = processed_fft[sorted_indices1]

    freqs = freqs[sorted_indices]
    freqs1 = freqs1[sorted_indices1]

    print(freqs)
    print(processed_fft.shape)

    # Plot the magnitude spectrum
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    #plt.plot(freqs, original_fft
    plt.plot(freqs, (original_fft))
    plt.grid()
    plt.xlim([0,8000])
    plt.ylim([0, 100])

    plt.title("Original Signal - Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(1, 2, 2)
    plt.plot(freqs1, (processed_fft))
    plt.title("Processed Signal - Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.xlim([0, 8000])
    plt.ylim([0, 100])

    plt.tight_layout()
    plt.show()

    display_spectrum( get_spectrum( (original_signal.detach() ).numpy() ),
                      sr= 16000, hop_length=2048//4, type = "high resolution")

    display_spectrum( get_spectrum( (processed_signal.detach() ).numpy()),
                      sr= 16000, hop_length=2048//4, type = "processed by model")

def get_spectrum(x, n_fft=2048):

  S = librosa.stft(x, n_fft = n_fft)
  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def display_spectrum(S, sr, hop_length, type = "high resolution"):
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
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr // 8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr // 8)])

    # Add a color bar with larger font size
    cbar = plt.colorbar(format='%+2.0f dB')
    #cbar.ax.tick_params(labelsize=15)  # Adjust the color bar tick labels size

    # Set title and axis labels with larger font size
    # plt.title('Spectrogram', fontsize=16)
    plt.title(type)
    plt.xlabel('Time (s)',)
    plt.ylabel('Frequency (kHz)')

    max_time = S.shape[1] * hop_length / sr  # Convert frames to seconds
    plt.xlim(0, max_time)  # Limit x-axis to the range of the data
    plt.xticks(ticks=[i for i in range(int(max_time) + 1) if i <= max_time])

    # Adjust tick parameters for both axes
    plt.tick_params(axis='both', which='major')  # Major ticks
    plt.tick_params(axis='both', which='minor')  # Minor ticks

    # Use tight layout for better spacing and save the figure
    plt.tight_layout()
    plt.show()

# Generate a sinusoidal signal for testing
def generate_sinusoidal_signal(frequency, sampling_rate, duration, amplitude=1.0):

    t = torch.arange(0, duration, 1 / sampling_rate)  # Time axis
    signal = amplitude * torch.sin(2 * np.pi * frequency * t)  # Sinusoidal signal
    signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return signal

def inference_wav(model, wav, r, sr, patch_size, model_path = None):

    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    if (model_path is not None):
    #    # Load the model
    #    device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    model.to(device)

    # Load signal
    x_hr, fs = librosa.load(wav, sr=int(sr))

    # Pad to multiple of patch size to ensure model runs over entire sample
    pad_length = patch_size - (x_hr.shape[0] % patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    x_lr = decimate(x_hr, r)
    x_lr_1 = upsample(x_lr, r)

    n_patches = x_lr_1.shape[0]//patch_size

    x_lr = librosa.resample(x_lr, orig_sr=fs // r, target_sr=fs)
    #x_lr = x_lr[:len(x_lr) - (len(x_lr) % (2 ** (layers + 1)))]

    P = []
    Y = []
    X = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

        #P = model(x_lr_tensor_part.to(device)).squeeze().numpy()
            lr_patch = np.array(x_lr_1[i * patch_size : (i+1)* patch_size ])

            lr_patch_1 = np.array(x_lr[i * patch_size: (i + 1) * patch_size])

            hr_patch = np.array(x_hr[i * patch_size: (i+1) * patch_size ])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            x_hr_tensor_part = torch.tensor(hr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            x_pr_part = model(x_lr_tensor_part.to(device)).squeeze().cpu().numpy()

            P.append( x_pr_part)
            #Y.append(model(x_hr_tensor_part.to(device)).squeeze().cpu().numpy())
            Y.append(hr_patch)

            X.append(lr_patch_1)

            #display_spectrum(get_mel_spectrum(x_pr_part, n_fft=1 * 2048), sr=args.sr, hop_length = 512,  type=wav)

    P = np.concatenate(P)
    Y = np.concatenate(Y)
    X = np.concatenate(X)

    return P, Y, X

def plt_wav(x_init, x_pr, sr):
    ########################################333

    t = np.arange(len(x_pr))/sr
    plt.figure(figsize=(12, 5))

    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    plt.figure

    plt.plot(t, x_init, color="blue")
    plt.plot(t, x_pr, marker='o', markersize=6, linestyle='', label="hr, 16 KHz", color="blue")

    plt.xlim([1.837, 1.841])

    plt.xlabel('Time [s]')
    plt.ylabel('Sound amplitude, a.u.')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    #file_name = wav.split("/")[-1]
    #figure_file_name = "../results/illustrations/" + file_name
    #plt.savefig(figure_file_name + "_16.png", format='png')

    plt.show()

# Parameters
sampling_rate = 16000  # Hz
duration = 3.50        # seconds
frequency = 1900      # Hz (example frequency)

# Generate the signal

sinusoidal_signal =0* 0.1*generate_sinusoidal_signal(frequency, sampling_rate, duration)

for frequency in range(1,2000,10):
    sinusoidal_signal = sinusoidal_signal + \
                        0.1 * generate_sinusoidal_signal(frequency, sampling_rate, duration)

model = Generator(layers=5)

line = "p362_147.wav"

input_file_path = '../data/vctk/VCTK-Corpus/wav48/p362/' + line
r=4
sr=16000
patch_size = 8192

P, Y, X = inference_wav(model, input_file_path, r, sr, patch_size, model_path=None)

#print(sinusoidal_signal.shape)
# Initialize SuperPixel1D layer
#super_pixel = SuperPixel1D(r=2)

model.eval()
# Process the signal through SuperPixel1D
processed_signal = model(sinusoidal_signal.transpose(1,2))

# Plot the frequency response
#plot_frequency_response(processed_signal.squeeze(), processed_signal.squeeze(), sampling_rate)
plot_frequency_response(X.flatten(), P.flatten(), sr)