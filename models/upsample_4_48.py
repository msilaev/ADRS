import os
import numpy as np
import h5py
import torch
import librosa
import soundfile as sf
from scipy import interpolate
import scipy.signal as signal
from scipy.signal import decimate
import matplotlib.pyplot as plt

from .calculate_snr_lsd import get_lsd, get_snr, get_lsd_kuleshov

#from torchmetrics.audio import SignalImprovementMeanOpinionScore
# ---------------------------------------------------------------

def load_h5(h5_path):
  # load training data
  #print(h5_path)
  with h5py.File(h5_path, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))

  return X, Y


def upsample_wav_4_48_1(model_1, model_2, x_hr,
                     args, model_path_1 = None, model_path_2 = None):

    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    state_dict = torch.load(model_path_1, map_location=device)
    model_1.load_state_dict(state_dict)
    model_1.eval()
    model_1.to(device)

    state_dict = torch.load(model_path_2, map_location=device)
    model_2.load_state_dict(state_dict)
    model_2.eval()
    model_2.to(device)

    # Downscale signal
    x_lr_4 = decimate(decimate(x_hr, args.r_48), args.r_16)

    x_hr_16 = decimate(x_hr, args.r_48)

    x_spline_16_48 = upsample(decimate(x_hr, args.r_48), args.r_48)

    x_spline_4_16 = upsample(x_lr_4, args.r_16)
    #x_lr_16_1= x_spline_4_16

    n_patches = x_spline_4_16.shape[0]//args.patch_size

    P = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

            lr_patch = np.array(x_spline_4_16[i * args.patch_size : (i+1)* args.patch_size ])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P.append((model_1 ( x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

    x_pr_4_16 = (np.concatenate(P)).flatten()

    #x_pr_4_16 = x_lr_1

    x_pr_4_16_spline_16_48 = upsample(x_pr_4_16, args.r_48)

    n_patches = x_pr_4_16_spline_16_48.shape[0] // args.patch_size

    P = []
    P_16 = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

            lr_patch = np.array(x_pr_4_16_spline_16_48[i * args.patch_size: (i + 1) * args.patch_size])

            lr_patch_16 = np.array(x_spline_16_48[i * args.patch_size: (i + 1) * args.patch_size])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P.append((model_2(x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

            x_lr_tensor_part = torch.tensor(lr_patch_16.flatten(),
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P_16.append((model_2(x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

    x_pr_4_48 = (np.concatenate(P)).flatten()

    x_pr_16_48 = (np.concatenate(P_16)).flatten()

    return x_hr, x_lr_4, x_pr_4_48, x_pr_16_48, x_pr_4_16, x_spline_4_16, x_hr_16

def upsample_wav_4_48(model_1, model_2, wav,
                     args, model_path_1 = None, model_path_2 = None):

    if torch.cuda.is_available():
            print("CUDA!")
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    state_dict = torch.load(model_path_1, map_location=device)
    model_1.load_state_dict(state_dict)
    model_1.eval()
    model_1.to(device)

    state_dict = torch.load(model_path_2, map_location=device)
    model_2.load_state_dict(state_dict)
    model_2.eval()
    model_2.to(device)

    # Load signal args.sr = 48000
    x_hr, fs = librosa.load(wav, sr=args.sr)

    pad_length = args.patch_size - (x_hr.shape[0] % args.patch_size)
    x_hr = np.pad(x_hr, (0, pad_length), 'constant', constant_values=(0, 0))

    # Downscale signal
    x_lr = decimate(decimate(x_hr, args.r_48), args.r_16)

    x_lr_16 = upsample(decimate(x_hr, args.r_48), args.r_48)

    x_lr_1 = upsample(x_lr, args.r_16)

    n_patches = x_lr_1.shape[0]//args.patch_size

    P = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

            lr_patch = np.array(x_lr_1[i * args.patch_size : (i+1)* args.patch_size ])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P.append((model_1 ( x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

    x_lr_1 = (np.concatenate(P)).flatten()

    x_lr_1 = upsample(x_lr_1, args.r_48)

    n_patches = x_lr_1.shape[0] // args.patch_size

    P = []
    P_16 = []

    with torch.no_grad():
        for i in range(0, n_patches, 1):

            lr_patch = np.array(x_lr_1[i * args.patch_size: (i + 1) * args.patch_size])

            lr_patch_16 = np.array(x_lr_16[i * args.patch_size: (i + 1) * args.patch_size])

            x_lr_tensor_part = torch.tensor(lr_patch.flatten(),
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P.append((model_2(x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

            x_lr_tensor_part = torch.tensor(lr_patch_16.flatten(),
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(2)

            P_16.append((model_2(x_lr_tensor_part.to(device))).squeeze().cpu().numpy())

    x_pr = (np.concatenate(P)).flatten()

    x_pr_16 = (np.concatenate(P_16)).flatten()

    x_lr = x_lr_1

    save_results(args, fs, x_hr, x_lr, x_pr, x_pr_16, x_lr_16, wav)


def save_results(args, fs, x_hr, x_lr, x_pr, x_pr_16, x_lr_16, wav):

    output_dir_spectrograms = '../results/spectrograms16/samples/' + args.model + '.sr_16.r_' + str(args.r_16) + "/"
    output_dir_audio = '../results/audio16/samples/' + args.model + '.sr_16.r_' + str(args.r_16) + "/"

    outname_spectrograms = output_dir_spectrograms + wav.split('/')[-1]
    outname_audio = output_dir_audio + wav.split('/')[-1]

    sf.write(outname_audio + '.r' + str(args.r_16) + '.lr.wav', x_lr, args.sr)
    sf.write(outname_audio + '.hr.wav', x_hr, fs)
    sf.write(outname_audio + '.r' + str(args.r_16) +
                 '.' + str(args.model) + '.pr.wav', x_pr, args.sr)

    save_spectrum(get_spectrum(x_hr, n_fft=3 * 2048), sr=args.sr, hop_length=3 * 2048 // 4,
                      outfile=outname_spectrograms + '.hr.png', type='high resolution')

    save_spectrum(get_spectrum(x_lr, n_fft=3 * 2048), sr=args.sr, hop_length=3 * 2048 // 4,
                      outfile=outname_spectrograms + '.r' + str(args.r_16) +
                              '.sr' + str(args.sr) + '.lr.png', type='low resolution')

    save_spectrum(get_spectrum(x_lr_16, n_fft=3 * 2048), sr=args.sr, hop_length=3 * 2048 // 4,
                      outfile=outname_spectrograms + '.r' + str(args.r_16) +
                              '.sr' + str(args.sr) + '.spline.png', type='upsampled, spline')

    save_spectrum(get_spectrum(x_pr, n_fft=3 * 2048), sr=args.sr, hop_length=3 * 2048 // 4,
                      outfile=outname_spectrograms + '.r' + str(args.r_16) +
                              '.' + str(args.model) +
                              '.sr' + str(args.sr) + '.pr.png',
                      type='upsampled ' + str(args.model))

def filter_artifacts(x_pr, sr):

    # Design a notch filter at 4 kHz
    Q = 30.0  # Quality factor (controls the bandwidth of the notch)


    freq = 4000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 3000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    freq = 5000  # Frequency to remove (4 kHz)
    b, a = signal.iirnotch(freq, Q, sr)
    x_pr = signal.filtfilt(b, a, x_pr)

    return x_pr


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp


# -----------------------------------------------------------
def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft = n_fft)
  p = np.angle(S)
  #S = np.log1p(np.abs(S))

  S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

  return S_dB

def save_spectrum_1(S, lim=800, outfile='spectrogram.png'):
#  plt.imshow(S.T, aspect=10)
  plt.imshow(S, aspect='auto', origin='lower')
  # plt.xlim([0,lim])
  plt.tight_layout()
  plt.savefig(outfile)


def save_spectrum_2(S, sr, hop_length, outfile='spectrogram.png'):
    plt.figure(figsize=(10, 6))
    # Plot the spectrogram with proper axes
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


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

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)
    return x_sp


def eval_snr_lsd(generator, val_loader, model_path):


    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    #device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    generator.to(device)

    Y = []
    P = []
    X = []

    lsd_val_avg = []
    snr_val_avg = []

    lsd_val_avg_librosa = []
    snr_val_avg_librosa = []

    for i, (lr_sound, hr_sound) in enumerate(val_loader, 0):

        hr_sound, lr_sound = hr_sound.to(device).float(), \
                             lr_sound.to(device).float()

        fake_patches = generator(lr_sound).detach()

        P1 = fake_patches.cpu().numpy().flatten()
        Y1 = hr_sound.cpu().numpy().flatten()
        X1 = lr_sound.cpu().numpy().flatten()

        P.append(P1)
        Y.append(Y1)
        X.append(X1)

        #print(get_snr (P1, Y1), get_snr(X1, Y1))
        lsd = get_lsd(P1, Y1, n_fft = 2048)
        snr = get_snr (P1, Y1)

        #print("lsd_snr", lsd, snr)

        lsd_val_avg.append( lsd)
        snr_val_avg.append( snr)

        lsd_val_avg_librosa.append(get_lsd(X1, Y1, n_fft=2048))
        snr_val_avg_librosa.append(get_snr(X1, Y1))

#########################################
        x_pr = P1.flatten()
        x_hr = Y1.flatten()
        x_init = X1.flatten()

        #plt.plot(x_hr, color="blue")
       # plt.plot(x_hr, color="green")
        #plt.plot(x_pr, color="red")
        #plt.show()

############################################

    Y = np.concatenate(Y)
    P = np.concatenate(P)
    X = np.concatenate(X)

    lsd_val = get_lsd(P, Y, n_fft=2048)
    lsd_val_kuleshov = get_lsd_kuleshov(P, Y)

    snr_val = get_snr(P, Y)

    lsd_val_spline = get_lsd(X, Y, n_fft=2048)
    snr_val_spline = get_snr(X, Y)

    return lsd_val_kuleshov, lsd_val, snr_val, lsd_val_spline, snr_val_spline, np.mean(lsd_val_avg), np.mean(snr_val_avg)


