"""
Extract time-averaged mel spectrogram embeddings for each audio patch
and write them to an HDF5 file suitable for analyze_mel_differences.py
and partial_band_accuracy.py.
"""

import os, argparse
import numpy as np
import h5py

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.io import load_h5, H5Dataset

import torchaudio.transforms as T
import torchaudio.functional as FF

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch.nn as nn


def parser_option():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', required=True, help='path to output h5 embedding archive')
  parser.add_argument('--input', required=True, help='path to input h5 patches archive')
  parser.add_argument('--len', type=int, required=True,
                      help='total number of samples in the input archive')
  parser.add_argument('--sr', type=int, default=48000, help='sample rate of the audio patches')
  parser.add_argument('--n_mels', type=int, default=256, help='number of mel bins')
  parser.add_argument('--n_fft', type=int, default=4*1024, help='FFT size for mel spectrogram')
  return parser.parse_args()


class Embeddings(nn.Module):
    def __init__(self, sr=48000, n_fft=4*1024, hop_length=256, n_mels=256):
        super(Embeddings, self).__init__()

        self.stft_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            mel_scale='slaney',  # matches librosa default (htk=False)
            norm=None,           # no area normalization — matches librosa default
        )

        self.layer = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        x = torch.squeeze(x, dim=2)
        stft_spec = self.stft_transform(x)  # (batch, n_mels, time)
        stft_spec = FF.amplitude_to_DB(stft_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
        stft_spec = self.layer(stft_spec)
        return stft_spec.view(stft_spec.size(0), -1)


def add_data(device, h5_file, args):

    get_embed = Embeddings(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels)
    d_embed = args.n_mels

    len_0 = 860
    dataset_list = []
    for ind in range(args.len // len_0):
        dataset_list.append(H5Dataset(args.input, start_idx=ind * len_0, end_idx=(ind + 1) * len_0))

    combined_dataset = ConcatDataset(dataset_list)
    val_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False, drop_last=True)

    def process_and_store(h5_file, loader):
        data_set = h5_file.create_dataset('data', (0, d_embed, 1),
                                          maxshape=(None, d_embed, 1), dtype=np.float32, chunks=True)
        label_set = h5_file.create_dataset('label', (0, 1, 1),
                                           maxshape=(None, 1, 1), dtype=np.float32, chunks=True)
        global_idx = 0
        for i, (sound, label) in enumerate(loader):
            print(f"{i}/{len(loader)}")
            sound_embed = get_embed(sound).detach().numpy()
            batch_size = sound.shape[0]
            data_set.resize(global_idx + batch_size, axis=0)
            label_set.resize(global_idx + batch_size, axis=0)
            data_set[global_idx: global_idx + batch_size] = sound_embed.reshape(batch_size, d_embed, 1)
            label_set[global_idx: global_idx + batch_size] = label.reshape(batch_size, 1, 1)
            global_idx += batch_size

    process_and_store(h5_file, val_loader)


if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args = parser_option()
    with h5py.File(args.out, 'w') as f_val:
        add_data(device, f_val, args)
