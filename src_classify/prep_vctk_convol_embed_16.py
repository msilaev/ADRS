"""
Extract convolutional embeddings from the discriminator for each audio patch
and write them to an HDF5 file suitable for convol_PCA_accuracy_*.py.
"""

import os, argparse
import numpy as np
import h5py

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.io import load_h5, H5Dataset
from models.multiScaleConv import MultiscaleConv1DBlock as MultiscaleConvBlock
from models.subPixel1D import SubPixel1D
from models.superPixel1D import SuperPixel1D

import torch
from torch.utils.data import ConcatDataset
from models.dataset_batch_norm import BatchData
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def parser_option():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', required=True, help='path to output h5 embedding archive')
  parser.add_argument('--input', required=True, help='path to input h5 patches archive')
  parser.add_argument('--len', type=int, required=True,
                      help='total number of samples in the input archive')
  parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='path to discriminator .pth checkpoint')
  parser.add_argument('--num_layers_disc', type=int, default=5,
                      help='number of downsampling layers in the discriminator')
  return parser.parse_args()


class Embeddings(nn.Module):

    def __init__(self, layers, time_dim, n_filters=(64, 128, 256, 256, 512)):

        super(Embeddings, self).__init__()

        self.layers = layers
        self.downsampling_layers = nn.ModuleList()

        n_in  = 1
        n_out = 128

        conv_layer = MultiscaleConvBlock(in_channels=n_in, out_channels=n_out//4)
        n_in = n_out

        x = nn.Sequential(conv_layer, nn.LeakyReLU(0.2))
        self.downsampling_layers.append(x)

        self.n_filters = n_filters

        for l in range(self.layers):
            n_out = self.n_filters[l] // 4
            conv_layer = MultiscaleConvBlock(in_channels=n_in, out_channels=n_out)
            batch_norm = nn.BatchNorm1d(4 * n_out)
            x = nn.Sequential(
                conv_layer,
                batch_norm,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SuperPixel1D(r=2))
            n_in = 8 * n_out
            self.downsampling_layers.append(x)

        self.input_features = n_in * time_dim // (2 ** self.layers)
        fc_outdim = 1024 // 32
        self.fc_1 = nn.Linear(self.input_features, fc_outdim)
        self.fc_2 = nn.Linear(fc_outdim, 1)
        self.final_layer = nn.Sequential(
            self.fc_1, nn.Dropout(0.5), nn.LeakyReLU(0.2), self.fc_2)

    def forward(self, x):
        batch_size, time_dim, channels = x.shape
        x = x.view(batch_size * 11, time_dim // 11, channels)
        x = x.transpose(1, 2)
        with torch.no_grad():
            for l in self.downsampling_layers:
                x = l(x)
        x = torch.mean(x, dim=2)  # (batch*11, features)
        x = x.view(batch_size, -1)
        return x


def add_data(device, h5_file, args):

    time_dim_1 = 8192
    d_embed = 11 * 1024

    get_embed = Embeddings(layers=args.num_layers_disc, time_dim=time_dim_1).to(device)
    get_embed.eval()

    print(f"Loading discriminator from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    get_embed.load_state_dict(state_dict)
    print("model loaded")

    len_0 = 860
    dataset_list = []
    for ind in range(args.len // len_0):
        dataset_list.append(H5Dataset(args.input, start_idx=ind * len_0, end_idx=(ind + 1) * len_0))

    combined_dataset = ConcatDataset(dataset_list)
    val_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False, drop_last=True)

    def process_and_store(h5_file, loader):
        data_set = h5_file.create_dataset('data', (0, d_embed, 1),
                                          maxshape=(None, d_embed, 1), dtype=np.float32, chunks=True)
        label_set = h5_file.create_dataset('label', (0, 1, 1),
                                           maxshape=(None, 1, 1), dtype=np.float32, chunks=True)
        global_idx = 0
        for i, (sound, label) in enumerate(loader):
            print(f"{i}/{len(loader)}")
            sound_embed = get_embed(sound.to(device)).detach().cpu().numpy()
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
