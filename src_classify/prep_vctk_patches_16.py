"""
Create an HDF5 file of raw audio patches for classifier training (16 kHz).

For each audio file: downsample to LR, run the generator, then write
HR patch (label=1) and generated patch (label=0) to the H5 archive.
"""

import os, argparse
import numpy as np
import h5py
import librosa

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy import interpolate
from scipy.signal import decimate
from models.gan import Generator
from models.audiounet import AudioUNet
import matplotlib.pyplot as plt

import torch

# ----------------------------------------------------------------------------
def parser_option():
  parser = argparse.ArgumentParser()

  parser.add_argument('--file-list',
    help='list of input wav files to process')
  parser.add_argument('--in-dir', default='~/',
    help='folder where input files are located')
  parser.add_argument('--out',
    help='path to output h5 archive')
  parser.add_argument('--scale', type=int, default=2,
    help='scaling factor')
  parser.add_argument('--dimension', type=int,
    help='dimension of patches--use -1 for no patching')
  parser.add_argument('--stride', type=int, default=3200,
    help='stride when extracting patches')
  parser.add_argument('--interpolate', action='store_true',
  help='interpolate low-res patches with cubic splines')
  parser.add_argument('--low-pass', action='store_true',
    help='apply low-pass filter when generating low-res patches')
  parser.add_argument('--batch-size', type=int, default=128,
    help='we produce # of patches that is a multiple of batch size')
  parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')
  parser.add_argument('--patch', type=int, default=48000)
  parser.add_argument('--sam', type=float, default=1.,
                      help='subsampling factor for the data')
  parser.add_argument('--full_sample', type=bool, default=True)
  parser.add_argument('--model', type=str)
  parser.add_argument('--checkpoint_path', type=str, default=None)
  parser.add_argument('--max_samples', type=int, default=None,
                      help='stop after writing this many samples (pairs count as 2)')

  return parser.parse_args()

def add_data(h5_file, inputfiles, args):

  if args.sam == 1:
      u_list = None
  else:
      if args.sam == 0.1:
          file_u = "sampling_u_01.txt"
      else:
          file_u = "sampling_u_05.txt"
      script_dir = os.path.dirname(os.path.abspath(__file__))
      with open(os.path.join(script_dir, file_u), "r") as f:
          u_list = f.read().strip().split()

  print(f"parameters sr {args.sr} r {args.scale}")

  if torch.cuda.is_available():
    print("CUDA!")
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  if args.model == "gan" or args.model == "gan_alt_1":
      generator = Generator(layers=5).to(device)
      checkpoint_name = "generator"
      strict_load = True
  elif args.model == "gan_melLoss":
      generator = Generator(layers=5).to(device)
      checkpoint_name = "generator"
      strict_load = False
  elif args.model == "audiounet":
      generator = AudioUNet(layers=4).to(device)
      checkpoint_name = "audiounet"
      strict_load = True
  elif args.model == "hifigan":
      from hifi_gan_bwe import BandwidthExtender
      generator = BandwidthExtender.from_pretrained(
          "hifi-gan-bwe-12-b086d8b-vctk-16kHz-48kHz"
      )
      checkpoint_name = None
      strict_load = True
  else:
      raise ValueError(f"Unknown model type: {args.model}")

  if checkpoint_name is not None:
      model_path = args.checkpoint_path
      print(f"Loading weights from: {model_path}")
      state_dict = torch.load(model_path, map_location=device)
      generator.load_state_dict(state_dict, strict=strict_load)
      print("model loaded")

  generator.eval()

  file_list = []
  file_extensions = set(['.wav', '.flac'])

  with open(inputfiles) as f:
    for line in f:
      filename = line.strip()
      ext = os.path.splitext(filename)[1]
      if ext in file_extensions:
        file_list.append(os.path.join(args.in_dir, filename))

  num_files = len(file_list)
  print(f"Files to process: {num_files}", flush=True)
  if num_files > 0:
      print(f"  First: {file_list[0]}", flush=True)
  else:
      print("  WARNING: file list is empty — check val-files.txt", flush=True)

  d, d_lr = args.dimension, args.dimension
  s, s_lr = args.stride, args.stride

  data_set = h5_file.create_dataset(
    'data', (0, d_lr, 1), maxshape=(None, d_lr, 1),
    dtype=np.float32, chunks=True
  )

  label_set = h5_file.create_dataset(
    'label', (0, 1, 1), maxshape=(None, 1, 1),
    dtype=np.float32, chunks=True
  )

  global_idx = 0
  total_ind = -1

  for j, file_path in enumerate(file_list):
    if j % 10 == 0:
      pct = 100.0 * j / num_files
      filled = int(30 * j / num_files)
      bar = '#' * filled + '-' * (30 - filled)
      print(f'[{bar}] {j}/{num_files} ({pct:.1f}%)', flush=True)

    x, fs = librosa.load(file_path, sr=args.sr)

    x_len = len(x)
    x = x[  (x_len % d)  :  ]

    for i in range(0, len(x), s):
        total_ind += 1

        if u_list is not None and u_list[total_ind] == "1": continue

        d_adjust = args.scale * (d // args.scale) + 3

        hr_patch_0 = np.array( x[i : i+d_adjust] )

        lr_patch = upsample(decimate(hr_patch_0, args.scale), args.scale)
        lr_patch_0 = decimate(hr_patch_0, args.scale)

        pr_patch = []
        hr_patch = []

        if args.model == "hifigan":

            x_lr_tensor = torch.tensor(lr_patch_0.flatten(), dtype=torch.float32)
            x_pr_hifigan = generator(x_lr_tensor, args.sr // args.scale).detach().cpu()

            for ii in range(0, len(hr_patch_0) // args.patch):

                hr_patch_1 = hr_patch_0[ii * args.patch: (ii + 1) * args.patch]
                pr_patch_1 = x_pr_hifigan[ii * args.patch: (ii + 1) * args.patch]

                x_hr_tensor_part = torch.tensor(hr_patch_1.flatten(),
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                x_hr_tensor_part = x_hr_tensor_part.squeeze()

                x_pr_tensor_part = torch.tensor(pr_patch_1.flatten(),
                                                dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                x_pr_tensor_part = x_pr_tensor_part.squeeze()

                pr_patch.append(x_pr_tensor_part)
                hr_patch.append(x_hr_tensor_part)

            pr_patch = torch.cat(pr_patch, dim=0)
            hr_patch = torch.cat(hr_patch, dim=0)

        else:

            num_patches = len(hr_patch_0) // args.patch
            hr_patch_0 = torch.tensor(hr_patch_0, dtype=torch.float32)
            lr_patch = torch.tensor(lr_patch, dtype=torch.float32).to(device)

            hr_patches = hr_patch_0[:num_patches*args.patch].reshape(args.patch, num_patches)
            lr_patches = lr_patch[:num_patches*args.patch].reshape( num_patches, args.patch)
            lr_patches = lr_patches.unsqueeze(2)

            pr_patches = generator(lr_patches).detach().cpu()
            pr_patch.extend(pr_patches)
            hr_patch.extend(hr_patches)

            pr_patch = torch.cat(pr_patch, dim =0)
            hr_patch = torch.cat(hr_patch, dim = 0)

        assert len(hr_patch) == d
        assert len(pr_patch) == d_lr

        data_set.resize(global_idx + 2, axis=0)
        label_set.resize(global_idx + 2, axis=0)

        data_set[global_idx] = hr_patch.reshape((d_lr, 1))
        data_set[global_idx+1] = pr_patch.reshape((d_lr, 1))

        label_patch = np.array(1)
        label_set[global_idx] = label_patch.reshape((1, 1))
        label_patch = np.array(0)
        label_set[global_idx+1] = label_patch.reshape((1, 1))

        global_idx += 2


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

if __name__ == '__main__':

  args = parser_option()
  with h5py.File(args.out, 'w') as f:
    add_data(f, args.file_list, args)
