"""
Create an HDF5 file of patches for training super-resolution model.
"""

import os, argparse
import numpy as np
import h5py
import pickle
import re
import librosa

from scipy import interpolate
from scipy.signal import decimate

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
  parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate');

  return parser.parse_args()

# ----------------------------------------------------------------------------
def add_data(h5_file, inputfiles, args):

  print(f"parameters sr {args.sr} r {args.scale}")

  # Make a list of all files to be processed
  file_list = []
  ID_list = []
  file_extensions = set(['.wav'])

  with open(inputfiles) as f:
    for line in f:
      filename = line.strip()
      ext = os.path.splitext(filename)[1]
      if ext in file_extensions:
        file_list.append(os.path.join(args.in_dir, filename))

  num_files = len(file_list)

  # patches to extract and their size
  #print("d_lr ", args.dimension ,  args.scale)

  d, d_lr = args.dimension, args.dimension
  s, s_lr = args.stride, args.stride

  hr_patches, lr_patches = list(), list()

  #print("len file list", len(file_list))

  for j, file_path in enumerate(file_list):

    if j % 10 == 0: print('%d/%d' % (j, num_files))

    directory_id_matches = re.search(fr'p(\d{{3}})\{os.path.sep}', file_path)
    ID = int(directory_id_matches.group(1))

    # load audio file
    x, fs = librosa.load(file_path, sr=args.sr)
    #x, fs = librosa.load(file_path, sr=None)

    # crop so that it works with scaling ratio
    x_len = len(x)
    x = x[ : x_len - (x_len % args.scale)]

    # generate low-res version
    x_lr = upsample(decimate(x, args.scale), args.scale)

    # alternative
    # x_lr = librosa.resample(x, orig_sr=fs, target_sr=fs // args.scale)
    #x_lr = librosa.resample(x_lr, orig_sr=fs // args.scale, target_sr=fs)

    #print(len(x_lr), args.scale, s)
    #input()

    #if args.dimension != -1:
        # generate patches

    max_i = len(x) - d + 1

    #print(max_i, s)
    for i in range(0, max_i, s):
        # keep only a fraction of all the patches

        i_lr = i #// args.scale

        #print("i_lr = ", i_lr)

        hr_patch = np.array( x[i : i+d] )
        lr_patch = np.array( x_lr[i_lr : i_lr+d_lr] )

        assert len(hr_patch) == d
        assert len(lr_patch) == d_lr

        hr_patches.append(hr_patch.reshape((d,1)))
        lr_patches.append(lr_patch.reshape((d_lr,1)))
        ID_list.append(ID)
        #print(ID)

 # if args.dimension != -1:
    # crop # of patches so that it's a multiple of mini-batch size
  num_patches = len(hr_patches)
  num_to_keep = int(np.floor(num_patches / args.batch_size) * args.batch_size)
  hr_patches = np.array(hr_patches[:num_to_keep])
  lr_patches = np.array(lr_patches[:num_to_keep])

  #print("num_to_keep", num_patches, args.batch_size, num_to_keep)
  ID_list = ID_list[:num_to_keep]

    # create the hdf5 file
  data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
  label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

  data_set[...] = lr_patches
  label_set[...] = hr_patches

  #print("patches shape", lr_patches.shape, hr_patches.shape)

  #print("len ID list", len(ID_list))
  pickle.dump(ID_list, open('ID_list_patches_'+str(d)+'_'+str(args.scale), 'wb'))


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
  # create train
  args = parser_option()
  with h5py.File(args.out, 'w') as f:
    add_data(f, args.file_list, args)
