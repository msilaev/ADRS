import librosa
import numpy as np
import os
import torch
from hifi_gan_bwe import BandwidthExtender
from scipy import interpolate
import soundfile as sf
from scipy.signal import decimate
import random

import argparse

FEMALE_SPKRS = ["p351", "p361", "p362"]

def make_parser():

  parser = argparse.ArgumentParser()

  ###############################
  parser.add_argument('--output')
  parser.add_argument('--data')
  parser.add_argument('--sr', type=int)
  parser.add_argument('--scale', type=int)
  parser.add_argument('--model')
  parser.add_argument('--patch_size', type=int, default=8192,
                            help='Size of patches over which the model operates')
  parser.add_argument('--r', type=int, default=3,
                            help='upscaling factor')

  return parser

def eval(args):

    model = BandwidthExtender.from_pretrained("hifi-gan-bwe-12-b086d8b-vctk-16kHz-48kHz")

    model.eval()

    output_dir = args.output
    directory = args.data
    model_type = args.model

    speakers_list_filename = "speakers.txt"
    speaker_list = []

    with open(speakers_list_filename, "r") as f:
        for line in f:
            speaker_list.append(line.strip())

    for root, dirs, files in os.walk(directory):

        for file in files:
            #print(file, speaker_list)
            #input()

            if file.endswith(".wav") and file in speaker_list:

                input_file_path = os.path.join(root, file)

                input_file_path_wav = os.path.join(root, file.split(".")[0])

                relative_path_wav = os.path.relpath(input_file_path_wav, directory)

                output_file_path = os.path.join(output_dir, relative_path_wav)

                output_dir_1 = os.path.dirname(output_file_path)
                os.makedirs(output_dir_1, exist_ok = True)

                # Load signal
                x_hr, fs = librosa.load(input_file_path, sr=int(args.sr))

                #x_len = len(x_hr)
                #x_hr = x_hr[: x_len - (x_len % args.scale)]

                x_lr = decimate(x_hr, args.scale)

                x_lr_tensor = torch.tensor(x_lr.flatten(), dtype=torch.float32)
                processed_audio = model(x_lr_tensor, args.sr // args.scale).detach().cpu().numpy()

                sf.write(output_file_path + '_' + model_type + '.wav', processed_audio, 48000)

def main():
    parser = make_parser()
    args = parser.parse_args()
    eval(args)

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