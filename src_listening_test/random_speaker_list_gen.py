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

    speaker_list_line = []

    for root, dirs, files in os.walk(directory):
        selected_speaker_files = []

        if files:
            speaker_name = files[0].split("_")[0]
            if speaker_name not in FEMALE_SPKRS:
                selected_speaker_files.append(random.choice(files))  # Select one random file
            else:
                selected_speakers = random.sample(files, 2)  # Select two random files
                selected_speaker_files = selected_speakers

        for file in selected_speaker_files:
            speaker_list_line.append(file+"\n")

    speakers_list_filename = args.output
    speakers_file = open(speakers_list_filename, "w")

    speakers_file.writelines(speaker_list_line)

    speakers_file.close()

def main():
    parser = make_parser()
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()