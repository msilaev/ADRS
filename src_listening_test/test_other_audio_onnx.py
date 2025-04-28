import librosa
import os

import soundfile as sf
from scipy.signal import decimate
from scipy import interpolate

import argparse

from upsample48 import Upsample48
from upsample16 import Upsample16
import numpy as np
import random
FEMALE_SPKRS = ["p351", "p361", "p362"]
# -----------------------------------------------------------------
def make_parser():

  parser = argparse.ArgumentParser()
    # train
  parser.add_argument('--model', type=str)
  parser.add_argument('--logname')
  parser.add_argument('--output')
  parser.add_argument('--data')
  parser.add_argument('--scale')
  parser.add_argument('--sr', type=int)
  parser.add_argument('--patch_size', type=int, default=8192,
                            help='Size of patches over which the model operates')

  return parser
def eval(args):

    model_root = "../logs/multispeaker/listening_test"
    model_type = args.model

    output_dir = args.output
    directory = args.data

    speakers_list_filename = "speakers.txt"
    speaker_list = []

    print(model_type)

    with open(speakers_list_filename, "r") as f:
        for line in f:

            speaker_list.append(line.strip())

    for root, dirs, files in os.walk(directory):

        for file in files:

            if file.endswith(".wav") and file in speaker_list:

                input_file_path = os.path.join(root, file)

                input_file_path_wav = os.path.join(root, file.split(".")[0])

                relative_path_wav = os.path.relpath(input_file_path_wav, directory)

                output_file_path = os.path.join(output_dir, relative_path_wav)

                output_dir_1 = os.path.dirname(output_file_path)
                os.makedirs(output_dir_1, exist_ok = True)

                x_hr, fs = librosa.load(input_file_path, sr=int(args.sr))

                print(model_type)

                if model_type == 'upsample48_gan':

                    hr_origin_type = "hr48"
                    lr_origin_type = "lr_input16"

                    model_file = 'upsample48_gan_500.onnx'
                    processor = Upsample48(x_hr)
                    model_path = os.path.join(model_root, model_file)

                    processed_audio, _, _,  _ = processor.predict(model_path)
                    sf.write(output_file_path + '_'+  model_type +'.wav', processed_audio, 48000)
                    sf.write(output_file_path + '_' + hr_origin_type + '.wav', x_hr, 48000)

                    x_lr_spline = decimate(x_hr, 3)
                    x_lr_spline = spline_up(x_lr_spline, 3)

                    sf.write(output_file_path + '_' + lr_origin_type + '.wav', x_lr_spline, 48000)

                elif model_type == 'upsample48_audiounet':

                    hr_origin_type = "hr48"
                    lr_origin_type = "lr_input16"

                    model_file = 'upsample48_audiounet_500.onnx'
                    processor = Upsample48(x_hr)
                    model_path = os.path.join(model_root, model_file)

                    processed_audio, _, _,  _ = processor.predict(model_path)
                    sf.write(output_file_path + '_'+  model_type +'.wav', processed_audio, 48000)
                    sf.write(output_file_path + '_' + hr_origin_type + '.wav', x_hr, 48000)

                    x_lr_spline = decimate(x_hr, 3)
                    x_lr_spline = spline_up(x_lr_spline, 3)

                    sf.write(output_file_path + '_' + lr_origin_type + '.wav', x_lr_spline, 48000)

                elif model_type == 'upsample16_gan':

                    hr_origin_type = "hr16"
                    lr_origin_type = "lr_input4"

                    model_file = 'upsample16_gan_500.onnx'
                    processor = Upsample16(x_hr)
                    model_path = os.path.join(model_root, model_file)

                    processed_audio, _, _,  _  = processor.predict(model_path)
                    sf.write(output_file_path + '_'+  model_type + '.wav', processed_audio, 16000)
                    sf.write(output_file_path + '_' + hr_origin_type + '.wav', x_hr, 16000)

                    x_lr_spline = decimate(x_hr, 4)
                    x_lr_spline = spline_up(x_lr_spline, 4)
                    sf.write(output_file_path + '_' + lr_origin_type + '.wav', x_lr_spline, 16000)
                    print(output_file_path)

                elif model_type == 'upsample16_audiounet':

                    hr_origin_type = "hr16"
                    lr_origin_type = "lr_input4"

                    model_file = 'upsample16_audiounet_500.onnx'
                    processor = Upsample16(x_hr)
                    model_path = os.path.join(model_root, model_file)

                    processed_audio, _, _, _ = processor.predict(model_path)
                    sf.write(output_file_path + '_' + model_type + '.wav', processed_audio, 16000)
                    sf.write(output_file_path + '_' + hr_origin_type + '.wav', x_hr, 16000)

                    x_lr_spline = decimate(x_hr, 4)
                    x_lr_spline = spline_up(x_lr_spline, 4)
                    sf.write(output_file_path + '_' + lr_origin_type + '.wav', x_lr_spline, 16000)

def spline_up(x, r):

    x = x.flatten()
    len_x_up = len(x)*r
    x_up = np.zeros(len_x_up)

    i_lr = np.arange(len_x_up, step=r)
    i_hr = np.arange(len_x_up)

    f = interpolate.splrep(i_lr, x)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp.astype(np.float32)

def main():
    parser = make_parser()
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()