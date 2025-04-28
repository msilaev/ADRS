#!/bin/sh

import os

import argparse
import numpy as np
import librosa

# -------------------
# parser
# -------------------
def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list',
        help='list of input wav files to process')
    parser.add_argument('--out_label', default='',
                        help='append label to output samples')
    parser.add_argument('--r', type=int, default=4, help='upscaling factor')
    parser.add_argument('--speaker', default='p225', help='help')

    parser.add_argument('--model')

    parser.add_argument('--sr')

    return parser

def get_snr(P, Y):
    # Compute L2 loss
    sqrt_l2_loss = (np.mean((P - Y) ** 2) )
    sqrn_l2_norm = (np.mean(Y ** 2))
    snr = 10 * np.log10(sqrn_l2_norm / (sqrt_l2_loss ))

    avg_sqrt_l2_loss = np.mean(sqrt_l2_loss)
    avg_snr = np.mean(snr)

    return avg_snr

def eval_snr(args):

  Y = []
  P = []

  #root_dir = '../data/vctk/VCTK-Corpus/wav48/' + args.speaker + '/'

  if args.file_list:

    with open(args.file_list) as f:
      for line in f:
        try:
          x_hr, x_pr, x_lr = load_wav(line.strip(), args)
          P.append(x_pr)
          Y.append(x_hr)

        except EOFError:
          print('WARNING: Error reading file:', line.strip())

  Y = np.concatenate(Y)
  P = np.concatenate(P)

  return get_snr(P, Y)

def load_wav(wav, args):

    outname = wav + '.' + args.out_label + '.torch'

    if args.sr == '16000':
        output_dir_audio = '../results/audio16/'
    elif args.sr == '48000':
        output_dir_audio = '../results/audio48/'

    outname_audio = output_dir_audio +  wav.split("/")[-1] + '.' + args.out_label + '.torch'

    x_lr, _ = librosa.load(outname_audio + '.r' + str(args.r) + '.lr.wav', sr= None)
    x_hr, _ = librosa.load(outname_audio +  '.hr.wav' , sr= None)
    x_pr, _ = librosa.load(outname_audio + '.r' + str(args.r)  + '.' + str(args.model)  +
                           '.pr.wav',  sr= None)

    return x_hr, x_pr, x_lr


def main():
    parser = make_parser()
    args = parser.parse_args()

    print(f"Upsampling to {args.sr} Hz with factor {args.r}, model {args.model}, SNR = {eval_snr(args)}")

if __name__ == "__main__":
    main()
