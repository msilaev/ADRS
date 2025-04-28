import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

from models.audiounet import AudioUNet

from models.gan import Generator

import torch
from models.io import  upsample_wav

# -----------------------------------------------------------------
def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)

  train_parser.add_argument('--model', default='gan',
    choices=('gan', 'gan_simple'), help='model to train')
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('-e', '--epochs', type=int, default=100,
    help='number of epochs to train')
  train_parser.add_argument('--batch_size', type=int, default=128,
    help='training batch size')
  train_parser.add_argument('--logname', default='tmp-run',
    help='folder where logs will be stored')
  train_parser.add_argument('--layers', default=4, type=int,
    help='number of layers in each of the D and U halves of the network')
  train_parser.add_argument('--alg', default='adam',
    help='optimization algorithm')
  train_parser.add_argument('--lr', default=1e-3, type=float,
    help='learning rate')
  train_parser.add_argument('--r', type=int, default=4,
                            help='upscaling factor')
  train_parser.add_argument('--speaker', default='single',
                            choices=('single', 'multi'),
    help='number of speakers being trained on')
  train_parser.add_argument('--pool_size', type=int, default=4,
                            help='size of pooling window')
  train_parser.add_argument('--strides', type=int, default=4,
                            help='pooling stide')
  train_parser.add_argument('--full', default='false',
                            choices=('true', 'false'))
  train_parser.add_argument('--sr', help='high-res sampling rate',
                           type=int, default=48000)
  train_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')

  # eval
  eval_parser = subparsers.add_parser('eval')
  eval_parser.add_argument('--batch_size', type=int, default=16,
    help='training batch size')
  eval_parser.set_defaults(func=eval)
  eval_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  eval_parser.add_argument('--logname', required=True,
    help='path to training checkpoint')
  eval_parser.add_argument('--out_label', default='',
    help='append label to output samples')
  eval_parser.add_argument('--layers', default= 4,
                           help='number of layers')
  eval_parser.add_argument('--wav_file_list',
    help='list of audio files for evaluation')
  eval_parser.add_argument('--r', help='upscaling factor',
                           default = 4, type=int)
  eval_parser.add_argument('--sr', help='high-res sampling rate',
                                   type=int, default=48000)
  eval_parser.add_argument('--model', default='audiounet')
   # choices=('gan', 'gan_multispeaker', 'gan_multiD', 'audiounet_multispeaker',
   #          'gan_audiounet', 'audiounet', 'gan_multispeaker_bs_128'), help='model to train')
  eval_parser.add_argument('--speaker', default='single', choices=('single', 'multi'),
    help='number of speakers being trained on')
  eval_parser.add_argument('--pool_size', type=int, default=4,
                           help='size of pooling window')
  eval_parser.add_argument('--strides', type=int, default=4,
                           help='pooling stide')
  eval_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')

  return parser

# --------
# training gan
# --------
def train( args):
    pass

def eval(args):

  if args.model in ["gan",  "gan_multispeaker",  "gan_multispeaker_bs_128",
                    "gan_singlespeaker", "gan_GenF_multispeaker",
                    "gan_16_Clipping_multispeaker", "gan_WideDil_multispeaker",
                    "gan_WideAlt_multispeaker", "gan_alt_5_multispeaker",
                    "gan_alt_3_multispeaker", "gen_dec", "gen_s_multispeaker"]:

      model = Generator(layers=5)

  elif args.model in ["audiounet",  "audiounet_multispeaker", "audiounet_singlespeaker" ]:

      model = AudioUNet(layers=4)

  model.eval()

  checkpoint_root = args.logname

  if args.wav_file_list:
    with open(args.wav_file_list) as f:
      for line in f:

        try:
          if(args.speaker == 'single'):

            upsample_wav(model, '../data/vctk/VCTK-Corpus/wav48/p225/'+line.strip(),
                         args, epoch= None, model_path=checkpoint_root)

          else:

            upsample_wav(model, '../data/vctk'+line.strip().split("..")[1],
                         args, epoch= None, model_path=checkpoint_root)

        except EOFError:
          print('WARNING: Error reading file:', line.strip())

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
