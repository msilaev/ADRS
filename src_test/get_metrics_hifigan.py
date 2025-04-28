import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from models.audiounet import AudioUNet

from models.gan import Generator

import torch
import torch.nn as nn
from hifi_gan_bwe import BandwidthExtender

from dataset_batch_norm import BatchData
from models.io import load_h5, upsample_wav, eval_snr_lsd_hifigan

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

  batch_size = args.batch_size

  X_val, Y_val = load_h5(args.val)

  lr_mean = 0
  lr_std = 1
  hr_mean = 0
  hr_std = 1

  dataset_val = BatchData(X_val, Y_val, lr_mean, lr_std, hr_mean, hr_std)

  val_loader = \
      torch.utils.data.DataLoader(dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False, drop_last=True)

  if args.model in ["gen_s_multispeaker_16", "gen_s_multispeaker_48", "gan_alt_3_multispeaker_16", "gan_alt_5_multispeaker_16", "gan_multispeaker_16", "gan_multispeaker_48", "gan_alt_5_multispeaker_48", "gan_alt_3_multispeaker_48", "gan_singlespeaker_16", "gan_singlespeaker_48"]:


    model = Generator(layers=5)

  elif args.model in ["audiounet_multispeaker_16", "audiounet_multispeaker_48", "audiounet_singlespeaker_16", "audiounet_singlespeaker_48"]:

    model = AudioUNet(layers=4)
    
  elif args.model in ["hifigan"]:
  
    model = BandwidthExtender.from_pretrained("hifi-gan-bwe-12-b086d8b-vctk-16kHz-48kHz")

  model.eval()

  lsd_val_kuleshov, lsd_val, snr_val, lsd_val_spline, snr_val_spline, lsd_val_avg, snr_val_avg = \
      eval_snr_lsd_hifigan(model, val_loader, model_path=None)

  metrics_filename_full = os.path.join("logs", "metrics_summary.txt")
  
  # Define table headers
  headers = ["Model", "LSD Kuleshov", "LSD", "SNR", "LSD Spline", "SNR Spline", "LSD Avg", "SNR Avg"]

  # Define table data
  data = [args.model, lsd_val_kuleshov, lsd_val, snr_val, lsd_val_spline, snr_val_spline, lsd_val_avg, snr_val_avg]

  # Format the table output manually
  header_row = f"{headers[0]:<10} {headers[1]:<12} {headers[2]:<8} {headers[3]:<6} {headers[4]:<12} {headers[5]:<12} {headers[6]:<8} {headers[7]:<8}"
  data_row = f"{data[0]:<10} {data[1]:<12.3f} {data[2]:<8.3f} {data[3]:<6.2f} {data[4]:<12.3f} {data[5]:<12.2f} {data[6]:<8.3f} {data[7]:<8.2f}"
  
  data_row = f"| {data[0]:<10} | {data[1]:<12.3f} | {data[2]:<8.3f} | {data[3]:<6.2f} | {data[4]:<12.3f} | {data[5]:<12.2f} | {data[6]:<8.3f} | {data[7]:<8.2f} |"
  separator = "-" * len(header_row)

  # Print formatted table
  #print(header_row)
  #print("-" * len(header_row))
  #print(data_row)

  with open(metrics_filename_full, "a") as f_write_metrics:

      
      f_write_metrics.write(separator)
      f_write_metrics.write(f"\n")
      f_write_metrics.write(data_row)
      f_write_metrics.write(f"\n")


def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
