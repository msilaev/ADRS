#!/bin/bash

# Define variables
SCA=3                             # Scaling factor
SR=48000                            # Sampling rate
DIM=8192                            # Patch dimension
STR=8192                             # Patch stride
SAM=0.1                          # Fraction of patches to keep
SPEAKER_DIR=../VCTK-Corpus/wav48           # Directory of speaker files
N_TRAIN_SPEAKERS=100                 # Number of training speakers
N_VAL_SPEAKERS=9                     # Number of validation speakers

echo "Creating val dataset..."
python3 ../prep_vctk_embed_classificator.py \
    --len 960 \
		--input vctk-hifigan-multispeaker-class-interp-val-sec-2.3.48000.90112.90112.0.1.h5 \
		--out vctk-hifigan-multispeaker-class-embed-val.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp
mv vctk-hifigan-multispeaker-class-embed-val.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp vctk-hifigan-multispeaker-class-embed-val-2.$SCA.$SR.$DIM.$STR.$SAM.h5

echo "Creating training dataset..."
python3 ../prep_vctk_embed_classificator.py \
    --len 9600 \
		--input vctk-hifigan-multispeaker-class-interp-train-sec-2.3.48000.90112.90112.0.1.h5 \
		--out vctk-hifigan-multispeaker-class-embed-train.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp
mv vctk-hifigan-multispeaker-class-embed-train.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp vctk-hifigan-multispeaker-class-embed-train-2.$SCA.$SR.$DIM.$STR.$SAM.h5

