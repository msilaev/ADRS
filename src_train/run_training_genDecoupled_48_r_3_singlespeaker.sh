#!/bin/bash

# Set parameters
SCA=3
SR=48000
TR_DIM=8192
TR_STR=4096
VA_DIM=8192
VA_STR=4096

# Run GAN training for a single speaker
python3 run_training_genDecoupled_48_r_3_singlespeaker.py train \
    --train ../data/vctk/speaker1/vctk-speaker1-train.$SCA.$SR.$TR_DIM.$TR_STR.h5 \
    --val ../data/vctk/speaker1/vctk-speaker1-val.$SCA.$SR.$VA_DIM.$VA_STR.h5 \
    --e 400 \
    --batch_size 16 \
    --lr 0.0001 \
    --logname singlespeaker \
    --model gan \
    --r 3 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 48000 \
    --full false
