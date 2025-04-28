#!/bin/bash

# Set parameters
SCA=4
SR=16000
TR_DIM=8192
TR_STR=4096
VA_DIM=8192
VA_STR=4096

# Run GAN training
python3 run_training_audiounet_16_r_4_singlespeaker.py train \
    --train ../data/vctk/speaker1/vctk-speaker1-train.$SCA.$SR.$TR_DIM.$TR_STR.h5 \
    --val ../data/vctk/speaker1/vctk-speaker1-val.$SCA.$SR.$VA_DIM.$VA_STR.h5 \
    --e 400 \
    --batch_size 16 \
    --lr 0.0001 \
    --logname singlespeaker \
    --model gan \
    --r 4 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 16000 \
    --full false
