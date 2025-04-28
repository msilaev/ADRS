#!/bin/bash

# Run GAN training for multispeaker data
python3 run_training_gan_16_r_4_multispeaker.py train \
    --train ../data/vctk/multispeaker/vctk-multispeaker-interp-train.4.16000.8192.8192.0.25.h5 \
    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
    --e 500 \
    --batch_size 128 \
    --lr 0.0001 \
    --logname multispeaker \
    --model gan \
    --r 4 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 16000 \
    --full false