#!/bin/bash

# Run GAN training
python3 run_training_genDecoupled_48_r_3_multispeaker.py train \
    --train ../data/vctk/multispeaker/vctk-multispeaker-interp-train.3.48000.8192.8192.0.25.h5 \
    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
    --e 500 \
    --batch_size 128 \
    --lr 0.0001 \
    --logname multispeaker \
    --model gan \
    --r 3 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 48000 \
    --full false


