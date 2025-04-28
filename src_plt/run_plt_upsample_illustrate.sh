#!/bin/bash

	python3 upsample_illustrate.py \
        --log_16 ../logs/multispeaker/sr16000/logsGAN_Alt3/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
		    --log_48 ../logs/multispeaker/sr48000/logsGAN_Alt5/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --r_16 4 \
        --r_48 3 \
        --model gan_multispeaker_4_48 \
        --sr 48000