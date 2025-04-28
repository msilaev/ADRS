#!/bin/bash

	 python3 plt_spectrograms_whole_track.py train \
		--train ../data/vctk/multispeaker/vctk-4-class-train.3.48000.8192.8192.0.5.h5\
		--val ../data/vctk/multispeaker/vctk-4-class-val.3.48000.8192.8192.0.5.h5 \
		--e 11 \
		--batch_size 128 \
		--lr 0.00002\
		--logname singlespeaker \
		--model audiounet \
		--discriminator stft \
		--r 3 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 48000 \
		--full false

