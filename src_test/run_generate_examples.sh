#!/bin/bash

# -------------------
# Evaluate models on test set
# -------------------
python3 generate_examples.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 \
        --logname ../logs/singlespeaker/sr16000/logsGAN/singlespeaker.r_4.gan.b16.sr_16000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_singlespeaker \
        --sr 16000 \
        --speaker single

python3 generate_examples.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 \
        --logname ../logs/singlespeaker/sr16000/logsAudiounet/singlespeaker.r_4.gan.b16.sr_16000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_singlespeaker \
        --sr 16000 \
        --speaker single

python3 generate_examples.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.3.48000.8192.4096.h5 \
        --logname ../logs/singlespeaker/sr48000/logsAudiounet/singlespeaker.r_3.gan.b16.sr_48000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_singlespeaker \
        --sr 48000 \
        --speaker single

python3 generate_examples.py eval \
	      --val ../data/vctk/speaker1/vctk-speaker1-val.3.48000.8192.4096.h5 \
        --logname ../logs/singlespeaker/sr48000/logsGAN/singlespeaker.r_3.gan.b16.sr_48000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_singlespeaker \
        --sr 48000 \
        --speaker single


python3 generate_examples.py eval \
		    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_445.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_multispeaker \
        --sr 48000 \
        --speaker multi


python3 generate_examples.py eval \
    		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsAudiounet/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_multispeaker \
        --sr 48000 \
        --speaker multi

python3 generate_examples.py eval \
    		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGenDecoupled/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gen_s_multispeaker \
        --sr 48000 \
        --speaker multi

python3 generate_examples.py eval \
   	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsAudiounet/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGenDecoupled/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gen_s_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN_Alt5/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_5_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN_Alt3/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_3_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
	      --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN_Alt3/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_3_multispeaker \
        --sr 16000 \
        --speaker multi

python3 generate_examples.py eval \
		    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN_Alt5/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_5_multispeaker \
        --sr 48000 \
        --speaker multi

python3 generate_examples.py eval \
	    	--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN_Alt3/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_3_multispeaker \
        --sr 48000 \
        --speaker multi

#python3 eval_4_48.py \
#	      --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
#        --log_16 ../logs/multispeaker/sr16000/logsGAN_Alt3/multispeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
#		    --log_48 ../logs/multispeaker/sr48000/logsGAN_Alt5/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
#        --out_label multispeakerspeaker-out \
#        --wav_file_list ../data/vctk/multispeaker/val-files-short-short.txt \
#        --r_16 4 \
#        --r_48 3 \
#        --model gan_multispeaker_4_48 \
#        --sr 48000 \
#        --batch_size 128 \
#        --speaker multi
