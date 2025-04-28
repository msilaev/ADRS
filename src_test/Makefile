SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p225

# patch generation

TR_DIM=8192
TR_STR=4096

VA_DIM=8192
VA_STR=4096
SR=16000
SCA=4

# fraction of patches to keep
SAM=0.25

# -------------------
# evaluate models on test set
# -------------------

# -------------------
# gan single speaker
# -------------------
run_evaluation_singlespeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.$(VA_DIM).$(VA_STR).h5 \
        --logname ../logs/singlespeaker/sr16000/logsGAN/singlespeaker.r_4.gan.b16.sr_16000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_singlespeaker_16 \
        --sr 16000

	python3 get_metrics.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.$(VA_DIM).$(VA_STR).h5 \
        --logname ../logs/singlespeaker/sr16000/logsAudiounet/singlespeaker.r_4.gan.b16.sr_16000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_singlespeaker_16 \
        --sr 16000

#run_evaluation_gan_48:

	python3 get_metrics.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.3.48000.$(VA_DIM).$(VA_STR).h5 \
        --logname ../logs/singlespeaker/sr48000/logsAudiounet/singlespeaker.r_3.gan.b16.sr_48000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_singlespeaker_48 \
        --sr 48000

	python3 get_metrics.py eval \
	    --val ../data/vctk/speaker1/vctk-speaker1-val.3.48000.$(VA_DIM).$(VA_STR).h5 \
        --logname ../logs/singlespeaker/sr48000/logsGAN/singlespeaker.r_3.gan.b16.sr_48000.generator_gan.epoch_400.pth\
        --out_label singlespeaker-out \
        --wav_file_list ../data/vctk/speaker1/speaker1-val-files.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_singlespeaker_48 \
        --sr 48000

# -------------------
# gan multispeaker
# -------------------
run_evaluation_multispeaker:

#run_evaluation_gan_16_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN/singlespeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_multispeaker_16 \
        --sr 16000 \
        --speaker multi

#run_evaluation_audiounet_16_multispeaker:

	python3 get_metrics.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsAudiounet/singlespeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_multispeaker_16 \
        --sr 16000 \
        --speaker multi

#run_evaluation_gan_alt_5_16_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN_Alt5/singlespeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_5_multispeaker_16 \
        --sr 16000 \
        --speaker multi

#run_evaluation_gan_alt_3_16_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGAN_Alt3/singlespeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_3_multispeaker_16 \
        --sr 16000 \
        --speaker multi

#run_evaluation_gen_s_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr16000/logsGenDecoupled/singlespeaker.r_4.gan.b128.sr_16000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gen_s_multispeaker_16 \
        --sr 16000 \
        --speaker multi

#run_evaluation_gen_s_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGenDecoupled/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gen_s_multispeaker_48 \
        --sr 48000 \
        --speaker multi



#run_evaluation_audiounet_48_multispeaker:

	python3 get_metrics.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsAudiounet/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model audiounet_multispeaker_48 \
        --sr 48000 \
        --speaker multi

#run_evaluation_gan_alt_5_48_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN_Alt5/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_5_multispeaker_48 \
        --sr 48000 \
        --speaker multi

#run_evaluation_gan_alt_3_48_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN_Alt3/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gan_alt_3_multispeaker_48 \
        --sr 48000 \
        --speaker multi

#run_evaluation_gen_s_multispeaker:

	python3 get_metrics.py eval \
	    --val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGenDecoupled/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 4 \
        --pool_size 2 \
        --strides 2 \
        --model gen_s \
        --sr 48000 \
        --speaker multi

#run_evaluation_gen_s_multispeaker:

	python3 get_metrics.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname ../logs/multispeaker/sr48000/logsGAN/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_445.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model gan_multispeaker_48 \
        --sr 48000 \
        --speaker multi

# -------------------
# hifigan multispeaker 
# -------------------
run_evaluation_multispeaker_hifigan:

	python3 get_metrics_hifigan.py eval \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
        --logname checkpoints/multispeaker/sr48000/gan/singlespeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_295.pth\
        --out_label multispeakerspeaker-out \
        --wav_file_list ../data/vctk/multispeaker/val-files-short.txt \
        --r 3 \
        --pool_size 2 \
        --strides 2 \
        --model hifigan \
        --sr 48000 \
        --speaker multi


