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
# train models
# -------------------
patches:
	make \
	run_training

run_training:
	 python3 run_supervised.py train \
		--train ../data/vctk/speaker1/vctk-speaker1-train.$(SCA).$(SR).$(TR_DIM).$(TR_STR).h5 \
		--val ../data/vctk/speaker1/vctk-speaker1-val.$(SCA).$(SR).$(VA_DIM).$(VA_STR).h5 \
		--e 200 \
		--batch_size 16 \
		--lr 0.0001\
		--logname singlespeaker \
		--model audiounet \
		--r 4 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 16000 \
		--full false

run_training_gan_16_r_4:
	 python3 run_gan.py train \
		--train ../data/vctk/speaker1/vctk-speaker1-train.4.16000.$(TR_DIM).$(TR_STR).h5 \
		--val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.$(VA_DIM).$(VA_STR).h5 \
		--e 400 \
		--batch_size 16 \
		--lr 0.0001\
		--logname singlespeaker \
		--model gan \
		--r 4 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 48000 \
		--full false

run_training_gan_16_r_4_multispeaker:
	 python3 run_gan.py train \
		--train ../data/vctk/multispeaker/vctk-multispeaker-interp-train.4.16000.8192.8192.0.25.h5 \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5 \
		--e 400 \
		--batch_size 4 \
		--lr 0.0001\
		--logname singlespeaker \
		--model gan \
		--r 4 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 16000 \
		--full false

run_training_16_r_4_autoencoder:
	 python3 train_autoencoder.py  \
		--train ../data/vctk/speaker1/vctk-speaker1-train.4.16000.$(TR_DIM).$(TR_STR).h5 \
		--val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.$(TR_DIM).$(TR_STR).h5 \
		--e 400 \
		--batch_size 4 \
		--lr 0.001\
		--logname singlespeaker \
		--model autoencoder \
		--r 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 16000 \
		--full false

run_training_gan_48_r_3_multispeaker:
	 python3 run_gan.py train \
		--train ../data/vctk/multispeaker/vctk-multispeaker-interp-train.3.48000.8192.8192.0.25.h5 \
		--val ../data/vctk/multispeaker/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5 \
		--e 400 \
		--batch_size 32 \
		--lr 0.0001\
		--logname singlespeaker \
		--model gan \
		--r 3 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 48000 \
		--full false

run_training_gan_48_r_3:
	 python3 run_gan.py train \
		--train ../data/vctk/speaker1/vctk-speaker1-train.3.48000.$(TR_DIM).$(TR_STR).h5 \
		--val ../data/vctk/speaker1/vctk-speaker1-val.3.48000.$(VA_DIM).$(VA_STR).h5 \
		--e 400 \
		--batch_size 16 \
		--lr 0.0001\
		--logname singlespeaker \
		--model gan \
		--r 3 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 48000 \
		--full false

run_training_gan_48_r_4:
	 python3 run_gan.py train \
		--train ../data/vctk/speaker1/vctk-speaker1-train.4.48000.$(TR_DIM).$(TR_STR).h5 \
		--val ../data/vctk/speaker1/vctk-speaker1-val.4.48000.$(VA_DIM).$(VA_STR).h5 \
		--e 400 \
		--batch_size 16 \
		--lr 0.0001\
		--logname singlespeaker \
		--model gan \
		--r 4 \
		--layers 4 \
		--pool_size 2 \
		--strides 2 \
		--sr 48000 \
		--full false


