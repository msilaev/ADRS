# plot histograms of MOS and SNR on the valudation set
# -------------------------
run_hist_48_multispeaker_scores:

	python3 plt_hist_thesis_48.py \
	    --r 3 \
	    --model gan \
	    --sr 48000 \
	    --speaker multi

# plot histograms of MOS and SNR on the valudation set
# -------------------------
run_hist_16_multispeaker_scores:

	python3 plt_hist_thesis_16.py \
	    --r 4 \
	    --model gan \
	    --sr 16000 \
	    --speaker multi


############################################
run_illustration_fig1:

	python3 upsample_illustrate.py \
        --log_16 logs/multispeaker/sr16000/logs128Alt3/singlespeaker.r_4.gan.b128.sr_16000.generator_autoencoder_gan.epoch_500.pth\
		--log_48 logs/multispeaker/sr48000/logs129Alt5/singlespeaker.r_3.gan.b128.sr_48000.generator_autoencoder_gan.epoch_500.pth\
        --r_16 4 \
        --r_48 3 \
        --model gan_multispeaker_4_48 \
        --sr 48000

