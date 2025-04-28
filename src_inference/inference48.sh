#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {gan|audiounet|both}"
    exit 1
fi

if [ "$1" == "gan" ] || [ "$1" == "both" ]; then
    echo "Running inference with GAN multispeaker model..."
    python3 test_other_audio.py \
        --model gan_multispeaker \
        --output ../results/other_audio_gan \
        --data ../data/test-other \
        --sr 48000 \
        --logname ../logs/multispeaker/sr48000/logsGAN_Alt5/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth
fi

if [ "$1" == "audiounet" ] || [ "$1" == "both" ]; then
    echo "Running inference with AudioUNet multispeaker model..."
    python3 test_other_audio.py \
        --model audiounet_multispeaker \
        --output ../results/other_audio_audiounet \
        --data ../data/test-other \
        --sr 48000 \
        --logname ../logs/multispeaker/sr48000/logsAudiounet/multispeaker.r_3.gan.b128.sr_48000.generator_gan.epoch_500.pth
fi
