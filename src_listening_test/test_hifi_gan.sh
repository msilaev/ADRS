#!/bin/bash

# -------------------
# Evaluate models on test set
# -------------------
python3 test_other_audio_hifi.py \
        --output ../data/VCTK-test/result  \
        --data ../data/VCTK-test/source  \
        --scale 3 \
        --model upsample48_hifigan \
        --sr 48000