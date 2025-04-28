#!/bin/bash

#python3 random_speaker_list_gen.py \
#        --output speakers.txt  \
#        --data ../data/VCTK-test/source  \
#        --scale 3 \
#        --model upsample48_hifigan \
#        --sr 48000

#echo "formed speakers list"
# -------------------
# Evaluate models on test set
# -------------------
#echo "test gan 16000 complete"

python3 test_other_audio_onnx.py \
        --output ../data/listening_test_source/upsampling_12samples/result  \
        --data ../data/listening_test_source/source \
        --scale 4 \
        --model upsample16_gan \
        --sr 16000

echo "test gan 16000 complete"

python3 test_other_audio_onnx.py \
        --output ../data/listening_test_source/upsampling_12samples/result  \
        --data ../data/listening_test_source/source \
        --scale 4 \
        --model upsample16_audiounet \
        --sr 16000

#echo "test audiounet 16000 complete"

#python3 test_other_audio_onnx.py \
#        --output ../data/VCTK-test/result  \
#        --data ../data/VCTK-test/source \
#        --scale 3 \
#        --model upsample48_audiounet \
#        --sr 48000

echo "test audiounet 16000 complete"


