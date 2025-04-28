#!/bin/bash

# Configuration
SCA=4
SR=16000
TR_DIM=8192
TR_STR=4096
VA_DIM=8192
VA_STR=4096
SINGLE_SPEAKER_DIR="../VCTK-Corpus/speaker1"

# Generate training dataset
echo "Generating training dataset..."
python ../prep_vctk_singlespeaker.py \
    --file-list speaker1-train-files.txt \
    --in-dir "$SINGLE_SPEAKER_DIR" \
    --out vctk-speaker1-train.${SCA}.${SR}.${TR_DIM}.${TR_STR}.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --dimension $TR_DIM \
    --stride $TR_STR \
    --interpolate \
    --low-pass
mv vctk-speaker1-train.${SCA}.${SR}.${TR_DIM}.${TR_STR}.h5.tmp vctk-speaker1-train.${SCA}.${SR}.${TR_DIM}.${TR_STR}.h5

# Generate validation dataset
echo "Generating validation dataset..."
python ../prep_vctk_singlespeaker.py \
    --file-list speaker1-val-files.txt \
    --in-dir "$SINGLE_SPEAKER_DIR" \
    --out vctk-speaker1-val.${SCA}.${SR}.${VA_DIM}.${VA_STR}.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --dimension $VA_DIM \
    --stride $VA_STR \
    --interpolate \
    --low-pass
mv vctk-speaker1-val.${SCA}.${SR}.${VA_DIM}.${VA_STR}.h5.tmp vctk-speaker1-val.${SCA}.${SR}.${VA_DIM}.${VA_STR}.h5

echo "Dataset generation completed."
