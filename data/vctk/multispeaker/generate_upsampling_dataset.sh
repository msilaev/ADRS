#!/bin/bash

# Configuration
SPEAKER_DIR="../VCTK-Corpus/wav48"
N_TRAIN_SPEAKERS=100
N_VAL_SPEAKERS=9
SCA=4
DIM=8192 # length of audio patch
STR=8192 # stride controlling overlap of patches
SR=16000
SAM=1.0 # fraction of patches to keep, use 1 for sr=16kHz and 0.5 for sr=48kHz

# Generate train and validation speaker lists
echo "Generating train and validation speaker lists..."
ls "$SPEAKER_DIR" | head -n $N_TRAIN_SPEAKERS > train-speakers.txt
ls "$SPEAKER_DIR" | tail -n $N_VAL_SPEAKERS > val-speakers.txt

# Generate train and validation file lists
echo "Generating train and validation file lists..."
find ../VCTK-Corpus/ -type f -name "*.wav" | grep -f train-speakers.txt > train-files.txt
find ../VCTK-Corpus/ -type f -name "*.wav" | grep -f val-speakers.txt > val-files.txt

# Generate datasets
echo "Generating training dataset..."
python ../prep_vctk_multispeaker.py \
    --interpolate \
    --file-list train-files.txt \
    --in-dir "$(pwd)" \
    --out vctk-multispeaker-interp-train.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --dimension $DIM \
    --stride $STR \
    --sam $SAM
mv vctk-multispeaker-interp-train.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5.tmp vctk-multispeaker-interp-train.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5

echo "Generating validation dataset..."
python ../prep_vctk_multispeaker.py \
    --interpolate \
    --file-list val-files.txt \
    --in-dir "$(pwd)" \
    --out vctk-multispeaker-interp-val.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --dimension $DIM \
    --stride $STR \
    --sam $SAM
mv vctk-multispeaker-interp-val.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5.tmp vctk-multispeaker-interp-val.${SCA}.${SR}.${DIM}.${STR}.${SAM}.h5

echo "Dataset generation completed."
