#!/bin/bash
#SBATCH --job-name=vctk-dataset-gen
#SBATCH --output=vctk-dataset-gen-%j.log
#SBATCH --error=vctk-dataset-gen-%j.err
#SBATCH --time=02:00:00             # Set appropriate time for your task
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=4           # Adjust CPU requirements
#SBATCH --mem=16G                   # Adjust memory requirements
#SBATCH --partition=standard        # Choose appropriate partition
#SBATCH --mail-type=END,FAIL        # Notifications on job completion or failure
#SBATCH --mail-user=your-email@example.com # Replace with your email

# Define variables
SCA=3                             # Scaling factor
SR=48000                            # Sampling rate
DIM=8192                            # Patch dimension
STR=8192                             # Patch stride
SAM=0.5                             # Fraction of patches to keep
SPEAKER_DIR=../VCTK-Corpus/wav48           # Directory of speaker files
N_TRAIN_SPEAKERS=100                 # Number of training speakers
N_VAL_SPEAKERS=9                     # Number of validation speakers

# Load required modules (if any)
# module load python/3.8

# Step 1: Generate train and validation speaker lists
echo "Generating speaker lists..."
ls $SPEAKER_DIR | head -n $N_TRAIN_SPEAKERS > train-speakers.txt
ls $SPEAKER_DIR | tail -n $N_VAL_SPEAKERS > val-speakers.txt

# Step 2: Generate file lists for train and validation
echo "Generating file lists..."
find ../VCTK-Corpus/ | grep -P '\.wav' | grep -f train-speakers.txt > train-files.txt
find ../VCTK-Corpus/ | grep -P '\.wav' | grep -f val-speakers.txt > val-files.txt

# Step 3: Create training dataset
echo "Creating training dataset..."
python ../prep_vctk_classificator.py \
    --interpolate \
    --file-list train-files.txt \
    --in-dir $(pwd) \
    --out vctk-audiounet-multispeaker-class-interp-train.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --model audiounet \
    --dimension $DIM \
    --stride $STR \
    --sam $SAM
mv vctk-audiounet-multispeaker-class-interp-train.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp vctk-audiounet-multispeaker-class-interp-train-1.$SCA.$SR.$DIM.$STR.$SAM.h5

# Step 4: Create validation dataset
echo "Creating validation dataset..."
python ../prep_vctk_classificator.py \
    --interpolate \
    --file-list val-files.txt \
    --in-dir $(pwd) \
    --out vctk-audiounet-multispeaker-class-interp-val.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp \
    --scale $SCA \
    --sr $SR \
    --model audiounet \
    --dimension $DIM \
    --stride $STR \
    --sam $SAM
mv vctk-audiounet-multispeaker-class-interp-val.$SCA.$SR.$DIM.$STR.$SAM.h5.tmp vctk-audiounet-multispeaker-class-interp-val-1.$SCA.$SR.$DIM.$STR.$SAM.h5

echo "Dataset generation completed!"
