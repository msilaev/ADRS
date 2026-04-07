#!/bin/bash
#SBATCH --job-name=ganAlt5-melLoss-48
#SBATCH --output=ganAlt5-melLoss-48-%j.log
#SBATCH --error=ganAlt5-melLoss-48-%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu-v100-32g

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate audio-enh

DATA_DIR="/scratch/elec/t412-speechcom/symptonic-r2b/simulation-r2b/data_ms/music/data/vctk"
SUBMIT_DIR="${SLURM_SUBMIT_DIR}"

export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH}"
cd "${SUBMIT_DIR}/src_train"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GAN Alt5 + mel loss training (48 kHz, r=3)..."

python -u run_training_ganAlt5_melLoss_48_r_3_multispeaker.py train \
    --train "${DATA_DIR}/vctk-multispeaker-interp-train.3.48000.8192.8192.0.25.h5" \
    --val   "${DATA_DIR}/vctk-multispeaker-interp-val.3.48000.8192.8192.0.25.h5" \
    --e 500 \
    --batch_size 128 \
    --lr 0.0001 \
    --logname multispeaker \
    --model gan \
    --r 3 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 48000 \
    --full false \
    --mel_loss_weight 0.1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
