#!/bin/bash
#SBATCH --job-name=ganAlt5-melLoss-16
#SBATCH --output=ganAlt5-melLoss-16-%j.log
#SBATCH --error=ganAlt5-melLoss-16-%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu-v100-32g

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate audio-enh

DATA_DIR="/scratch/elec/t412-speechcom/symptonic-r2b/simulation-r2b/data_ms/music/data/vctk"
SUBMIT_DIR="${SLURM_SUBMIT_DIR}"

export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH}"
cd "${SUBMIT_DIR}/src_train"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming GAN Alt5 + mel loss training (16 kHz, r=4) from epoch 95..."

python -u run_training_ganAlt5_melLoss_16_r_4_multispeaker.py train \
    --train "${DATA_DIR}/vctk-multispeaker-interp-train.4.16000.8192.8192.0.25.h5" \
    --val   "${DATA_DIR}/vctk-multispeaker-interp-val.4.16000.8192.8192.0.25.h5" \
    --e 500 \
    --batch_size 128 \
    --lr 0.0001 \
    --logname multispeaker \
    --model gan \
    --r 4 \
    --layers 4 \
    --pool_size 2 \
    --strides 2 \
    --sr 16000 \
    --full false \
    --mel_loss_weight 0.1 \
    --resume_epoch 95

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
