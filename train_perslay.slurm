#!/bin/bash
#SBATCH --job-name=prh-perslay
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/fafb/logs/perslay_%j.out
#SBATCH --error=outputs/fafb/logs/perslay_%j.err

# Submit from the repo root: sbatch train_perslay.slurm
# Override partition/seed: PARTITION=part_0 SEED=2 sbatch train_perslay.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Repo:      $SLURM_SUBMIT_DIR"
echo "Started:   $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"

# Activate the morphoclass conda environment
# Adjust CONDA_ENV if your env is installed elsewhere
CONDA_ENV=${CONDA_ENV:-morphoclass}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

python experiments/fafb_perslay/03_train_perslay.py \
    ${PARTITION:+--partition $PARTITION} \
    ${SEED:+--seed $SEED}

echo "Finished:  $(date)"
