#!/bin/bash
#SBATCH --job-name=prh-cnn
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/fafb/logs/cnn_%j.out
#SBATCH --error=outputs/fafb/logs/cnn_%j.err

# Trains 15 CNNet models (persistence image CNN) — cross-architecture PRH test.
# CKA(CorianderNet-logit, CNNet-logit) tests whether different input encodings
# of the same topology converge to the same representation.
#
# Run after 03_train_perslay.py has completed (uses same diagrams.npz).
# Submit standalone: sbatch train_cnn.slurm
# Single model:      sbatch --export=ALL,PARTITION=part_0,SEED=0 train_cnn.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

"$PYTHON" -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

ARGS=""
[[ -n "${PARTITION:-}" ]] && ARGS="$ARGS --partition $PARTITION"
[[ -n "${SEED:-}" ]]      && ARGS="$ARGS --seed $SEED"

echo "--- Training CNNet (persistence image CNN) ---"
"$PYTHON" experiments/fafb_perslay/14_train_cnn.py $ARGS

echo "Finished: $(date)"
echo "Models     → outputs/fafb/models/cnn_part_*_s*.pt"
echo "Embeddings → outputs/fafb/embeddings/cnn_*_part_*_s*.npy"
