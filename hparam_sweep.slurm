#!/bin/bash
#SBATCH --job-name=prh-sweep
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/fafb/logs/sweep_%j.out
#SBATCH --error=outputs/fafb/logs/sweep_%j.err

# Submit from the repo root: sbatch hparam_sweep.slurm
# Override partition/seed: PARTITION=part_0 SEED=0 sbatch hparam_sweep.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Repo:      $SLURM_SUBMIT_DIR"
echo "Started:   $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

"$PYTHON" -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

"$PYTHON" experiments/fafb_perslay/11_hparam_sweep.py \
    --partition ${PARTITION:-part_0} \
    --seed      ${SEED:-0}

echo "Finished:  $(date)"
