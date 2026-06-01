#!/bin/bash
#SBATCH --job-name=prh-nblast
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/fafb/logs/nblast_%j.out
#SBATCH --error=outputs/fafb/logs/nblast_%j.err

# Step 07: NBLAST pretrained model comparison.
# Needs navis >= 1.8. CPU-only — no GPU required.
# Submit standalone:  sbatch nblast.slurm
# Or chain after figures: sbatch --dependency=afterok:<fig_jobid> nblast.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

# navis >= 1.8 requires Python >= 3.9; use miniconda (3.13) for this step
PYTHON=${PYTHON:-/gscratch/stf/emazuh/miniconda3/bin/python}

echo "--- Step 07: NBLAST ---"
"$PYTHON" experiments/fafb_perslay/07_nblast.py

echo "Finished: $(date)"
