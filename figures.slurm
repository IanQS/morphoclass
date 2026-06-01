#!/bin/bash
#SBATCH --job-name=prh-figs
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/fafb/logs/figures_%j.out
#SBATCH --error=outputs/fafb/logs/figures_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

echo "--- Step 06: generate figures ---"
"$PYTHON" experiments/fafb_perslay/06_figures.py

echo "Finished: $(date)"
echo "Figures → outputs/fafb/figures/"
