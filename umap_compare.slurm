#!/bin/bash
#SBATCH --job-name=prh-umap-cmp
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/fafb/logs/umap_compare_%j.out
#SBATCH --error=outputs/fafb/logs/umap_compare_%j.err

# 15_umap_compare.py — side-by-side PersLay vs CNNet UMAP. No retraining.
# Submit:  sbatch umap_compare.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}
SCRIPT=${SCRIPT:-experiments/fafb_perslay/15_umap_compare.py}

"$PYTHON" "$SCRIPT"

echo "Finished: $(date)"
echo "Figure → outputs/fafb/figures/"
