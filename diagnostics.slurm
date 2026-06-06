#!/bin/bash
#SBATCH --job-name=prh-diagnostics
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/fafb/logs/diagnostics_%j.out
#SBATCH --error=outputs/fafb/logs/diagnostics_%j.err

# 12_diagnostics.py — three diagnostic experiments on existing embeddings.
# No retraining, no NBLAST. Full-N (17,288) debiased CKA is memory-heavy, so
# this runs on a compute node (64G) rather than the login node.
# Submit:  sbatch diagnostics.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

"$PYTHON" experiments/fafb_perslay/12_diagnostics.py

echo "Finished: $(date)"
echo "Outputs → outputs/fafb/metrics/ and outputs/fafb/figures/"
