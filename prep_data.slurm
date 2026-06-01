#!/bin/bash
#SBATCH --job-name=prh-prep
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/fafb/logs/prep_%j.out
#SBATCH --error=outputs/fafb/logs/prep_%j.err

# Runs steps 01 (partitions) and 02 (persistence diagrams) on a CPU node.
# Submit standalone:          sbatch prep_data.slurm
# Or use submit_all.sh which chains prep → train → eval automatically.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

echo "--- Step 01: build dataset + partitions ---"
"$PYTHON" experiments/fafb_perslay/01_make_dataset.py

echo "--- Step 02: compute persistence diagrams ---"
"$PYTHON" experiments/fafb_perslay/02_compute_persistence.py

echo "Finished: $(date)"
