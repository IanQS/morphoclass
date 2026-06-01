#!/bin/bash
#SBATCH --job-name=prh-analysis
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/fafb/logs/analysis_%j.out
#SBATCH --error=outputs/fafb/logs/analysis_%j.err

# Steps 08 + 09 + 10: cross-representation analysis, subset sweep, tree analysis.
# All CPU-only. Depends on step 07 (NBLAST) outputs.
# Submit standalone:  sbatch analysis.slurm
# Or chain: sbatch --dependency=afterok:<nblast_jobid> analysis.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

echo "--- Step 08: cross-representation analysis ---"
"$PYTHON" experiments/fafb_perslay/08_cross_representations.py

echo "--- Step 09: subset scaling sweep ---"
"$PYTHON" experiments/fafb_perslay/09_subset_sweep.py --rerun

echo "--- Step 10: tree analysis ---"
"$PYTHON" experiments/fafb_perslay/10_tree_analysis.py

echo "Finished: $(date)"
echo "Figures → outputs/fafb/figures/"
