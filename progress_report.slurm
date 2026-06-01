#!/bin/bash
#SBATCH --job-name=prh-report
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/fafb/logs/report_%j.out
#SBATCH --error=outputs/fafb/logs/report_%j.err

# PersLay-only progress report — no NBLAST, no cross-rep analysis.
# Reads existing logit embeddings and training logs.
# Submit after eval_pipeline.slurm completes:
#   sbatch --dependency=afterok:<eval_jobid> progress_report.slurm
# Or standalone:
#   sbatch progress_report.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

echo "--- Step 12: PersLay progress report ---"
"$PYTHON" experiments/fafb_perslay/12_progress_report.py

echo "Finished: $(date)"
echo "Figures → outputs/fafb/figures/report_*.png"
echo "Summary → outputs/fafb/metrics/progress_report.json"
