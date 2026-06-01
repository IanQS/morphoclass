#!/bin/bash
#SBATCH --job-name=prh-pretrain-eval
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/fafb/logs/pretrain_eval_%j.out
#SBATCH --error=outputs/fafb/logs/pretrain_eval_%j.err

# Runs steps 04 + 12 on pretrained (contrastive) models.
# Chain after pretrain_contrastive.slurm:
#   PRETRAIN=$(sbatch --parsable pretrain_contrastive.slurm)
#   sbatch --dependency=afterok:$PRETRAIN pretrain_eval.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

echo "--- Step 04: extract pretrain embeddings ---"
"$PYTHON" experiments/fafb_perslay/04_extract_embeddings.py --tag pretrain

echo "--- Step 12: pretrain progress report ---"
"$PYTHON" experiments/fafb_perslay/12_progress_report.py --tag pretrain

echo "Finished: $(date)"
echo "Figures → outputs/fafb/figures/report_*_pretrain.png"
echo "Summary → outputs/fafb/metrics/progress_report_pretrain.json"
