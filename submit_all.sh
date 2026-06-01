#!/bin/bash
# submit_all.sh — submit the full pipeline as a SLURM dependency chain.
#
# Usage:
#   bash submit_all.sh                  # prep → train → eval  (full run)
#   bash submit_all.sh --skip-prep      # train → eval  (data already prepared)
#   bash submit_all.sh --eval-only      # eval only  (models already trained)
#
# Environment overrides passed to the training job:
#   PARTITION=part_0 SEED=2 bash submit_all.sh --skip-prep

set -euo pipefail

SKIP_PREP=0
EVAL_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --skip-prep)  SKIP_PREP=1 ;;
    --eval-only)  EVAL_ONLY=1 ;;
  esac
done

mkdir -p outputs/fafb/logs

if [[ $EVAL_ONLY -eq 1 ]]; then
  EVAL_JOB=$(sbatch --parsable eval_pipeline.slurm)
  echo "Submitted eval pipeline: $EVAL_JOB"
  echo "  tail -f outputs/fafb/logs/eval_${EVAL_JOB}.out"
  exit 0
fi

if [[ $SKIP_PREP -eq 0 ]]; then
  PREP_JOB=$(sbatch --parsable prep_data.slurm)
  echo "Submitted prep (steps 01+02): $PREP_JOB"
  TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$PREP_JOB" train_perslay.slurm)
else
  TRAIN_JOB=$(sbatch --parsable train_perslay.slurm)
fi

echo "Submitted training (step 03): $TRAIN_JOB"
EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" eval_pipeline.slurm)
echo "Submitted eval (steps 04-06): $EVAL_JOB"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
[[ $SKIP_PREP -eq 0 ]] && echo "  tail -f outputs/fafb/logs/prep_${PREP_JOB}.out"
echo "  tail -f outputs/fafb/logs/perslay_${TRAIN_JOB}.out"
echo "  tail -f outputs/fafb/logs/eval_${EVAL_JOB}.out"
