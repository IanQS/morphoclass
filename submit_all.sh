#!/bin/bash
# submit_all.sh — submit training + eval pipeline as a dependent chain.
#
# Usage:
#   bash submit_all.sh                     # train all 15 models, then eval
#   bash submit_all.sh --partition part_0  # train one partition only, then eval
#   bash submit_all.sh --eval-only         # skip training, just run eval pipeline
#
# Environment overrides (passed through to the training job):
#   PARTITION=part_0 SEED=2 bash submit_all.sh

set -euo pipefail

EVAL_ONLY=0
TRAIN_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --eval-only) EVAL_ONLY=1 ;;
    *) TRAIN_ARGS+=("$arg") ;;
  esac
done

if [[ $EVAL_ONLY -eq 0 ]]; then
  TRAIN_JOB=$(sbatch --parsable train_perslay.slurm)
  echo "Submitted training job: $TRAIN_JOB"
  EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" eval_pipeline.slurm)
  echo "Submitted eval pipeline: $EVAL_JOB (runs after $TRAIN_JOB)"
  echo ""
  echo "Monitor:"
  echo "  squeue -u $USER"
  echo "  tail -f outputs/fafb/logs/perslay_${TRAIN_JOB}.out"
  echo "  tail -f outputs/fafb/logs/eval_${EVAL_JOB}.out"
else
  EVAL_JOB=$(sbatch --parsable eval_pipeline.slurm)
  echo "Submitted eval pipeline: $EVAL_JOB"
  echo "  tail -f outputs/fafb/logs/eval_${EVAL_JOB}.out"
fi
