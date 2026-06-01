#!/bin/bash
#SBATCH --job-name=prh-pretrain
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/fafb/logs/pretrain_%j.out
#SBATCH --error=outputs/fafb/logs/pretrain_%j.err

# Pretrains CorianderNet with SupCon + CE loss — one model per partition (3 total).
# Intended to run on a large sample (10k/class) before the standard training sweep.
#
# Typical workflow:
#   python sample_dataset.py --per-class 10000 --seed 42
#   PREP=$(sbatch --parsable prep_data.slurm)
#   sbatch --dependency=afterok:$PREP pretrain_contrastive.slurm
#
# Override partition and seeds via environment:
#   sbatch --export=ALL,PARTITION=part_0,SEEDS="0 1" pretrain_contrastive.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

"$PYTHON" -c "import torch; print('CUDA:', torch.cuda.is_available(), \
    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo "--- Contrastive pretraining (SupCon + CE) ---"
ARGS=""
[[ -n "${PARTITION:-}" ]] && ARGS="$ARGS --partition $PARTITION"
[[ -n "${SEEDS:-}" ]]     && ARGS="$ARGS --seeds $SEEDS"
"$PYTHON" experiments/fafb_perslay/13_pretrain_contrastive.py $ARGS

echo "Finished: $(date)"
echo "Models → outputs/fafb/models/perslay_pretrain_part_*.pt"
