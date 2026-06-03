#!/bin/bash
#SBATCH --job-name=prh-eval
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/fafb/logs/eval_%j.out
#SBATCH --error=outputs/fafb/logs/eval_%j.err

# Runs steps 04 → 05 → 06 in sequence after training completes.
# Submit standalone:   sbatch eval_pipeline.slurm
# Chain after training: sbatch --dependency=afterok:<train_jobid> eval_pipeline.slurm
# Or use submit_all.sh to handle the dependency automatically.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/fafb/logs

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"

PYTHON=${PYTHON:-/gscratch/scrubbed/emazuh/morphoclass/bin/python}

# Unbuffered output so logs appear in real time
export PYTHONUNBUFFERED=1
# Limit BLAS to 1 thread per call — prevents over-subscription when numpy
# is called from multiple threads or repeatedly in a loop
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON" -c "import torch; print('torch ok, CUDA:', torch.cuda.is_available())"

echo "--- Step 04: extract embeddings ---"
"$PYTHON" experiments/fafb_perslay/04_extract_embeddings.py

echo "--- Step 05: compute metrics ---"
"$PYTHON" experiments/fafb_perslay/05_metrics.py

echo "--- Step 06: generate figures ---"
"$PYTHON" experiments/fafb_perslay/06_figures.py

echo "Finished: $(date)"
echo "Figures → outputs/fafb/figures/"
