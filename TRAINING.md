# Training Guide — PRH Morphology Experiment
## Hyak / Klone (UW Research Computing)

This guide gets a new team member from a fresh clone to a running SLURM
training job in under 10 minutes using the committed sample dataset
(96 neurons, 8 classes × 12, ~39 MB — no download required).

---

## 1. Clone and enter the repo

```bash
git clone <repo-url>
cd morphoclass
git checkout prh-experiments
```

---

## 2. Activate the shared conda environment

The `morphoclass` environment is pre-built at:
```
/gscratch/scrubbed/emazuh/morphoclass
```

```bash
conda activate /gscratch/scrubbed/emazuh/morphoclass
```

Verify:
```bash
python -c "from morphoclass.models import CorianderNet; print('OK')"
python -c "import torch; print('torch', torch.__version__, '| CUDA built for', torch.version.cuda)"
```

If the environment is missing, rebuild it from scratch:
```bash
conda create -n morphoclass python=3.8
conda activate morphoclass
pip install -e .                                          # installs morphoclass from source
pip install -r experiments/fafb_perslay/requirements.txt
pip install torch==2.4.1 torch-geometric==2.6.1
pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
```

---

## 3. Run the data pipeline (CPU, ~2 min)

These steps read `data/fafb_sample/` (already in the repo) and write
precomputed diagrams to `outputs/` (gitignored).

```bash
# Step 1 — build stratified partitions (3 folds × 5 seeds = 15 runs)
python experiments/fafb_perslay/01_make_dataset.py \
    --dataset data/fafb_sample/dataset.csv \
    --n_splits 3 --n_seeds 5 --min_per_class 8

# Step 2 — compute persistence diagrams + morphometric features (~1 min)
python experiments/fafb_perslay/02_compute_persistence.py
```

Expected output after step 2:
```
outputs/fafb/data/dataset.csv
outputs/fafb/data/partitions.json   ← fixed partitions, random_state=42
outputs/fafb/data/diagrams.npz      ← 96 persistence diagrams (µm scale)
outputs/fafb/data/morphometrics.npy ← (96, 8) morphometric features
```

> **Do not regenerate `partitions.json`** once training has started.
> All 15 models share the same partition boundaries.

---

## 4. Submit the training job

Run **from the repo root** so `$SLURM_SUBMIT_DIR` resolves correctly:

```bash
sbatch train_perslay.slurm
```

This trains all 15 CorianderNet models (3 partitions × 5 seeds) on a
single A40 GPU. Expected wall time: ~20–30 min for the 96-neuron sample.

### Optional overrides

Train a single partition or seed (useful for debugging):
```bash
PARTITION=part_0 sbatch train_perslay.slurm
PARTITION=part_0 SEED=2 sbatch train_perslay.slurm
```

Or run interactively on a GPU node:
```bash
srun --partition=ckpt --account=stf-ckpt --gres=gpu:a40:1 \
     --mem=16G --time=01:00:00 --pty bash
cd /path/to/morphoclass
conda activate /gscratch/scrubbed/emazuh/morphoclass
python experiments/fafb_perslay/03_train_perslay.py --partition part_0 --seed 0
```

---

## 5. Monitor the job

```bash
squeue --me                                      # check queue status
tail -f outputs/fafb/logs/perslay_<JOBID>.out   # live stdout
cat  outputs/fafb/logs/perslay_<JOBID>.err       # errors (if any)
```

The log prints val_acc every 50 epochs and a summary at the end:
```
Training part_0 seed=0
    epoch  50  val_acc=0.167
    epoch 100  val_acc=0.250
    epoch 150  val_acc=0.333
    epoch 200  val_acc=0.375
  val_acc=0.375  test_acc=0.312  elapsed=42.3s
...
Mean test accuracy: 0.305 ± 0.041
```

> At n=96 (8 classes, chance=0.125), test accuracy of ~0.30 is the
> expected result for the sample dataset. See README.md Key Results.

---

## 6. Outputs after training

```
outputs/fafb/models/
    perslay_part_{0,1,2}_s{0,1,2,3,4}.npz   ← embeddings + metrics (15 files)
    perslay_part_{0,1,2}_s{0,1,2,3,4}.pt    ← model state dicts (15 files)
    run_log.csv                              ← one row per run
```

Each `.npz` contains:
- `embeddings` — `(96, 32)` PersLay feature vectors for all neurons
- `labels`, `train_idx`, `val_idx`, `test_idx`
- `val_acc`, `test_acc`, `val_f1`, `test_f1`, `elapsed`

> `.npz` and `.pt` files are gitignored (large binaries). Do not commit them.

---

## 7. Run the analysis pipeline

After all 15 models finish:

```bash
# Extract and align embeddings across all runs
python experiments/fafb_perslay/04_extract_embeddings.py

# Compute CKA, silhouette, kNN Jaccard, RF baseline
python experiments/fafb_perslay/05_metrics.py

# Generate core figures (UMAP grid, CKA heatmap, silhouette, kNN)
python experiments/fafb_perslay/06_figures.py

# Cross-representation analysis (PI-PCA, graph-topology, interpretability)
python experiments/fafb_perslay/08_cross_representations.py

# Scaling sweep n=16→96 (uses cached results by default)
python experiments/fafb_perslay/09_subset_sweep.py

# Decision tree / RF analysis
python experiments/fafb_perslay/10_tree_analysis.py
```

Figures land in `outputs/fafb/figures/` and metrics in `outputs/fafb/metrics/`.

---

## 8. Scale to the full dataset (n=783)

Download from [codex.flywire.ai](https://codex.flywire.ai):
- `sk_lod1_783_healed.zip` → SWC skeletons (~13 GB)
- `classification.csv` → per-neuron labels

```bash
unzip sk_lod1_783_healed.zip -d data/fafb/swc/
cp classification.csv data/fafb/classification.csv

python experiments/fafb_perslay/01_make_dataset.py \
    --dataset data/fafb/classification.csv

python experiments/fafb_perslay/02_compute_persistence.py   # ~15 min

sbatch train_perslay.slurm   # ~2 h on A40
```

Expected at n=783: test accuracy ~0.65, positive silhouette,
cross-partition CKA gap opens clearly (Brbić 2026 inflation effect reduced).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch_scatter` | Missing dependency | `pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html` |
| `CUDA available: False` on compute node | Wrong node type | Check `squeue --me` — must be on a GPU node |
| All persistence diagrams look identical | Coordinate scale bug | Confirm `utils.py` applies `÷1000` (nm→µm); see CLAUDE.md §1 |
| Job pending for >30 min | Queue pressure | Try `--gres=gpu:2080ti:1` (more available) or check `sinfo` |
| `val_acc` stuck at chance | Too few epochs or bad init | Increase `N_EPOCHS` in `03_train_perslay.py` or rerun with different seed |
