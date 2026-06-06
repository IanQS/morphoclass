# PRH Morphology Validation — Reproduction Commands
# FlyWire FAFB (n=17,288 neurons, 20 classes)

## Environment

```bash
# Python binary for all experiments
PYTHON=/gscratch/scrubbed/emazuh/morphoclass/bin/python

# NBLAST uses a separate Python 3.13 environment with navis
PYTHON_NAVIS=/gscratch/stf/emazuh/miniconda3/bin/python

# Clone and install (from repo root)
pip install -e .
```

---

## Step 0: Sample dataset from full FlyWire archive

Sample 20 classes × up to 2,000 neurons (capped at available) from the full
139k-neuron zip **without** extracting all 13 GB:

```bash
# Interactive — CPU-light, ~2 min
python sample_dataset.py --per-class 2000 --seed 42
```

Outputs: `data/fafb_sample/swc/`, `data/fafb_sample/dataset.csv`,
`data/fafb_sample/classification_subset.csv`, `data/fafb_sample/root_ids.txt`

---

## Step 1–2: Prepare data (persist. diagrams) — SLURM

```bash
PREP_JOB=$(sbatch --parsable prep_data.slurm)
echo "Prep job: $PREP_JOB"
```

Resources: 8 CPUs, 32 GB, 2 h

Outputs:
- `outputs/fafb/data/dataset.csv` — copy of manifest used by pipeline
- `outputs/fafb/data/partitions.json` — 3-fold stratified splits (seed=42, fixed)
- `outputs/fafb/data/label_map.json`
- `outputs/fafb/data/diagrams.npz` — persistence diagrams for all neurons
- `outputs/fafb/data/morphometrics.npy`

---

## Step 3: Train 15 CorianderNet (PersLay) models — SLURM GPU

```bash
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB train_perslay.slurm)
echo "Train job: $TRAIN_JOB"
```

Resources: 1× A40 GPU, 4 CPUs, 16 GB, 2 h

Trains 15 models: 3 partitions × 5 seeds (LR=5e-4, batch=32, 500 epochs)

Outputs: `outputs/fafb/models/perslay_part_{0,1,2}_s{0..4}.pt` + `run_log.csv`

---

## Step 4–6: Extract embeddings, metrics, and standard figures — SLURM

```bash
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB eval_pipeline.slurm)
echo "Eval job: $EVAL_JOB"
```

Resources: 1× A40 GPU, 8 CPUs, 16 GB, 1 h

Outputs:
- `outputs/fafb/embeddings/emb_part_*_s*.npy` — 32-dim features
- `outputs/fafb/embeddings/logit_part_*_s*.npy` — 20-dim log-softmax logits
- `outputs/fafb/embeddings/labels.npy`
- `outputs/fafb/metrics/cka_logit_matrix.npy` — 15×15 CKA (primary PRH metric)
- `outputs/fafb/metrics/summary.json`
- `outputs/fafb/metrics/knn_jaccard.csv`
- `outputs/fafb/metrics/silhouette.csv`
- `outputs/fafb/figures/fig1_umap_grid.png` … `fig4_knn_stability.png`

---

## Step 7: NBLAST (navis, Python 3.13) — SLURM

```bash
FIG_JOB=$(sbatch --parsable --dependency=afterok:$EVAL_JOB figures.slurm)
NBLAST_JOB=$(sbatch --parsable --dependency=afterok:$FIG_JOB nblast.slurm)
```

Resources: 16 CPUs, 64 GB, 8 h  (uses miniconda Python — see nblast.slurm)

---

## Step 8–10: Cross-representation, subset sweep, tree analysis — SLURM

```bash
sbatch --dependency=afterok:$NBLAST_JOB analysis.slurm
```

---

## Step 12: CorianderNet progress report (standalone) — SLURM

```bash
sbatch progress_report.slurm
```

Or for CNNet:
```bash
sbatch --export=ALL,MODEL_TYPE=cnn progress_report.slurm
```

Resources: 4 CPUs, 16 GB, 1.5 h

Outputs:
- `outputs/fafb/figures/report_combined.png`
- `outputs/fafb/metrics/progress_report.json`

---

## Step 14: Train 15 CNNet (persistence image CNN) models — SLURM GPU

```bash
CNN_JOB=$(sbatch --parsable train_cnn.slurm)
```

Resources: 1× A40 GPU, 4 CPUs, 16 GB, 2 h

Outputs: `outputs/fafb/models/cnn_part_{0,1,2}_s{0..4}.pt` + `cnn_run_log.csv`
+ embeddings: `cnn_emb_part_*_s*.npy`, `cnn_logit_part_*_s*.npy`

---

## Cross-architecture CKA figure (instant, data already in metrics/)

```bash
$PYTHON experiments/fafb_perslay/fig_crossarch.py
```

Outputs: `outputs/fafb/figures/fig_crossarch_cka.png`

---

## Full pipeline (fresh start)

```bash
bash submit_all.sh
```

Or step-by-step with manual job IDs:
```bash
PREP=$(sbatch --parsable prep_data.slurm)
TRAIN=$(sbatch --parsable --dependency=afterok:$PREP train_perslay.slurm)
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN eval_pipeline.slurm)
FIG=$(sbatch --parsable --dependency=afterok:$EVAL figures.slurm)
NBLAST=$(sbatch --parsable --dependency=afterok:$FIG nblast.slurm)
sbatch --dependency=afterok:$NBLAST analysis.slurm
sbatch --dependency=afterok:$EVAL progress_report.slurm
```

---

## Key hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| LR | 5e-4 | hparam sweep (11_hparam_sweep.py) |
| Batch size | 32 | hparam sweep |
| Epochs | 500 | hparam sweep |
| Embedding dim | 32 | PersLay GaussianPointTransformer |
| Partitions | 3 | 3-fold stratified (seed=42) |
| Seeds per partition | 5 | 0,1,2,3,4 |
| CKA subsample | 4,000 | stratified by class (seed=42) |
| Null permutations | 200 | balanced speed/precision |

---

## Key results (n=17,288, 20 classes)

| Metric | Value |
|--------|-------|
| CorianderNet test accuracy | 0.487 ± 0.075 |
| CNNet test accuracy | 0.696 ± 0.038 |
| RF baseline accuracy | 0.804 ± 0.001 |
| CKA logit within-partition | 0.890 ± 0.132 |
| CKA logit cross-partition | 0.899 ± 0.103 |
| Permutation null p95 | 0.0001 |
| RSA (Kendall's τ) within | 0.847 ± 0.056 |
| RSA (Kendall's τ) cross | 0.845 ± 0.051 |
| Cross-arch CKA (Cori × CNN) | 0.406 ± 0.092 |
| Cross-arch RSA (Cori × CNN) | 0.434 ± 0.042 |
| Silhouette (feat space) | -0.199 ± 0.068 |
| kNN Jaccard logit within | 0.102 ± 0.081 |
| kNN Jaccard logit cross | 0.108 ± 0.092 |

---

## Data paths

| Path | Description |
|------|-------------|
| `/gscratch/scrubbed/emazuh/otterpack/data/sk_lod1_783_healed.zip` | Full FlyWire SWC archive (139,273 neurons) |
| `/gscratch/scrubbed/emazuh/otterpack/data/classification.csv.gz` | Full labels CSV |
| `data/fafb_sample/swc/` | 17,288 SWC files (20 classes, up to 2,000/class, seed=42) |
| `data/fafb_sample/dataset.csv` | Path↔label manifest |
| `outputs/fafb/data/partitions.json` | Fixed 3-fold split indices (seed=42) |
| `outputs/fafb/models/` | All 36 trained model weights (.pt) |
| `outputs/fafb/embeddings/` | All feature and logit embeddings (.npy) |
| `outputs/fafb/metrics/` | CKA matrices, summary JSONs, CSVs |
| `outputs/fafb/figures/` | All publication figures |
