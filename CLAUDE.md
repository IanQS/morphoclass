# CLAUDE.md
# PRH Morphology Validation — FlyWire FAFB
# Guidance for Claude Code (VS Code / terminal)

This file tells Claude Code how this project works so it can help you
run experiments, generate commits, and debug without needing to read
all the scripts first.

---

## Project summary

We test the **Platonic Representation Hypothesis** (Huh et al. 2024) on
neuron morphology from the FlyWire FAFB connectome. The core question:
do neural networks trained on different subsets of the same skeleton data
converge to geometrically similar embedding spaces?

**Key constraint:** All experiment code lives in `experiments/fafb_perslay/`.
**Never modify** anything under `src/morphoclass/` — that is the upstream library.
**Never run training or compute-heavy steps directly on the login node — always use sbatch.**

---

## Data sources

| Path | Description |
|------|-------------|
| `/gscratch/scrubbed/emazuh/otterpack/data/sk_lod1_783_healed.zip` | Full FlyWire SWC archive (139,273 neurons, 13 GB) |
| `/gscratch/scrubbed/emazuh/otterpack/data/classification.csv.gz` | Labels: root_id, flow, super_class, class, sub_class, … (107,591 labeled) |
| `data/fafb_sample/swc/` | Sampled SWC files extracted from zip (current experiment) |
| `data/fafb_sample/dataset.csv` | path↔label manifest read by pipeline scripts |
| `data/fafb_sample/classification_subset.csv` | Full metadata for sampled neurons |
| `data/fafb_sample/root_ids.txt` | All 4,000 root IDs, one per line |
| `data/fafb_sample/viewer_sample.csv` | 5 per class (8 classes) for neuron viewer |
| `data/fafb_sample/viewer_sample_all_classes.csv` | 5 per class across all 29 classes for viewer |

**Current sample:** 4,000 neurons — 500 per class × 8 classes (ALPN, AN, CX, Kenyon_Cell,
mechanosensory, olfactory, optic_lobe_intrinsic, visual), seed=42.

**Next run (planned):** 20 classes ≥ 50 neurons, capped at 500/class → ~6,586 neurons.
`TARGET_CLASSES` in `sample_dataset.py` is already updated to all 20 classes.

To resample (e.g. different size or seed):
```bash
# This is CPU-light enough to run interactively
python sample_dataset.py --per-class 500 --seed 42
# Then rerun the full pipeline via SLURM:
bash submit_all.sh
```

---

## Repository layout

```
src/morphoclass/              ← upstream library — DO NOT MODIFY
experiments/
└── fafb_perslay/
    ├── utils.py              ← shared primitives (always import from here)
    ├── 00_sanity_checks.py
    ├── 01_make_dataset.py
    ├── 02_compute_persistence.py
    ├── 03_train_perslay.py
    ├── 04_extract_embeddings.py
    ├── 05_metrics.py
    ├── 06_figures.py
    ├── 07_nblast.py
    ├── 08_cross_representations.py
    ├── 09_subset_sweep.py
    ├── 10_tree_analysis.py
    └── 11_hparam_sweep.py
data/
└── fafb_sample/
    ├── swc/                  ← *.swc skeleton files (sampled subset)
    ├── dataset.csv           ← path<TAB>label manifest
    ├── classification_subset.csv
    ├── root_ids.txt
    └── viewer_sample*.csv
outputs/
└── fafb/
    ├── data/                 ← dataset.csv, partitions.json, diagrams.npz
    ├── models/               ← perslay_part_*_s*.pt (15 files), run_log.csv
    ├── embeddings/           ← emb_*.npy, logit_*.npy, labels.npy
    ├── metrics/              ← summary.json, cka_*_matrix.npy, sweep.json
    ├── figures/              ← fig1_*.png … fig8_*.png
    └── logs/                 ← SLURM *.out / *.err per job
sample_dataset.py             ← re-sample from full zip without extracting all 13 GB
submit_all.sh                 ← submit full pipeline as a SLURM dependency chain
prep_data.slurm               ← steps 01+02 (CPU, 8 cores, 32 GB, 2 h)
train_perslay.slurm           ← step 03  (GPU, A40, 2 h)
eval_pipeline.slurm           ← steps 04+05+06 (CPU, 4 cores, 8 GB, 1 h)
figures.slurm                 ← step 06 only (CPU, 4 cores, 8 GB, 30 min)
nblast.slurm                  ← step 07  (CPU, 16 cores, 64 GB, 8 h) — uses miniconda Python for navis
analysis.slurm                ← steps 08+09+10 (CPU, 4 cores, 16 GB, 1 h)
hparam_sweep.slurm            ← 11_hparam_sweep.py (GPU)
```

---

## How to run the full pipeline (SLURM)

**Always submit via sbatch — never run training or persistence computation on the login node.**

### Full run (new data or fresh start)
```bash
bash submit_all.sh                # prep → train → eval  (full chain)
bash submit_all.sh --skip-prep    # train → eval  (data already prepared)
bash submit_all.sh --eval-only    # eval only  (models already trained)
```

### Run remaining analysis steps (after eval completes)
```bash
FIG_JOB=$(sbatch --parsable figures.slurm)
NBLAST_JOB=$(sbatch --parsable --dependency=afterok:"$FIG_JOB" nblast.slurm)
sbatch --dependency=afterok:"$NBLAST_JOB" analysis.slurm
squeue -u $USER
```

### Monitor running jobs
```bash
squeue -u $USER
tail -f outputs/fafb/logs/prep_<JOBID>.out
tail -f outputs/fafb/logs/perslay_<JOBID>.out
tail -f outputs/fafb/logs/eval_<JOBID>.out
tail -f outputs/fafb/logs/figures_<JOBID>.out
tail -f outputs/fafb/logs/nblast_<JOBID>.out
tail -f outputs/fafb/logs/analysis_<JOBID>.out
```

### Single model (for debugging — still use sbatch)
```bash
sbatch --export=ALL,PARTITION=part_0,SEED=0 train_perslay.slurm
```

---

## What each script produces

| Script | Key outputs |
|--------|-------------|
| `00_sanity_checks.py` | `outputs/fafb/data/class_distribution.png` |
| `01_make_dataset.py` | `outputs/fafb/data/dataset.csv`, `partitions.json`, `label_map.json` |
| `02_compute_persistence.py` | `diagrams.npz`, `morphometrics.npy`, `persistence_sanity.png` |
| `03_train_perslay.py` | `models/perslay_part_{0,1,2}_s{0..4}.pt` (15 files), `run_log.csv` |
| `04_extract_embeddings.py` | `embeddings/emb_part_*_s*.npy` (feat, 32-dim) + `logit_part_*_s*.npy` (8-dim), `labels.npy` |
| `05_metrics.py` | `cka_logit_matrix.npy`, `cka_feat_matrix.npy`, `silhouette.csv`, `knn_jaccard.csv`, `summary.json` |
| `06_figures.py` | `fig1_umap_grid.png`, `fig2_cka_heatmap.png`, `fig3_silhouette.png`, `fig4_knn_stability.png` |
| `07_nblast.py` | `nblast_scores.npy`, `nblast_pca32_emb.npy`, `nblast_results.json` |
| `08_cross_representations.py` | `pi_pca_emb.npy`, `graph_topology_emb.npy`, `fig5_interpretability.png`, `fig8_cka_gradient.png` |
| `09_subset_sweep.py` | `subset_sweep.json`, `fig6_subset_sweep.png` |
| `10_tree_analysis.py` | `fig7_tree_analysis.png` |

---

## Critical facts Claude must know

### 1. Coordinate scale — the most common bug
FlyWire SWC files use **nanometers**. All morphoclass transforms and NBLAST
expect **micrometers**. The rescaling (÷1000) is applied inside `utils.py`
at parse time. If you ever see persistence diagrams that look identical across
cell types, or NBLAST warnings about non-micron data, this is the cause.

### 2. Recursion depth — fixed, do not revert
`utils.py::swc_to_persistence_diagram` uses iterative reverse-BFS to compute
max subtree tip distances. An earlier recursive version caused stack overflows
on neurons with ~2000+ nodes. Do not replace with a recursive version.

### 3. PersLay embedding dimension = 32
All downstream CKA comparisons assume 32-dim embeddings. If you change
`N_LANDMARKS` in `03_train_perslay.py`, update all CKA and kNN code.

### 4. Partition indices are fixed
`outputs/fafb/data/partitions.json` uses `random_state=42`. All 15 models
share the same partition boundaries. Never regenerate this file mid-experiment.

### 5. Test split is held out
The `test` key in `partitions.json` is **never** passed to the trainer.
Only `train` and `val` indices go to `03_train_perslay.py`.

### 6. Debiased CKA, not naive CKA — logit space is the primary metric
`utils.py::debiased_cka` uses the linear kernel K=XX^T with no L2-normalisation
(Kornblith 2019). L2-normalisation converts it to a cosine kernel and inflates
CKA to ~1.0 for always-positive PersLay features.

**Primary PRH metric: `cka_logit`** — computed on the 8-dim log-softmax output.
`cka_feat` (32-dim feature space) is informational only and is always ~1.0 due
to GaussianPointTransformer outputs being always non-negative.

### 7. NBLAST requires navis ≥ 1.8 and Python ≥ 3.9
The morphoclass venv is Python 3.8 — navis ≥ 1.6 requires Python ≥ 3.9.
`nblast.slurm` therefore uses `/gscratch/stf/emazuh/miniconda3/bin/python` (Python 3.13,
navis 1.11.0) instead of the standard morphoclass Python. Do not change this line.
To install/upgrade in the miniconda env: `/gscratch/stf/emazuh/miniconda3/bin/pip install navis`
The pretrained Drosophila scoring matrix is bundled with navis ≥ 1.8.

### 8. Training hyperparameters (sweep-validated)
```
LR = 5e-4, BATCH_SIZE = 32, N_EPOCHS = 500
```
These were selected by `11_hparam_sweep.py`. Do not revert to LR=5e-3 or
BATCH_SIZE=8 — those caused 12/15 models to collapse to predicting one class.

### 9. n=4,000 split sizes
With 500 neurons/class × 8 classes and 3-fold stratified partitions:
- **Train**: ~267/class (~2,133 total)
- **Val**: ~67/class (~533 total)
- **Test**: ~167/class (~1,333 total)

---

## Key numbers (n=4,000 — current experiment)

```
PersLay mean test accuracy:          0.647 ± 0.024
PersLay mean val  accuracy:          0.646 ± 0.024
CKA logit within-partition:          0.969 ± 0.024   (p < 0.001, null p95=0.0006)
CKA logit cross-partition:           0.969 ± 0.023   (p < 0.001)
CKA feat  within-partition:          1.000 ± 0.001   (inflated — not PRH metric)
Silhouette (PersLay feat):          -0.018 ± 0.011   (11/15 significant)
Silhouette (morphometric RF):        0.013
kNN Jaccard logit within-partition:  0.339 ± 0.068
kNN Jaccard logit cross-partition:   0.333 ± 0.066
kNN Jaccard feat  within-partition:  0.804 ± 0.048
RF baseline accuracy:                0.844 ± 0.015
```

NBLAST, cross-representation (step 08), subset sweep (step 09), and tree
analysis (step 10) numbers will be filled in once those jobs complete.

---

## Key numbers (n=96 — archived reference)

These are from the original 96-neuron validation run. Kept for comparison only.

```
PersLay within-partition CKA:   0.873 ± 0.052   (p < 0.001)
PersLay cross-partition CKA:    0.881 ± 0.060   (p < 0.001)
Permutation null p95:           0.027
PersLay mean test accuracy:     0.305
NBLAST-PCA32 accuracy:          0.740
NBLAST-PCA32 silhouette:        +0.155
CKA(PersLay, PI-PCA32):         0.508 ± 0.045
CKA(PersLay, NBLAST-PCA32):     0.151 ± 0.011
CKA(PersLay, Graph-Topology):   0.040 ± 0.012   (p = 0.111, not significant)
```

---

## Common debugging tasks

### "All persistence diagrams look identical"
Coordinate scale issue. Check:
```python
from experiments.fafb_perslay.utils import parse_swc
import numpy as np
arr = parse_swc("data/fafb_sample/swc/<any>.swc")
print(arr[:, 2:5].max(axis=0) - arr[:, 2:5].min(axis=0))
# Should print ~[50k, 50k, 20k] if in nm → divide by 1000
# Should print ~[50, 50, 20] if already in µm → no rescaling needed
```

### "RecursionError in persistence computation"
Should not happen — the fix is in `utils.py::swc_to_persistence_diagram`.
If it recurs, check that `utils.py` has not been replaced with an older version.
The signature to look for is `bfs_order` and `max_tip` dict built iteratively.

### "NBLAST warns about non-micron data"
The `07_nblast.py` script divides `nl / 1000` before making dotprops.
If you see this warning, check that line is present and not commented out.

### "CKA feat is trivially ~1.0"
Expected — GaussianPointTransformer outputs are always non-negative, so all
feature embeddings lie in the positive orthant of R^32. Use `cka_logit` instead.

### "Test accuracy near chance for all models"
Check that `test` split indices are not being passed to the trainer.
Only `train` and `val` keys from `partitions.json` should reach `03_train_perslay.py`.

### Adding a new figure
1. Add the plot function to the appropriate `0X_*.py` script
2. Call it from `main()`
3. Save to `outputs/fafb/figures/figN_<name>.png`
4. Add the output path to this CLAUDE.md table above
5. Commit with `git add experiments/fafb_perslay/0X_*.py outputs/fafb/figures/figN_*`

### Adding a new representation family
1. Add the embedding function to `utils.py` or a new `0X_*.py` script
2. Save embeddings to `outputs/fafb/embeddings/<name>_emb.npy`
3. Add CKA computation in `08_cross_representations.py`
4. The embedding must be shape `(N, D)` where N = number of neurons (same order as `labels.npy`)

---

## Git workflow

This project adds code to a fork of `IanQS/morphoclass`. Seven logical commits:

```
commit 1 — utils.py + requirements.txt + README.md
commit 2 — 00_sanity_checks.py + 01_make_dataset.py + 02_compute_persistence.py
commit 3 — 03_train_perslay.py + 04_extract_embeddings.py + 05_metrics.py + 06_figures.py
commit 4 — 07_nblast.py
commit 5 — 08_cross_representations.py
commit 6 — 09_subset_sweep.py
commit 7 — 10_tree_analysis.py
```

Full commit messages are in `GIT_COMMITS.md`.

### Quick commit helper

```bash
# Stage only experiment code (never stage src/morphoclass changes)
git add experiments/fafb_perslay/
git add outputs/fafb/figures/    # if committing updated figures
git status                        # confirm nothing from src/ is staged
git commit -m "fix: <what changed> in <script name>"
```

---

## Dependencies

```
numpy scipy pandas scikit-learn matplotlib seaborn
gudhi ripser persim umap-learn
navis >= 1.8        # NBLAST + SWC loading
networkx
# morphoclass — install from repo root: pip install -e .
```

Python binary for SLURM jobs: `/gscratch/scrubbed/emazuh/morphoclass/bin/python`

Check installed versions:
```bash
python -c "import navis; print(navis.__version__)"
python -c "import gudhi; print(gudhi.__version__)"
python -c "import morphoclass; print('morphoclass ok')"
```

---

## What NOT to do

- **Do not run training or persistence computation on the login node** — use sbatch.
- Do not import from `src/morphoclass/` directly in experiment scripts without
  checking the import works in the standalone numpy fallback path too.
- Do not hardcode absolute paths — use `Path("outputs/fafb/...")` relative to
  the repo root and run scripts from the repo root.
- Do not commit `outputs/fafb/models/*.pt` — these are large binary files.
  Commit only the scripts and figures.
- Do not run `09_subset_sweep.py` without the `--rerun` flag if you want
  fresh results — it caches to `subset_sweep.json` by default.
- Do not change the label column from `class` to `sub_class` or `hemilineage`
  mid-experiment — the partitions.json would need to be regenerated.
- Do not L2-normalise embeddings before passing to `debiased_cka` — that
  converts the linear kernel to a cosine kernel and inflates scores to ~1.0.
