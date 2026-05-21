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
    └── 10_tree_analysis.py
data/
└── fafb/
    ├── swc/                  ← *.swc skeleton files (one per neuron)
    └── classification.csv    ← root_id, flow, super_class, class, ...
outputs/
└── fafb/
    ├── data/                 ← dataset.csv, partitions.json, diagrams.npz
    ├── models/               ← perslay_part_*_s*.npz, run_log.csv
    ├── embeddings/           ← emb_*.npy, nblast_*.npy, pi_*.npy
    ├── metrics/              ← summary.json, cka_matrix.npy, sweep.json
    └── figures/              ← fig1_*.png … fig8_*.png
```

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

### 6. Debiased CKA, not naive CKA
`utils.py::debiased_cka` removes diagonal before summing (Kornblith 2019).
Do not replace with naive `np.trace(K @ L) / (||K||_F * ||L||_F)`.
At n=96, the bias correction matters significantly.

### 7. NBLAST requires navis ≥ 1.8
`pip install navis` — check version before running `07_nblast.py`.
The pretrained Drosophila scoring matrix is bundled with navis.

---

## How to run the full pipeline

```bash
# From repo root. Run in order — each script depends on previous outputs.

# Step 0: validate data before touching any training code
python experiments/fafb_perslay/00_sanity_checks.py \
    --swc_dir data/fafb/swc \
    --csv data/fafb/classification.csv

# Step 1: build dataset CSV and stratified partitions
python experiments/fafb_perslay/01_make_dataset.py \
    --dataset data/fafb/classification.csv \
    --n_splits 3 \
    --n_seeds 5 \
    --min_per_class 50

# Step 2: compute persistence diagrams and morphometric features
python experiments/fafb_perslay/02_compute_persistence.py

# Step 3: train 15 PersLay models (3 partitions × 5 seeds)
# Full sweep (sequential):
python experiments/fafb_perslay/03_train_perslay.py
# Single model (for debugging):
python experiments/fafb_perslay/03_train_perslay.py --partition part_0 --seed 0

# Step 4: extract and align embeddings
python experiments/fafb_perslay/04_extract_embeddings.py

# Step 5: compute all metrics (CKA, silhouette, kNN Jaccard, RF baseline)
python experiments/fafb_perslay/05_metrics.py

# Step 6: generate core figures (Figs 1–4)
python experiments/fafb_perslay/06_figures.py

# Step 7: NBLAST pretrained model (needs navis)
python experiments/fafb_perslay/07_nblast.py
# Force recompute if scores already exist:
python experiments/fafb_perslay/07_nblast.py --force

# Step 8: cross-representation analysis + interpretability (Figs 5, 8)
python experiments/fafb_perslay/08_cross_representations.py

# Step 9: subset scaling sweep n=16→96 (Fig 6)
python experiments/fafb_perslay/09_subset_sweep.py
# Force rerun (ignores cached subset_sweep.json):
python experiments/fafb_perslay/09_subset_sweep.py --rerun

# Step 10: decision tree / RF analysis (Fig 7)
python experiments/fafb_perslay/10_tree_analysis.py
```

---

## What each script produces

| Script | Key outputs |
|--------|-------------|
| `00_sanity_checks.py` | `outputs/fafb/sanity/class_distribution.png`, `fafb_dataset.csv` |
| `01_make_dataset.py` | `outputs/fafb/data/dataset.csv`, `partitions.json`, `label_map.json` |
| `02_compute_persistence.py` | `diagrams.npz`, `morphometrics.npy`, `persistence_sanity.png` |
| `03_train_perslay.py` | `models/perslay_part_{0,1,2}_s{0..4}.npz` (15 files), `run_log.csv` |
| `04_extract_embeddings.py` | `embeddings/emb_part_{0,1,2}_s{0..4}.npy` (15 files), `labels.npy` |
| `05_metrics.py` | `cka_matrix.npy`, `silhouette.csv`, `knn_jaccard.csv`, `summary.json` |
| `06_figures.py` | `fig1_umap_grid.png`, `fig2_cka_heatmap.png`, `fig3_silhouette.png`, `fig4_knn_stability.png` |
| `07_nblast.py` | `nblast_scores.npy`, `nblast_pca32_emb.npy`, `nblast_results.json` |
| `08_cross_representations.py` | `pi_pca_emb.npy`, `graph_topology_emb.npy`, `fig5_interpretability.png`, `fig8_cka_gradient.png` |
| `09_subset_sweep.py` | `subset_sweep.json`, `fig6_subset_sweep.png` |
| `10_tree_analysis.py` | `fig7_tree_analysis.png` |

---

## Generating individual figures

All figures read from `outputs/fafb/` — they can be regenerated at any time
without rerunning training:

```bash
# Regenerate all core figures (Figs 1–4)
python experiments/fafb_perslay/06_figures.py

# Regenerate interpretability + CKA gradient (Figs 5, 8)
python experiments/fafb_perslay/08_cross_representations.py

# Regenerate subset sweep figure only (uses cached JSON)
python experiments/fafb_perslay/09_subset_sweep.py

# Regenerate tree analysis figure
python experiments/fafb_perslay/10_tree_analysis.py
```

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

When committing a new figure or fix, use this pattern:

```bash
# Stage only experiment code (never stage src/morphoclass changes)
git add experiments/fafb_perslay/
git add outputs/fafb/figures/    # if committing updated figures
git status                        # confirm nothing from src/ is staged
git commit -m "fix: <what changed> in <script name>"
```

### Branch name

```bash
git checkout -b feat/fafb-prh-experiment
```

---

## Common debugging tasks

### "All persistence diagrams look identical"
Coordinate scale issue. Check:
```python
from experiments.fafb_perslay.utils import parse_swc
import numpy as np
arr = parse_swc("data/fafb/swc/<any>.swc")
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

### "CKA is very high (>0.95) at small n"
Expected — this is the Brbić (2026) inflation effect. At n=16–32 with 32-dim
embeddings the permutation null is also high (0.10–0.14). Report null-corrected
values. Use kNN Jaccard as the primary metric at small n.

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

## Key numbers (n=96 validation subset)

These are the ground-truth results to compare against when debugging:

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

If any of these numbers change substantially, something in the pipeline has broken.

---

## Dependencies

```
numpy scipy pandas scikit-learn matplotlib seaborn
gudhi ripser persim umap-learn
navis >= 1.8        # NBLAST + SWC loading
networkx
# morphoclass — install from repo root: pip install -e .
```

Check installed versions:
```bash
python -c "import navis; print(navis.__version__)"
python -c "import gudhi; print(gudhi.__version__)"
python -c "import morphoclass; print('morphoclass ok')"
```

---

## What NOT to do

- Do not import from `src/morphoclass/` directly in experiment scripts without
  checking the import works in the standalone numpy fallback path too.
- Do not hardcode absolute paths — use `Path("outputs/fafb/...")` relative to
  the repo root and run scripts from the repo root.
- Do not commit `outputs/fafb/models/*.npz` — these are large binary files.
  Commit only the scripts and figures.
- Do not run `09_subset_sweep.py` without the `--rerun` flag if you want
  fresh results — it caches to `subset_sweep.json` by default.
- Do not change the label column from `class` to `sub_class` or `hemilineage`
  mid-experiment — the partitions.json would need to be regenerated.
