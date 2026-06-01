# PRH Morphology Validation — FlyWire FAFB
### Testing the Platonic Representation Hypothesis on Neuron Morphology

> **CSE 493G1 · Spring 2026 · University of Washington**

This repository extends [IanQS/morphoclass](https://github.com/IanQS/morphoclass)
with a complete experiment pipeline testing whether topological representations of
neuron morphology converge across independently trained models — a test of the
[Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (Huh et al. 2024).

---

## Quick Start (Hyak / SLURM)

> **All compute-heavy steps must be run via `sbatch` — never on the login node.**

```bash
# 1. Install dependencies (login node is fine for this)
pip install -r experiments/fafb_perslay/requirements.txt

# 2. Sample data from the full FlyWire archive
#    Edit TARGET_CLASSES in sample_dataset.py if needed, then:
python sample_dataset.py --per-class 500 --seed 42

# 3. Submit the full pipeline as a SLURM dependency chain
bash submit_all.sh              # prep → train → eval (full run)
bash submit_all.sh --skip-prep  # train → eval (data already prepared)
bash submit_all.sh --eval-only  # eval only (models already trained)

# 4. After eval completes, submit analysis steps
FIG=$(sbatch --parsable figures.slurm)
NB=$(sbatch --parsable --dependency=afterok:$FIG nblast.slurm)
sbatch --dependency=afterok:$NB analysis.slurm

# 5. Monitor jobs
squeue -u $USER
tail -f outputs/fafb/logs/prep_<JOBID>.out
tail -f outputs/fafb/logs/perslay_<JOBID>.out
tail -f outputs/fafb/logs/eval_<JOBID>.out
tail -f outputs/fafb/logs/nblast_<JOBID>.out
tail -f outputs/fafb/logs/analysis_<JOBID>.out
```

### SLURM scripts at a glance

| Script | Steps | Resources |
|--------|-------|-----------|
| `prep_data.slurm` | 01 + 02 — partitions + persistence diagrams | CPU, 8 cores, 32 GB, 2 h |
| `train_perslay.slurm` | 03 — train 15 PersLay models | GPU (A40), 16 GB, 2 h |
| `eval_pipeline.slurm` | 04 + 05 + 06 — embeddings, metrics, figures | CPU, 4 cores, 8 GB, 1 h |
| `figures.slurm` | 06 only — regenerate figures | CPU, 4 cores, 8 GB, 30 min |
| `nblast.slurm` | 07 — NBLAST all-by-all | CPU, 16 cores, 64 GB, 8 h |
| `analysis.slurm` | 08 + 09 + 10 — cross-rep, subset sweep, tree | CPU, 4 cores, 16 GB, 1 h |

**Note:** `nblast.slurm` uses `/gscratch/stf/emazuh/miniconda3/bin/python` (Python 3.13,
navis 1.11.0) because navis ≥ 1.6 requires Python ≥ 3.9 and the morphoclass venv is
Python 3.8.

---

## Repository Layout

```
morphoclass/                        ← upstream library (DO NOT MODIFY)

experiments/
└── fafb_perslay/                   ← all experiment code lives here
    ├── requirements.txt
    ├── utils.py                    ← shared primitives (SWC, PersLay, CKA, metrics)
    ├── 00_sanity_checks.py         ← coordinate scale, SWC validity, class balance
    ├── 01_make_dataset.py          ← CSV + stratified partitions
    ├── 02_compute_persistence.py   ← SWC → persistence diagrams + morphometrics
    ├── 03_train_perslay.py         ← 3 partitions × 5 seeds = 15 PersLay models
    ├── 04_extract_embeddings.py    ← feat (32-dim) + logit (8-dim) embeddings
    ├── 05_metrics.py               ← CKA, silhouette, kNN Jaccard, RF baseline
    ├── 06_figures.py               ← Figs 1–4 (UMAP, CKA heatmap, silhouette, kNN)
    ├── 07_nblast.py                ← NBLAST pretrained model (navis)
    ├── 08_cross_representations.py ← PI-PCA, graph-topology, cross-rep CKA (Fig 5)
    ├── 09_subset_sweep.py          ← scaling sweep (Fig 6)
    ├── 10_tree_analysis.py         ← decision tree / RF analysis (Fig 7)
    └── 11_hparam_sweep.py          ← hyperparameter grid search (GPU)

data/
└── fafb_sample/
    ├── swc/                        ← sampled SWC files (extracted from full zip)
    ├── dataset.csv                 ← path↔label manifest for the pipeline
    ├── classification_subset.csv   ← full metadata for sampled neurons
    ├── root_ids.txt                ← all sampled root IDs, one per line
    └── viewer_sample*.csv          ← 5/class subsets for neuron viewer

outputs/
└── fafb/
    ├── data/        ← dataset.csv, partitions.json, diagrams.npz, label_map.json
    ├── models/      ← perslay_part_*_s*.pt (15 files), run_log.csv
    ├── embeddings/  ← emb_*.npy (feat), logit_*.npy, nblast_*.npy, labels.npy
    ├── metrics/     ← summary.json, cka_logit_matrix.npy, knn_jaccard.csv …
    ├── figures/     ← fig1_*.png … fig8_*.png
    └── logs/        ← SLURM *.out / *.err per job

sample_dataset.py   ← sample N neurons/class from the full 13 GB zip
submit_all.sh       ← SLURM dependency chain orchestrator
```

---

## Data

**FlyWire FAFB** full archive (Hyak):
- SWC zip: `/gscratch/scrubbed/emazuh/otterpack/data/sk_lod1_783_healed.zip`
- Labels:  `/gscratch/scrubbed/emazuh/otterpack/data/classification.csv.gz`

`sample_dataset.py` extracts only the selected neurons from the zip — it does **not**
unpack all 13 GB. Edit `TARGET_CLASSES` in that file to change which classes to include,
then run `python sample_dataset.py --per-class 500 --seed 42`.

**Current sample:** 4,000 neurons — 500/class × 8 classes, seed=42.  
**Next planned run:** ~6,586 neurons — 500/class × 20 classes (all classes ≥ 50 neurons).

---

## Key Results

### n=4,000 (current)

| Metric | Value |
|--------|-------|
| PersLay test accuracy | 0.647 ± 0.024 |
| PersLay test macro-F1 | 0.620 ± 0.025 |
| **CKA logit within-partition** ★ | **0.969 ± 0.024** (p < 0.001, null p95 = 0.0006) |
| **CKA logit cross-partition** ★ | **0.969 ± 0.023** (p < 0.001) |
| kNN Jaccard logit within-partition | 0.339 ± 0.068 |
| kNN Jaccard logit cross-partition | 0.333 ± 0.066 |
| Silhouette (PersLay feat) | −0.018 ± 0.011 |
| RF baseline accuracy | 0.844 ± 0.015 |

★ Primary PRH metric. Feature-space CKA is ~1.0 for all model pairs including random
initialisations — this is an artifact of GaussianPointTransformer outputs always being
non-negative. Logit space (8-dim log-softmax) is the valid PRH metric.

### n=96 (archived reference)

| Representation | Test Acc | Silhouette | CKA vs PersLay |
|---|---|---|---|
| **NBLAST-PCA32 ★** | **0.740** | **+0.155** | 0.151 ± 0.011 |
| PI-PCA32 | 0.406 | +0.003 | **0.508 ± 0.045** |
| PersLay-PD | 0.305 | −0.069 | — (self) |
| Graph-Topology | 0.240 | −0.175 | 0.040 ± 0.012 n.s. |

★ Pretrained — zero-shot transfer, no FlyWire label supervision.

---

## Honest Caveats

1. **Negative silhouette in feature space.** PersLay embeddings do not geometrically
   separate classes (silhouette = −0.018) while RF on morphometrics hits 0.844. PersLay
   is learning something predictive (CKA high, accuracy 0.647) but the convergence
   is in logit space, not embedding space.
2. **CKA vs kNN Jaccard gap.** CKA logit ≈ 0.97 but kNN Jaccard logit ≈ 0.34.
   High CKA does not imply identical neighborhood structure — worth flagging in the paper.
3. **NBLAST uses 3D coordinates**, so its high accuracy partly reflects spatial
   position (brain region) rather than branching topology alone.
4. **Within ≈ cross-partition CKA** is the key PRH result: representations converge
   regardless of which neurons the model was trained on.

---

## References

- Huh et al. 2024 — [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)
- Brbić et al. 2026 — [Aristotelian Representation Hypothesis](https://arxiv.org/abs/2602.14486)
- Kornblith et al. 2019 — [CKA](https://arxiv.org/abs/1905.00414)
- Costa et al. 2016 — [NBLAST](https://www.cell.com/neuron/fulltext/S0896-6273(16)30265-3)
- Carrière et al. 2020 — [PersLay](https://arxiv.org/abs/1904.09378)
- Dorkenwald et al. 2023 — [FlyWire](https://www.nature.com/articles/s41592-022-01697-0)
- Kanari et al. 2018 — [TMD](https://link.springer.com/article/10.1007/s12021-017-9341-1)
