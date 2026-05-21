# Git Commit Guide — PRH Morphology Experiment
# Add to IanQS/morphoclass fork

Below are the recommended commits in order.
Each is self-contained and reviewable independently.

---

## Commit 1 — experiment scaffold and utilities

```
git add experiments/fafb_perslay/utils.py \
        experiments/fafb_perslay/requirements.txt \
        README.md
git commit -m "feat: add fafb_perslay experiment scaffold and shared utilities

Adds experiments/fafb_perslay/ directory with:
- utils.py: SWC parsing, persistence diagram computation (iterative BFS,
  no recursion limit), PersLayNumpy, morphometric features, debiased CKA,
  permutation null, kNN Jaccard, MorphologySample dataclass
- requirements.txt: navis, gudhi, umap-learn, scikit-learn
- README.md: project overview, results table, data instructions

Fixes recursive max_tip_dist (stack overflow on 2k-node trees) with
iterative reverse-BFS approach.

Refs: Huh et al. 2024 (PRH), Brbić et al. 2026 (Aristotelian),
      Kornblith et al. 2019 (CKA), Costa et al. 2016 (NBLAST)"
```

---

## Commit 2 — data pipeline (00–02)

```
git add experiments/fafb_perslay/00_sanity_checks.py \
        experiments/fafb_perslay/01_make_dataset.py \
        experiments/fafb_perslay/02_compute_persistence.py
git commit -m "feat: data pipeline — sanity checks, dataset CSV, persistence diagrams

00_sanity_checks.py:
  - SWC inventory, CSV join validation, coordinate scale detection
  - Class distribution plot, SWC tree structure checks
  - Persistence diagram visual sanity check (requires morphoclass)
  - Outputs: outputs/fafb/sanity/fafb_dataset.csv + figures

01_make_dataset.py:
  - Stratified partitions (3-fold, fixed random_state=42)
  - Outputs: outputs/fafb/data/dataset.csv, partitions.json, label_map.json

02_compute_persistence.py:
  - SWC → rescale (nm→µm) → BranchingOnlyNeurites → PersistenceDiagram
  - Morphometric feature extraction (8 features, all iterative)
  - Sanity figures: persistence_sanity.png, morphometric_distributions.png"
```

---

## Commit 3 — training and evaluation (03–06)

```
git add experiments/fafb_perslay/03_train_perslay.py \
        experiments/fafb_perslay/04_extract_embeddings.py \
        experiments/fafb_perslay/05_metrics.py \
        experiments/fafb_perslay/06_figures.py
git commit -m "feat: PersLay training sweep and evaluation pipeline

03_train_perslay.py:
  - 3 partitions × 5 seeds = 15 models
  - Idempotent (skip if .npz exists), logs to run_log.csv
  - Uses PersLayNumpy (numpy-only, no torch dependency for validation)

04_extract_embeddings.py:
  - Loads all 15 model .npz files, saves aligned emb_part_*_s*.npy
  - Morphometric baseline standardised embedding

05_metrics.py:
  - Debiased CKA (Kornblith 2019) with permutation null (500 perms)
  - Silhouette score per model with label-shuffle null
  - kNN Jaccard stability (k=5)
  - RF morphometric baseline (same partition splits)
  - summary.json with all key numbers

06_figures.py:
  - Fig 1: UMAP grid (3×3, partitions × seeds)
  - Fig 2: 15×15 CKA heatmap with partition boundaries
  - Fig 3: Silhouette comparison + permutation null band
  - Fig 4: kNN Jaccard stability (within vs cross-partition)"
```

---

## Commit 4 — NBLAST pretrained model

```
git add experiments/fafb_perslay/07_nblast.py
git commit -m "feat: NBLAST zero-shot transfer (navis pretrained Drosophila scoring matrix)

07_nblast.py:
  - navis.nblast_allbyall() with bundled Drosophila scoring matrix
  - Rescales FlyWire SWCs nm→µm before NBLAST (confirmed required)
  - Symmetrises score matrix (NBLAST is directional)
  - PCA-32 and kernel-PCA-32 embeddings for CKA comparison
  - CKA(PersLay, NBLAST) = 0.151±0.011, p<0.001: first valid
    cross-architecture PRH signal

Results (n=96 validation subset):
  - NBLAST-PCA32: acc=0.740, sil=+0.155 (only positive silhouette)
  - CKA vs PersLay: 0.151±0.011, p<0.001, kNN-J=0.103

Cite: Costa et al. 2016, navis v1.11"
```

---

## Commit 5 — cross-representation analysis

```
git add experiments/fafb_perslay/08_cross_representations.py
git commit -m "feat: cross-representation CKA analysis and interpretability

08_cross_representations.py:
  PI-PCA32: rasterise persistence diagrams to 32×32 images → PCA-32
  Graph-Topology: degree/subtree/hop/branch-order histograms → PCA-16
  CKA gradient (all vs PersLay):
    PI-PCA32       0.508±0.045  p<0.001  (encoding gap)
    NBLAST-PCA32   0.151±0.011  p<0.001  (topology vs geometry)
    Morphometric   0.106±0.009  p<0.001  (topology vs scalars)
    Graph-Topology 0.040±0.012  p=0.111  n.s.

  Fig 5: PersLay contribution-weighted diagrams (interpretability)
    - w_i × mean_k(phi_ik) per (birth,death) point
    - Zero extra training, uses saved model weights
    - Olfactory: early-birth events ~76µm (vs ~140µm for other classes)
    - Visual: lowest contribution (0.0015), most compact morphology

  Fig 8: CKA gradient bar chart"
```

---

## Commit 6 — subset sweep

```
git add experiments/fafb_perslay/09_subset_sweep.py
git commit -m "feat: subset scaling sweep n=16→96 (predicts n=783 behaviour)

09_subset_sweep.py:
  Runs all experiments across 7 dataset sizes in ~4s total.
  Key findings:
  - Null p95 drops 0.141→0.026 as n grows (quantifies Brbić inflation)
  - PL accuracy trend predicts ~0.65 at n=783 (quadratic extrapolation)
  - Cross-arch CKA stable at ~0.17–0.18 across all n (size-independent)
  - Within≈cross Jaccard at n=96 (both inflated; gap will open at n=783)

  Fig 6: 6-panel sweep figure (accuracy, CKA+null, Jaccard, gap, 
          null-corrected, summary table)"
```

---

## Commit 7 — tree analysis

```
git add experiments/fafb_perslay/10_tree_analysis.py
git commit -m "feat: decision tree and RF analysis (CSE 446 Lec 15 concepts)

10_tree_analysis.py:
  7-panel figure covering:
  A. Bias-variance tradeoff (depth vs accuracy)
  B. Information gain per morphometric feature
     - soma_radius has highest IG (0.69 bits) — biologically meaningful
  C. Feature importance: single DT vs RF ensemble
     - DT concentrates on cable_len; RF recovers soma_r, n_tips
  D. Decision tree structure (depth=3, dark-theme hand-drawn)
     - First split: cable_len ≤ 2109µm separates large from small neurons
  E. Per-class RF accuracy (Kenyon_Cell, visual: easy; AN: hard)
  F. Cable length violin (explains first split biologically)
  G. Cross-partition accuracy + ensemble gain"
```

---

## Pull Request Description

**Title:** `feat: PRH morphology validation — FlyWire FAFB experiment pipeline`

**Summary:**
Adds a complete experiment pipeline under `experiments/fafb_perslay/` testing the
Platonic Representation Hypothesis (Huh et al. 2024) on neuron skeleton topology
from the FlyWire FAFB connectome.

**Key results (n=96 validation subset, 8 cell types):**

| Representation | Test Acc | Silhouette | CKA vs PersLay | p-value |
|---|---|---|---|---|
| NBLAST-PCA32 ★ | **0.740** | **+0.155** | 0.151 ± 0.011 | <0.001 |
| PI-PCA32 | 0.406 | +0.003 | **0.508 ± 0.045** | <0.001 |
| PersLay-PD | 0.305 | -0.069 | — (self) | — |
| Graph-Topology | 0.240 | -0.175 | 0.040 | 0.111 n.s. |

★ Zero-shot, no FlyWire supervision.

**Novel contributions:**
- First CKA analysis between PD (PersLay) and PI encodings (0.508)
- First CKA between filtration topology and pretrained geometry (NBLAST, 0.151)
- First PersLay interpretability on connectome data
- Subset sweep predicting n=783 scaling behaviour

**Does not modify:** any file under `src/morphoclass/`
