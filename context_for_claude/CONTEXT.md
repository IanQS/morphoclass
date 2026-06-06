# PRH Morphology — Context for Claude Chat
## Next steps: improving geometric class separation in PersLay embeddings

---

## The core problem

We are testing the **Platonic Representation Hypothesis (PRH)** on FlyWire FAFB neuron
morphology. PersLay/CorianderNet models trained on different data subsets converge strongly
in **logit space** (CKA ≈ 0.90–0.97, p < 0.001), confirming PRH. However the embeddings
**do not geometrically separate cell types** — silhouette scores are negative at every scale
tested, while a Random Forest on hand-crafted morphometrics achieves 0.844 accuracy.

```
                    n=4,000 (8 cls)   n=17,288 (20 cls)
Test accuracy           0.647             0.487
CKA logit within        0.969             0.890
CKA logit cross         0.969             0.899
Silhouette (feat)      -0.018            -0.17 (partial)
RF baseline acc         0.844             —
Null p95               0.0006            0.000
```

**Interpretation:** PersLay learns a representation that converges across models (PRH holds)
but the 32-dim feature space is geometrically disorganised — neurons from different cell types
intermingle. The convergence is happening in the classifier's output, not the embedding space.

---

## Architecture

```
SWC skeleton
    ↓ Vietoris-Rips filtration (nm → µm ÷1000)
Persistence diagram  [(birth, death), ...]
    ↓ GaussianPointTransformer (32 landmarks, Gaussian soft-counts)
32-dim feature vector  [always ≥ 0 by construction]
    ↓ Scatter pooling + MLP
20-dim log-softmax logits  ← PRH metric lives here
```

**Key constraint:** GaussianPointTransformer outputs are always non-negative (Gaussian
soft-counts). All 32-dim feature embeddings live in the positive orthant of R^32. This
means linear-kernel CKA on features is trivially ~1.0 for any two models, and silhouette
scores are bounded to be negative or near-zero (cosine distances are bounded 0–90°).

**The geometric organisation question:** can we reshape the 32-dim feature space so that
same-class neurons cluster together, while preserving the PRH convergence property?

---

## What we've tried

### 1. Standard CE training (done)
- 500 epochs, lr=5e-4, bs=32
- Silhouette at n=4k: -0.018; at n=17k: -0.17
- CKA convergence: strong

### 2. Supervised Contrastive Loss (SupCon, Khosla 2020) — no warmup
- Combined loss: 0.5×CE + 0.5×SupCon(τ=0.07)
- 300 epochs, class-balanced batches (32/class × 20 classes = 640/step)
- Result: silhouette **-0.293** — worse than CE alone
- Reason: without warmup, SupCon dominates early training before classification
  is stable, pushing embeddings into a geometry not aligned with cell types

### 3. CE warmup then SupCon (running)
- 200 epochs pure CE → 300 epochs 0.5×CE + 0.5×SupCon
- val_acc reached 0.346 at epoch 500 (vs 0.487 for standard CE)
- Silhouette result pending

---

## Open questions / next steps

1. **Does CE warmup + SupCon improve silhouette?** (results pending from running job)
   If yes → sweep λ and τ. If no → SupCon may not be the right tool here.

2. **Why is silhouette negative even though accuracy is ~0.49–0.65?**
   The classifier learns discriminative features but doesn't need to organise them
   geometrically. The positive-orthant constraint on GaussianPointTransformer may
   be the fundamental blocker.

3. **Triplet loss vs SupCon:** SupCon uses all positives in the batch; triplet uses one.
   For 20 classes with 32/class per batch, SupCon has 31 positives per anchor.
   Triplet loss might be more stable.

4. **Projected head for contrastive:** Standard practice (Chen et al. SimCLR 2020,
   Khosla SupCon 2020) uses a small 2-layer MLP projection head for the contrastive
   loss, NOT the main feature extractor directly. This lets the backbone features
   develop freely while the projected space is shaped by contrastive objectives.
   We are currently applying SupCon directly to the 32-dim features — adding a
   projection head might be the key missing piece.

5. **Can we measure convergence in a different way?** RSA (Representational Similarity
   Analysis) using rank correlations might be more robust to the positive-orthant issue
   than CKA. Worth computing as a sanity check.

6. **Does NBLAST (CKA≈0.15 at n=96) improve at n=17k?** NBLAST doesn't have the
   positive-orthant constraint — its PCA32 space clusters properly (silhouette +0.155
   at n=96). CKA(PersLay, NBLAST) is the cross-architecture PRH test.

---

## Relevant code files (included)

- `utils.py` — `debiased_cka`, `parse_swc`, `swc_to_persistence_diagram`
- `03_train_perslay.py` — standard CE training
- `13_pretrain_contrastive.py` — SupCon + CE with warmup
- `04_extract_embeddings.py` — extracts feat (32-dim) + logit (20-dim) per model
- `05_metrics.py` — CKA, silhouette, kNN Jaccard, RF baseline

---

## Figures included

- `fig1_umap_grid.png` — UMAP of PersLay feature embeddings coloured by cell type
  (shows poor geometric separation — 20 classes intermingle)
- `fig2_cka_heatmap.png` — 15×15 CKA matrix in logit space (shows convergence)
- `fig3_silhouette.png` — per-model silhouette scores (all negative)
- `report_fig1_cka_heatmap.png` — CKA heatmap from progress report
- `report_fig2_accuracy.png` — per-model test accuracy
- `report_fig3_knn_silhouette.png` — kNN Jaccard and silhouette combined
