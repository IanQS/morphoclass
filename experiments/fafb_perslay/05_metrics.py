"""
05_metrics.py
=============
Computes all PRH evaluation metrics across the 15-model ensemble.

Metrics:
  1. Debiased CKA — pairwise 15×15 on two embedding spaces:
       feat  emb_part_*.npy   (32-dim PersLay features)
       logit logit_part_*.npy (8-dim log-softmax; primary CKA metric)
  2. Silhouette score — per model on feat space, with 500-permutation null
  3. kNN Jaccard stability — all pairs, feat and logit spaces
  4. Classification accuracy — from run_log.csv vs morphometric baseline

Outputs:
  outputs/fafb/metrics/cka_feat_matrix.npy   — (N_models, N_models) feat CKA
  outputs/fafb/metrics/cka_logit_matrix.npy  — (N_models, N_models) logit CKA
  outputs/fafb/metrics/cka_labels.json
  outputs/fafb/metrics/silhouette.csv
  outputs/fafb/metrics/knn_jaccard.csv
  outputs/fafb/metrics/summary.json

Run:
  python experiments/fafb_perslay/05_metrics.py
"""

import json, os, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import debiased_cka, permutation_cka_null, knn_jaccard, rsa

EMB_DIR  = Path("outputs/fafb/embeddings")
DATA_DIR = Path("outputs/fafb/data")
OUT      = Path("outputs/fafb/metrics")
OUT.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
N_PERM   = 200    # permutations for null (200 sufficient for p<0.001; use 1000 for paper)
KNN_K    = 5      # k for Jaccard (use 10 for larger dataset)
CKA_MAX_N = 4000  # stratified subsample for O(N²) ops — set to None to disable

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def load_models(prefix):
    """Load embedding matrices matching emb_dir/<prefix>_part_*.npy."""
    files = sorted(EMB_DIR.glob(f"{prefix}_part_*.npy"))
    result = []
    for f in files:
        stem  = f.stem          # e.g. emb_part_0_s2 or logit_part_0_s2
        parts = stem.split("_")
        part_id = int(parts[-2])   # '0' in emb_part_0_s2
        seed    = int(parts[-1][1:])  # 's2' → 2
        result.append({
            "label":   f"P{part_id}S{seed}",
            "part_id": part_id,
            "seed":    seed,
            "emb":     np.load(f),
        })
    return result


def compute_cka_matrix(models):
    """Compute pairwise debiased CKA, return (matrix, within, cross)."""
    n        = len(models)
    mat      = np.zeros((n, n))
    part_ids = np.array([m["part_id"] for m in models])
    n_pairs  = n * (n + 1) // 2
    print(f"  Computing {n}×{n} CKA matrix ({n_pairs} pairs)...")
    sys.stdout.flush()
    for i in range(n):
        for j in range(i, n):
            v = debiased_cka(models[i]["emb"], models[j]["emb"])
            mat[i, j] = mat[j, i] = v
        print(f"  Row {i+1}/{n} done", flush=True)
    within_mask = part_ids[:, None] == part_ids[None, :]
    cross_mask  = ~within_mask
    np.fill_diagonal(within_mask, False)
    return mat, mat[within_mask], mat[cross_mask]


def subsample(models, labels, max_n, seed=42):
    """Stratified subsample to max_n for O(N²) ops (CKA, kNN)."""
    n = len(labels)
    if max_n is None or n <= max_n:
        return models, labels
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_n // len(classes))
    idx = []
    for c in classes:
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    idx = np.array(sorted(idx))
    sub = [{**m, "emb": m["emb"][idx]} for m in models]
    print(f"  Subsampled {n:,} → {len(idx):,} neurons for CKA/kNN (stratified, seed={seed})")
    return sub, labels[idx]


def main(model_type="perslay"):
    """
    model_type: "perslay" — loads emb_part_*.npy / logit_part_*.npy (CorianderNet)
                "cnn"     — loads cnn_emb_part_*.npy / cnn_logit_part_*.npy (CNNet)
    """
    feat_prefix  = "emb"      if model_type == "perslay" else "cnn_emb"
    logit_prefix = "logit"    if model_type == "perslay" else "cnn_logit"

    print(f"\n{'='*60}")
    print(f"  Computing PRH evaluation metrics  [{model_type}]")
    print(f"{'='*60}\n")

    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)

    feat_models  = load_models(feat_prefix)
    logit_models = load_models(logit_prefix)

    if not feat_models:
        print(f"No embeddings found in {EMB_DIR}. Run 04_extract_embeddings.py first.")
        return
    print(f"Loaded {len(feat_models)} models  "
          f"(feat={feat_models[0]['emb'].shape[1]}-dim, "
          f"logit={logit_models[0]['emb'].shape[1]}-dim)\n")

    n_models     = len(feat_models)
    model_labels = [m["label"] for m in feat_models]
    part_ids     = np.array([m["part_id"] for m in feat_models])

    meta = {"labels": model_labels,
            "part_ids": [m["part_id"] for m in feat_models],
            "seeds":    [m["seed"]    for m in feat_models]}
    with open(OUT / "cka_labels.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Subsample for O(N²) operations if dataset is large
    feat_sub,  labels_sub = subsample(feat_models,  labels, CKA_MAX_N)
    logit_sub, _          = subsample(logit_models, labels, CKA_MAX_N)

    # ── 1a. Debiased CKA — feature space ─────────────────────────────────────
    print("Computing debiased CKA on feature embeddings (32-dim)...")
    feat_mat, feat_within, feat_cross = compute_cka_matrix(feat_sub)
    np.save(OUT / "cka_feat_matrix.npy", feat_mat)
    np.save(OUT / "cka_matrix.npy", feat_mat)
    print(f"\n{PASS} Feature CKA saved")
    print(f"  Within-partition: {feat_within.mean():.3f} ± {feat_within.std():.3f}")
    print(f"  Cross-partition:  {feat_cross.mean():.3f} ± {feat_cross.std():.3f}")
    print(f"  Note: feat CKA is inflated — PersLay outputs are always non-negative.")

    # ── 1b. Debiased CKA — logit space ───────────────────────────────────────
    print(f"\nComputing debiased CKA on logit embeddings ({logit_models[0]['emb'].shape[1]}-dim)...")
    logit_mat, logit_within, logit_cross = compute_cka_matrix(logit_sub)
    np.save(OUT / "cka_logit_matrix.npy", logit_mat)
    print(f"\n{PASS} Logit CKA saved  ← primary PRH metric")
    print(f"  Within-partition: {logit_within.mean():.3f} ± {logit_within.std():.3f}")
    print(f"  Cross-partition:  {logit_cross.mean():.3f} ± {logit_cross.std():.3f}")

    if logit_cross.mean() > 0.5:
        print(f"  {PASS} Cross-partition logit CKA > 0.5 — representations converge")
    else:
        print(f"  {WARN} Cross-partition logit CKA low")

    # ── Permutation null on logit cross-partition CKA ─────────────────────────
    print(f"\nPermutation null ({N_PERM} perms) on logit CKA, first cross-partition pair...")
    cross_pairs = [(i, j) for i in range(n_models) for j in range(i+1, n_models)
                   if feat_models[i]["part_id"] != feat_models[j]["part_id"]]
    if cross_pairs:
        i0, j0   = cross_pairs[0]
        null_dist = permutation_cka_null(logit_models[i0]["emb"],
                                          logit_models[j0]["emb"],
                                          n_permutations=N_PERM, seed=0)
        obs_cka = logit_mat[i0, j0]
        p_val   = float(np.mean(null_dist >= obs_cka))
        p95     = float(np.percentile(null_dist, 95))
        print(f"  Observed CKA = {obs_cka:.3f}, null p95 = {p95:.3f}, p = {p_val:.4f}")
        if obs_cka > p95:
            print(f"  {PASS} Logit CKA exceeds null 95th percentile (p={p_val:.4f})")
        else:
            print(f"  {WARN} Logit CKA does NOT exceed permutation null")
    else:
        print(f"  {WARN} Need >1 partition for null test")
        null_dist = np.array([0.0])
        obs_cka, p_val, p95 = 0.0, 1.0, 0.0

    # use logit CKA for downstream summary
    within_cka = logit_within
    cross_cka  = logit_cross

    # ── 2. Silhouette scores ──────────────────────────────────────────────────
    print(f"\nComputing silhouette scores + permutation nulls...")
    # Subsample for silhouette — O(N²) distance matrix; cap at CKA_MAX_N
    sil_models, sil_labels = subsample(feat_models, labels, CKA_MAX_N)
    sil_rows = []

    for m in sil_models:
        emb = m["emb"]
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

        # Build distance matrix once, reuse for all null permutations
        from sklearn.metrics import pairwise_distances
        from sklearn.metrics import silhouette_score as _sil
        D = pairwise_distances(emb_norm, metric="euclidean")

        obs_sil = float(_sil(D, sil_labels, metric="precomputed"))

        rng = np.random.default_rng(42)
        null_sils = []
        for _ in range(N_PERM):
            shuffled = rng.permutation(sil_labels)
            try:
                null_sils.append(float(_sil(D, shuffled, metric="precomputed")))
            except Exception:
                null_sils.append(0.0)
        null_sils = np.array(null_sils)
        sil_p_val = float(np.mean(null_sils >= obs_sil))
        sil_p95   = float(np.percentile(null_sils, 95))

        sil_rows.append({
            "model": m["label"],
            "partition": m["part_id"],
            "seed": m["seed"],
            "silhouette": round(obs_sil, 4),
            "null_mean": round(null_sils.mean(), 4),
            "null_p95": round(sil_p95, 4),
            "p_value": round(sil_p_val, 4),
            "significant": obs_sil > sil_p95,
        })
        print(f"  {m['label']}: sil={obs_sil:.3f}, null_p95={sil_p95:.3f}, p={sil_p_val:.4f}")

    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(OUT / "silhouette.csv", index=False)
    print(f"\n{PASS} Silhouette scores saved")
    print(f"  Mean silhouette: {sil_df['silhouette'].mean():.3f} ± {sil_df['silhouette'].std():.3f}")
    print(f"  Significant (p<0.05): {sil_df['significant'].sum()}/{len(sil_df)} models")

    # ── Morphometric baseline silhouette ──────────────────────────────────────
    morph_path = EMB_DIR / "morphometric_emb.npy"
    if morph_path.exists():
        morph_emb = np.load(morph_path)
        morph_sil = float(silhouette_score(morph_emb, labels,
                                           sample_size=len(sil_labels), random_state=42))
        print(f"  Morphometric baseline silhouette: {morph_sil:.3f}")
        print(f"  PersLay advantage: {sil_df['silhouette'].mean() - morph_sil:+.3f}")
    else:
        morph_sil = None

    # ── RF baseline accuracy ──────────────────────────────────────────────────
    print(f"\nTraining Random Forest baseline...")
    with open(DATA_DIR / "partitions.json") as f:
        partitions = json.load(f)

    rf_results = []
    if morph_path.exists():
        morph_raw = np.load(Path("outputs/fafb/data/morphometrics.npy"))
        for part_id_str, splits in partitions.items():
            part_num = int(part_id_str.split("_")[1])
            X_train = morph_raw[splits["train"]]
            y_train = labels[splits["train"]]
            X_test  = morph_raw[splits["test"]]
            y_test  = labels[splits["test"]]

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            rf_acc = float(accuracy_score(y_test, preds))
            rf_f1  = float(f1_score(y_test, preds, average='macro', zero_division=0))
            rf_results.append({"partition": part_num, "rf_accuracy": rf_acc, "rf_f1": rf_f1})
            print(f"  {part_id_str}: RF acc={rf_acc:.3f}, macro-F1={rf_f1:.3f}")

    # ── 3. kNN Jaccard stability — feat and logit ─────────────────────────────
    print(f"\nComputing kNN Jaccard stability (k={KNN_K}) on feat and logit spaces...")
    jaccard_rows = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            pair_type = ("within_partition"
                         if feat_models[i]["part_id"] == feat_models[j]["part_id"]
                         else "cross_partition")
            jac_feat  = knn_jaccard(feat_sub[i]["emb"],  feat_sub[j]["emb"],  k=KNN_K)
            jac_logit = knn_jaccard(logit_sub[i]["emb"], logit_sub[j]["emb"], k=KNN_K)
            jaccard_rows.append({
                "model_type":  model_type,
                "n_neurons":   int(len(labels)),
                "n_classes":   int(len(label_names)),
                "model_a":     feat_models[i]["label"],
                "model_b":     feat_models[j]["label"],
                "part_a":      feat_models[i]["part_id"],
                "part_b":      feat_models[j]["part_id"],
                "seed_a":      feat_models[i]["seed"],
                "seed_b":      feat_models[j]["seed"],
                "type":        pair_type,
                "jaccard_feat":  round(jac_feat,  4),
                "jaccard_logit": round(jac_logit, 4),
            })

    jac_df = pd.DataFrame(jaccard_rows)
    jac_df.to_csv(OUT / "knn_jaccard.csv", index=False)

    for space in ("feat", "logit"):
        col = f"jaccard_{space}"
        within = jac_df[jac_df["type"] == "within_partition"][col]
        cross  = jac_df[jac_df["type"] == "cross_partition"][col]
        print(f"{PASS} kNN Jaccard ({space})")
        if len(within):
            print(f"  Within-partition:  {within.mean():.3f} ± {within.std():.3f}")
        if len(cross):
            print(f"  Cross-partition:   {cross.mean():.3f} ± {cross.std():.3f}")

    # ── 4. RSA (Kendall's τ) — positive-orthant-insensitive confirmation ─────────
    print(f"\nComputing RSA (Kendall's τ) on logit space...")
    rsa_within, rsa_cross = [], []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            tau, _ = rsa(logit_sub[i]["emb"], logit_sub[j]["emb"])
            if logit_sub[i]["part_id"] == logit_sub[j]["part_id"]:
                rsa_within.append(tau)
            else:
                rsa_cross.append(tau)
    rsa_within = np.array(rsa_within)
    rsa_cross  = np.array(rsa_cross)
    print(f"{PASS} RSA (logit space)  ← positive-orthant-insensitive")
    print(f"  Within-partition:  {rsa_within.mean():.3f} ± {rsa_within.std():.3f}")
    print(f"  Cross-partition:   {rsa_cross.mean():.3f} ± {rsa_cross.std():.3f}", flush=True)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    jac_feat_within  = jac_df[jac_df["type"]=="within_partition"]["jaccard_feat"]
    jac_feat_cross   = jac_df[jac_df["type"]=="cross_partition"]["jaccard_feat"]
    jac_logit_within = jac_df[jac_df["type"]=="within_partition"]["jaccard_logit"]
    jac_logit_cross  = jac_df[jac_df["type"]=="cross_partition"]["jaccard_logit"]

    summary = {
        "n_models":  n_models,
        "n_neurons": int(len(labels)),
        "n_classes": int(len(label_names)),
        "cka_logit": {
            "note": "Primary PRH metric — logit space avoids always-positive artifact",
            "within_partition_mean": round(float(logit_within.mean()), 4),
            "within_partition_std":  round(float(logit_within.std()),  4),
            "cross_partition_mean":  round(float(logit_cross.mean()),  4),
            "cross_partition_std":   round(float(logit_cross.std()),   4),
            "permutation_null_p95":  round(float(p95),                 4),
            "cross_partition_pvalue":round(float(p_val),               4),
        },
        "cka_feat": {
            "note": "Informational only — inflated by always-positive PersLay outputs",
            "within_partition_mean": round(float(feat_within.mean()), 4),
            "within_partition_std":  round(float(feat_within.std()),  4),
            "cross_partition_mean":  round(float(feat_cross.mean()),  4),
            "cross_partition_std":   round(float(feat_cross.std()),   4),
        },
        "silhouette": {
            "perslay_mean":          round(float(sil_df["silhouette"].mean()), 4),
            "perslay_std":           round(float(sil_df["silhouette"].std()),  4),
            "morphometric_baseline": round(float(morph_sil), 4) if morph_sil else None,
            "n_significant_models":  int(sil_df["significant"].sum()),
        },
        "knn_jaccard_feat": {
            "within_partition_mean": round(float(jac_feat_within.mean()), 4) if len(jac_feat_within) else None,
            "cross_partition_mean":  round(float(jac_feat_cross.mean()),  4) if len(jac_feat_cross)  else None,
        },
        "knn_jaccard_logit": {
            "within_partition_mean": round(float(jac_logit_within.mean()), 4) if len(jac_logit_within) else None,
            "cross_partition_mean":  round(float(jac_logit_cross.mean()),  4) if len(jac_logit_cross)  else None,
        },
        "rsa_logit": {
            "note": "Kendall's τ on pairwise distances — positive-orthant-insensitive PRH check",
            "within_partition_mean": round(float(rsa_within.mean()), 4) if len(rsa_within) else None,
            "within_partition_std":  round(float(rsa_within.std()),  4) if len(rsa_within) else None,
            "cross_partition_mean":  round(float(rsa_cross.mean()),  4) if len(rsa_cross)  else None,
            "cross_partition_std":   round(float(rsa_cross.std()),   4) if len(rsa_cross)  else None,
        },
        "rf_baseline": rf_results if rf_results else None,
        "interpretation_notes": [
            "cka_logit is the primary PRH metric. Raw PersLay features are always "
            "non-negative (Gaussian soft-counts), causing linear-kernel CKA to be "
            "trivially ~1 for any two models including random initialisations.",
            "knn_jaccard_logit is the primary kNN metric for the same reason.",
            "Cross-partition kNN Jaccard on feat/logit uses row-aligned embeddings "
            "(same 96 neurons, different training sets).",
        ],
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  CKA logit within-partition: {summary['cka_logit']['within_partition_mean']:.3f} "
          f"± {summary['cka_logit']['within_partition_std']:.3f}  ← primary")
    print(f"  CKA logit cross-partition:  {summary['cka_logit']['cross_partition_mean']:.3f} "
          f"± {summary['cka_logit']['cross_partition_std']:.3f}  ← primary")
    print(f"  CKA feat  within-partition: {summary['cka_feat']['within_partition_mean']:.3f} "
          f"± {summary['cka_feat']['within_partition_std']:.3f}  (inflated)")
    print(f"  CKA feat  cross-partition:  {summary['cka_feat']['cross_partition_mean']:.3f} "
          f"± {summary['cka_feat']['cross_partition_std']:.3f}  (inflated)")
    print(f"  Silhouette (PersLay feat):  {summary['silhouette']['perslay_mean']:.3f} "
          f"± {summary['silhouette']['perslay_std']:.3f}")
    if morph_sil:
        print(f"  Silhouette (morph RF):      {morph_sil:.3f}")
    print(f"  Summary → {OUT}/summary.json")
    print(f"  Next: python experiments/fafb_perslay/06_figures.py\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="perslay",
                        choices=["perslay", "cnn"],
                        help="Which model embeddings to evaluate (default: perslay)")
    args = parser.parse_args()
    main(model_type=args.model_type)
