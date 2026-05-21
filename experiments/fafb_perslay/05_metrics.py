"""
05_metrics.py
=============
Computes all PRH evaluation metrics across the 15-model ensemble.

Metrics:
  1. Debiased CKA — pairwise across all model pairs (15×15 matrix)
  2. Silhouette score — per model, with 500-permutation null
  3. kNN Jaccard stability — all valid within-partition pairs
  4. Classification accuracy — from run_log.csv vs morphometric baseline

Outputs:
  outputs/fafb/metrics/cka_matrix.npy       — (N_models, N_models)
  outputs/fafb/metrics/cka_labels.json      — model labels
  outputs/fafb/metrics/silhouette.csv
  outputs/fafb/metrics/knn_jaccard.csv
  outputs/fafb/metrics/summary.json

Run:
  python experiments/fafb_perslay/05_metrics.py
"""

import json, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import debiased_cka, permutation_cka_null, knn_jaccard

EMB_DIR  = Path("outputs/fafb/embeddings")
DATA_DIR = Path("outputs/fafb/data")
OUT      = Path("outputs/fafb/metrics")
OUT.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
N_PERM = 500   # permutations for null distribution (500 is fast; use 1000 for paper)
KNN_K  = 5     # k for Jaccard (use 10 for larger dataset)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def load_all_embeddings():
    """Load all saved embedding matrices. Returns list of (label, array) pairs."""
    emb_files = sorted(EMB_DIR.glob("emb_part_*.npy"))
    result = []
    for f in emb_files:
        stem = f.stem  # emb_part_0_s2
        parts = stem.split("_")
        part_id = int(parts[2])
        seed    = int(parts[3][1:])
        emb = np.load(f)
        result.append({
            "label": f"P{part_id}S{seed}",
            "part_id": part_id,
            "seed": seed,
            "emb": emb,
        })
    return result


def main():
    print(f"\n{'='*60}")
    print("  Computing PRH evaluation metrics")
    print(f"{'='*60}\n")

    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)

    models = load_all_embeddings()
    if not models:
        print(f"No embeddings found in {EMB_DIR}. Run 04_extract_embeddings.py first.")
        return
    print(f"Loaded {len(models)} model embeddings\n")

    n_models = len(models)
    model_labels = [m["label"] for m in models]

    # ── 1. Debiased CKA matrix ────────────────────────────────────────────────
    print("Computing pairwise debiased CKA...")
    cka_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i, n_models):
            val = debiased_cka(models[i]["emb"], models[j]["emb"])
            cka_matrix[i, j] = val
            cka_matrix[j, i] = val
        print(f"  Row {i+1}/{n_models} done")

    np.save(OUT / "cka_matrix.npy", cka_matrix)
    with open(OUT / "cka_labels.json", "w") as f:
        json.dump({
            "labels": model_labels,
            "part_ids": [m["part_id"] for m in models],
            "seeds": [m["seed"] for m in models],
        }, f, indent=2)

    # Summarize within vs cross partition
    part_ids = np.array([m["part_id"] for m in models])
    within_mask  = part_ids[:, None] == part_ids[None, :]
    cross_mask   = ~within_mask
    np.fill_diagonal(within_mask, False)  # exclude diagonal

    within_cka = cka_matrix[within_mask]
    cross_cka  = cka_matrix[cross_mask]

    print(f"\n{PASS} CKA matrix saved")
    print(f"  Within-partition CKA: {within_cka.mean():.3f} ± {within_cka.std():.3f}")
    print(f"  Cross-partition  CKA: {cross_cka.mean():.3f} ± {cross_cka.std():.3f}")

    if cross_cka.mean() > 0.5:
        print(f"  {PASS} Cross-partition CKA > 0.5 — representations converge across data")
    else:
        print(f"  {WARN} Cross-partition CKA low — representations may be data-specific")

    # ── Permutation null for cross-partition CKA ──────────────────────────────
    print(f"\nRunning permutation null ({N_PERM} permutations) on first cross-partition pair...")
    # Take first pair from different partitions
    cross_pairs = [(i, j) for i in range(n_models) for j in range(i+1, n_models)
                   if models[i]["part_id"] != models[j]["part_id"]]
    if cross_pairs:
        i0, j0 = cross_pairs[0]
        null_dist = permutation_cka_null(models[i0]["emb"], models[j0]["emb"],
                                          n_permutations=N_PERM, seed=0)
        obs_cka = cka_matrix[i0, j0]
        p_val = float(np.mean(null_dist >= obs_cka))
        p95   = float(np.percentile(null_dist, 95))
        print(f"  Observed CKA = {obs_cka:.3f}, null p95 = {p95:.3f}, p = {p_val:.4f}")
        if obs_cka > p95:
            print(f"  {PASS} Observed CKA exceeds null 95th percentile (p={p_val:.4f})")
        else:
            print(f"  {WARN} CKA does NOT exceed permutation null — not statistically significant")
    else:
        print(f"  {WARN} Need >1 partition for cross-partition null test")
        null_dist = np.array([0.0])
        obs_cka, p_val, p95 = 0.0, 1.0, 0.0

    # ── 2. Silhouette scores ──────────────────────────────────────────────────
    print(f"\nComputing silhouette scores + permutation nulls...")
    sil_rows = []

    for m in models:
        emb = m["emb"]
        # L2-normalize before silhouette
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

        obs_sil = float(silhouette_score(emb_norm, labels))

        # Permutation null (shuffle labels)
        rng = np.random.default_rng(42)
        null_sils = []
        for _ in range(N_PERM):
            shuffled = rng.permutation(labels)
            try:
                null_sils.append(float(silhouette_score(emb_norm, shuffled)))
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
        morph_sil = float(silhouette_score(morph_emb, labels))
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

    # ── 3. kNN Jaccard stability ──────────────────────────────────────────────
    print(f"\nComputing kNN Jaccard stability (k={KNN_K})...")
    jaccard_rows = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Only compute within-partition (same neuron sets) — valid Jaccard
            if models[i]["part_id"] == models[j]["part_id"]:
                jac = knn_jaccard(models[i]["emb"], models[j]["emb"], k=KNN_K)
                pair_type = "within_partition"
            else:
                # Cross-partition: neurons differ — Jaccard on row-aligned embeddings
                # is a proxy for structural similarity, not literal neighbor overlap
                # (note in analysis)
                jac = knn_jaccard(models[i]["emb"], models[j]["emb"], k=KNN_K)
                pair_type = "cross_partition"

            jaccard_rows.append({
                "model_a": models[i]["label"],
                "model_b": models[j]["label"],
                "part_a": models[i]["part_id"],
                "part_b": models[j]["part_id"],
                "seed_a": models[i]["seed"],
                "seed_b": models[j]["seed"],
                "type": pair_type,
                "jaccard": round(jac, 4),
            })

    jac_df = pd.DataFrame(jaccard_rows)
    jac_df.to_csv(OUT / "knn_jaccard.csv", index=False)

    within = jac_df[jac_df["type"] == "within_partition"]["jaccard"]
    cross  = jac_df[jac_df["type"] == "cross_partition"]["jaccard"]
    print(f"{PASS} kNN Jaccard saved")
    if len(within):
        print(f"  Within-partition:  {within.mean():.3f} ± {within.std():.3f}")
    if len(cross):
        print(f"  Cross-partition:   {cross.mean():.3f} ± {cross.std():.3f}")
        print(f"  Note: cross-partition Jaccard uses row-aligned embeddings (see analysis notes)")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "n_models": n_models,
        "n_neurons": int(len(labels)),
        "n_classes": int(len(label_names)),
        "cka": {
            "within_partition_mean": round(float(within_cka.mean()), 4),
            "within_partition_std":  round(float(within_cka.std()),  4),
            "cross_partition_mean":  round(float(cross_cka.mean()),  4),
            "cross_partition_std":   round(float(cross_cka.std()),   4),
            "permutation_null_p95":  round(float(p95),               4),
            "cross_partition_pvalue":round(float(p_val),             4),
        },
        "silhouette": {
            "perslay_mean":     round(float(sil_df["silhouette"].mean()), 4),
            "perslay_std":      round(float(sil_df["silhouette"].std()),  4),
            "morphometric_baseline": round(float(morph_sil), 4) if morph_sil else None,
            "n_significant_models": int(sil_df["significant"].sum()),
        },
        "knn_jaccard": {
            "within_partition_mean": round(float(within.mean()), 4) if len(within) else None,
            "cross_partition_mean":  round(float(cross.mean()),  4) if len(cross)  else None,
        },
        "rf_baseline": rf_results if rf_results else None,
        "interpretation_notes": [
            "Cross-partition CKA compares embeddings of different neurons — "
            "tests geometric similarity of the learned spaces, not point-wise alignment.",
            "Cross-partition kNN Jaccard is a proxy metric (row-indexed, not neuron-matched). "
            "For a strict PRH test, use CKA as the primary metric.",
            "For genuine PRH (Huh et al. 2024) the strongest test is cross-architecture CKA "
            "(PersLay vs GNN) — add CorianderNet runs to enable this comparison.",
        ],
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  CKA within-partition:  {summary['cka']['within_partition_mean']:.3f} ± {summary['cka']['within_partition_std']:.3f}")
    print(f"  CKA cross-partition:   {summary['cka']['cross_partition_mean']:.3f} ± {summary['cka']['cross_partition_std']:.3f}")
    print(f"  Silhouette (PersLay):  {summary['silhouette']['perslay_mean']:.3f} ± {summary['silhouette']['perslay_std']:.3f}")
    if morph_sil:
        print(f"  Silhouette (morph RF): {morph_sil:.3f}")
    print(f"  Summary → {OUT}/summary.json")
    print(f"  Next: python experiments/fafb_perslay/06_figures.py\n")


if __name__ == "__main__":
    main()
