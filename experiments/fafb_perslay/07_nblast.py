"""
07_nblast.py
============
Computes NBLAST all-by-all pairwise similarity using navis's pretrained
Drosophila scoring matrix (Costa et al. 2016) — zero-shot transfer to
FlyWire FAFB with no fine-tuning on FlyWire labels.

NBLAST considers both 3D position and local geometry (tangent vectors),
making it architecturally distinct from PersLay's filtration-based topology.
CKA(PersLay, NBLAST) is therefore a genuine cross-architecture PRH test.

Outputs:
  outputs/fafb/embeddings/nblast_scores.npy      — (N, N) raw score matrix
  outputs/fafb/embeddings/nblast_pca32_emb.npy   — (N, 32) PCA embedding
  outputs/fafb/embeddings/nblast_kpca32_emb.npy  — (N, 32) kernel PCA embedding
  outputs/fafb/metrics/nblast_results.json        — accuracy, silhouette, CKA

Run:
  python experiments/fafb_perslay/07_nblast.py
"""

import sys, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset, debiased_cka, permutation_cka_null, knn_jaccard

DATA_CSV   = Path("outputs/fafb/data/dataset.csv")
PARTS_JSON = Path("outputs/fafb/data/partitions.json")
EMB_DIR    = Path("outputs/fafb/embeddings")
OUT_MET    = Path("outputs/fafb/metrics")
EMB_DIR.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

NBLAST_SCORES_PATH = EMB_DIR / "nblast_scores.npy"
NBLAST_PCA_PATH    = EMB_DIR / "nblast_pca32_emb.npy"
NBLAST_KPCA_PATH   = EMB_DIR / "nblast_kpca32_emb.npy"


def compute_nblast(samples, force=False):
    """Load SWCs, rescale nm→µm, compute NBLAST all-by-all."""
    if NBLAST_SCORES_PATH.exists() and not force:
        print(f"{PASS} NBLAST scores already computed → {NBLAST_SCORES_PATH}")
        return np.load(NBLAST_SCORES_PATH)

    try:
        import navis
    except ImportError:
        print(f"{FAIL} navis not installed. Run: pip install navis")
        raise

    swc_paths = [str(s.path) for s in samples]
    print(f"Loading {len(swc_paths)} SWC files...")
    nl = navis.read_swc(swc_paths)

    # Rescale: FlyWire SWCs are in nm; NBLAST is calibrated for µm
    print("Rescaling nm → µm (÷1000)...")
    nl_um = nl / 1000

    print("Making dotprops (k=5 nearest-neighbour tangent vectors)...")
    nl_dp = navis.make_dotprops(nl_um, k=5, progress=False)

    print("Running NBLAST all-by-all (pretrained Drosophila scoring matrix)...")
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Running NBLAST all-by-all on {n_cores} core(s)...")
    scores_df = navis.nblast_allbyall(
        nl_dp,
        normalized=True,
        use_alpha=True,
        progress=True,
        n_cores=n_cores,
    )
    scores = scores_df.values.astype(np.float32)
    print(f"{PASS} NBLAST done: shape={scores.shape}, "
          f"range=[{scores.min():.3f}, {scores.max():.3f}]")

    np.save(NBLAST_SCORES_PATH, scores)
    # Save neuron IDs for alignment
    (EMB_DIR / "nblast_neuron_ids.txt").write_text(
        "\n".join(str(x) for x in scores_df.index)
    )
    return scores


def build_embeddings(scores):
    """Symmetrize scores and build PCA / kernel-PCA embeddings."""
    S_sym = (scores + scores.T) / 2   # symmetrize (NBLAST is directional)

    # PCA to 32-dim (matches PersLay embedding dimension)
    pca = PCA(n_components=32, random_state=0)
    emb_pca = pca.fit_transform(S_sym).astype(np.float32)
    print(f"{PASS} PCA 32-dim: variance explained = {pca.explained_variance_ratio_.sum():.3f}")

    # Kernel PCA using clipped score matrix as kernel
    S_kernel = np.clip(S_sym, 0, None)
    kpca = KernelPCA(n_components=32, kernel="precomputed", random_state=0)
    emb_kpca = kpca.fit_transform(S_kernel).astype(np.float32)

    np.save(NBLAST_PCA_PATH,  emb_pca)
    np.save(NBLAST_KPCA_PATH, emb_kpca)
    return S_sym, emb_pca, emb_kpca


def evaluate(S_sym, emb_pca, labels, label_names, partitions):
    """Classification, silhouette, CKA against PersLay, kNN Jaccard."""

    print(f"\n--- NBLAST class structure (within − cross score gap) ---")
    gaps = {}
    for ci, cls in enumerate(label_names):
        mask = labels == ci
        within = S_sym[np.ix_(mask, mask)].copy()
        np.fill_diagonal(within, np.nan)
        cross = S_sym[np.ix_(mask, ~mask)]
        gap = float(np.nanmean(within) - cross.mean())
        gaps[cls] = round(gap, 3)
        print(f"  {cls:<25}: within={np.nanmean(within):.3f}  "
              f"cross={cross.mean():.3f}  gap={gap:.3f}")

    print(f"\n--- Classification accuracy ---")
    accs_raw, accs_pca = [], []
    for pname, sp in partitions.items():
        tr, te = sp["train"], sp["test"]
        for emb, accs in [(S_sym, accs_raw), (emb_pca, accs_pca)]:
            clf = LogisticRegression(max_iter=500, C=0.1, random_state=0)
            clf.fit(emb[tr], labels[tr])
            accs.append(accuracy_score(labels[te], clf.predict(emb[te])))

    sil_raw = silhouette_score(normalize(S_sym), labels)
    sil_pca = silhouette_score(normalize(emb_pca), labels)
    print(f"  NBLAST-raw96: acc={np.mean(accs_raw):.3f}±{np.std(accs_raw):.3f}  sil={sil_raw:+.3f}")
    print(f"  NBLAST-PCA32: acc={np.mean(accs_pca):.3f}±{np.std(accs_pca):.3f}  sil={sil_pca:+.3f}")

    print(f"\n--- CKA: NBLAST-PCA32 vs PersLay-PD (cross-architecture PRH test) ---")
    emb_files = sorted((EMB_DIR).glob("emb_part_*.npy"))
    ckas, jacs = [], []
    for ef in emb_files:
        pl_emb = np.load(ef)
        ckas.append(debiased_cka(pl_emb, emb_pca))
        jacs.append(knn_jaccard(pl_emb, emb_pca, k=5))

    null = permutation_cka_null(
        np.load(emb_files[0]), emb_pca, n_permutations=500, seed=0
    )
    obs   = float(np.mean(ckas))
    p_val = float(np.mean(null >= obs))
    p95   = float(np.percentile(null, 95))

    print(f"  Mean CKA:     {obs:.3f} ± {np.std(ckas):.3f}")
    print(f"  Null p95:     {p95:.3f}   p={p_val:.4f}")
    print(f"  kNN Jaccard:  {np.mean(jacs):.3f} ± {np.std(jacs):.3f}")

    if obs > p95:
        print(f"  {PASS} Observed CKA exceeds permutation null")
    else:
        print(f"  {WARN} CKA does NOT exceed permutation null")

    results = dict(
        nblast_raw96_acc=round(float(np.mean(accs_raw)), 3),
        nblast_raw96_acc_std=round(float(np.std(accs_raw)), 3),
        nblast_raw96_sil=round(sil_raw, 3),
        nblast_pca32_acc=round(float(np.mean(accs_pca)), 3),
        nblast_pca32_acc_std=round(float(np.std(accs_pca)), 3),
        nblast_pca32_sil=round(sil_pca, 3),
        cka_pl_vs_nblast=round(obs, 3),
        cka_pl_vs_nblast_std=round(float(np.std(ckas)), 3),
        cka_pvalue=round(p_val, 4),
        cka_null_p95=round(p95, 3),
        knn_jaccard_mean=round(float(np.mean(jacs)), 3),
        knn_jaccard_std=round(float(np.std(jacs)), 3),
        nblast_class_gaps=gaps,
    )
    out_path = OUT_MET / "nblast_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{PASS} Saved → {out_path}")
    return results


def main(force=False):
    print(f"\n{'='*60}")
    print("  07 — NBLAST pretrained model (navis, Costa et al. 2016)")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    labels = np.array([s.label_idx for s in samples])

    with open(PARTS_JSON) as f:
        partitions = json.load(f)

    scores           = compute_nblast(samples, force=force)
    S_sym, emb_pca, emb_kpca = build_embeddings(scores)
    results          = evaluate(S_sym, emb_pca, labels, label_names, partitions)

    print(f"\n{'='*60}")
    print(f"  NBLAST-PCA32: acc={results['nblast_pca32_acc']:.3f}  "
          f"sil={results['nblast_pca32_sil']:+.3f}  "
          f"CKA vs PersLay={results['cka_pl_vs_nblast']:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Recompute NBLAST even if scores already exist")
    args = p.parse_args()
    main(force=args.force)
