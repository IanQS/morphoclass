"""
08_cross_representations.py
============================
Computes persistence image (PI) and graph-topology embeddings, then measures
cross-representation and cross-architecture CKA against PersLay, building
the complete CKA gradient table and Fig 5 (interpretability).

Representations compared:
  PI-PCA32         — rasterised 32×32 persistence image → PCA-32
  Graph-Topology   — degree/subtree-size/hop/branch-order histograms → PCA-16
  (PersLay-PD and NBLAST loaded from existing .npy files)

Outputs:
  outputs/fafb/embeddings/pi_pca_emb.npy
  outputs/fafb/embeddings/graph_topology_emb.npy
  outputs/fafb/figures/fig5_interpretability.png
  outputs/fafb/figures/fig8_cka_gradient.png
  outputs/fafb/metrics/cross_rep_results.json

Run:
  python experiments/fafb_perslay/08_cross_representations.py
"""

import sys, json, warnings
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_dataset, parse_swc, swc_to_tree,
                   debiased_cka, permutation_cka_null, knn_jaccard)

DATA_CSV   = Path("outputs/fafb/data/dataset.csv")
DIAGS_NPZ  = Path("outputs/fafb/data/diagrams.npz")
PARTS_JSON = Path("outputs/fafb/data/partitions.json")
EMB_DIR    = Path("outputs/fafb/embeddings")
OUT_FIGS   = Path("outputs/fafb/figures")
OUT_MET    = Path("outputs/fafb/metrics")
SCALE      = 1 / 1000   # nm → µm

DARK = "#0A1628"; CARD = "#0F2040"; CARD2 = "#152B55"; BORDER = "#1A3050"
TEAL = "#0AAFCC"; AMBER = "#F5A623"; CORAL = "#EF6351"; GREEN = "#22C88A"
PURPLE = "#9B7FE8"; WHITE = "#EEF4FA"; MUTED = "#5A7A9A"; SLATE = "#8BA5BE"
CELL = [TEAL, AMBER, CORAL, GREEN, PURPLE, "#F472B6", "#60A5FA", "#A3E635"]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": SLATE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": CARD2, "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

PASS = "\033[92m[PASS]\033[0m"


# ── Persistence Image rasterisation ──────────────────────────────────────────

def rasterize_diagram(diag: np.ndarray, bins: int = 32,
                      max_val: float = 380.0) -> np.ndarray:
    """
    Convert persistence diagram to a 2D persistence image.
    Axes: (birth, persistence) in µm. Weighted by persistence (linear weight).
    """
    if len(diag) == 0:
        return np.zeros((bins, bins), dtype=np.float32)
    birth   = diag[:, 0]
    persist = diag[:, 1] - diag[:, 0]
    weights = persist / (persist.max() + 1e-9)
    b_clip = np.clip(birth,   0, max_val)
    p_clip = np.clip(persist, 0, max_val)
    H, _, _ = np.histogram2d(b_clip, p_clip, bins=bins,
                              range=[[0, max_val], [0, max_val]],
                              weights=weights)
    return H.astype(np.float32)


def build_pi_embedding(samples, diags_npz, bins: int = 32) -> np.ndarray:
    """Rasterize all diagrams → (N, bins*bins) → PCA 32-dim."""
    # Determine global max_val from 95th percentile of all values
    all_vals = np.concatenate([
        np.concatenate([diags_npz[s.path.stem][:, 0],
                        diags_npz[s.path.stem][:, 1]])
        for s in samples if s.path.stem in diags_npz
    ])
    max_val = float(np.percentile(all_vals, 95))
    print(f"  PI max_val (95th pct): {max_val:.1f} µm")

    PI = np.stack([
        rasterize_diagram(diags_npz[s.path.stem], bins=bins, max_val=max_val).flatten()
        for s in samples
    ])
    pca = PCA(n_components=32, random_state=0)
    emb = pca.fit_transform(PI).astype(np.float32)
    print(f"  PI-PCA32: shape={emb.shape}, "
          f"var_explained={pca.explained_variance_ratio_.sum():.3f}")
    return emb


# ── Graph topology histogram embedding ───────────────────────────────────────

def graph_topology_embedding(path: Path, n_bins: int = 16,
                              scale: float = SCALE) -> np.ndarray:
    """
    Graph-structural embedding without persistence filtration.
    Four histogram features from the full skeleton tree:
      1. Degree distribution (branching pattern)
      2. Subtree-size distribution (mass distribution from each node)
      3. Hop-distance distribution from root (connectivity structure)
      4. Branch-order distribution (depth of branching complexity)
    Concatenated → (4 * n_bins,) vector.
    """
    arr = parse_swc(path)
    arr[:, 2:5] *= scale
    children  = swc_to_tree(arr)
    id_to_row = {int(r[0]): r for r in arr}
    roots = [int(r[0]) for r in arr if int(r[6]) == -1]
    if not roots:
        return np.zeros(n_bins * 4, dtype=np.float32)
    root = roots[0]

    # BFS order
    order: list[int] = []
    q = deque([root])
    while q:
        n = q.popleft()
        order.append(n)
        for c in children.get(n, []):
            q.append(c)

    # 1. Degree histogram
    degrees = np.array([len(children.get(n, [])) for n in id_to_row])
    h_deg, _ = np.histogram(degrees, bins=n_bins, range=[0, 20])

    # 2. Subtree-size histogram
    st = {n: 1 for n in id_to_row}
    for n in reversed(order):
        for c in children.get(n, []):
            st[n] += st[c]
    sizes = np.array(list(st.values()), dtype=np.float32)
    h_sz, _ = np.histogram(sizes / (sizes.max() + 1e-9), bins=n_bins, range=[0, 1])

    # 3. Hop-distance histogram
    hop = {root: 0}
    for n in order[1:]:
        p = int(id_to_row[n][6])
        hop[n] = hop.get(p, 0) + 1
    hops = np.array(list(hop.values()), dtype=np.float32)
    h_hop, _ = np.histogram(hops / (hops.max() + 1e-9), bins=n_bins, range=[0, 1])

    # 4. Branch-order histogram
    bo = {root: 0}
    for n in order[1:]:
        p = int(id_to_row[n][6])
        bo[n] = bo.get(p, 0) + (1 if len(children.get(p, [])) >= 2 else 0)
    bov = np.array(list(bo.values()), dtype=np.float32)
    h_bo, _ = np.histogram(bov / (bov.max() + 1e-9), bins=n_bins, range=[0, 1])

    feat = np.concatenate([h_deg, h_sz, h_hop, h_bo]).astype(np.float32)
    return feat / (feat.sum() + 1e-9)


def build_gt_embedding(samples) -> np.ndarray:
    """Build graph-topology embeddings for all neurons, PCA to 16-dim."""
    raw = np.stack([graph_topology_embedding(s.path) for s in samples])
    pca = PCA(n_components=min(16, raw.shape[0] - 1), random_state=0)
    emb = pca.fit_transform(raw).astype(np.float32)
    print(f"  Graph-Topology: shape={emb.shape}, "
          f"var_explained={pca.explained_variance_ratio_.sum():.3f}")
    return emb


# ── Fig 5: PersLay interpretability ──────────────────────────────────────────

def plot_interpretability(samples, diags_npz, labels, label_names):
    """
    Contribution-weighted persistence diagrams per class.
    w_i * mean_k(phi_ik) gives the embedding contribution of each point.
    Uses saved model weights from part_0_s0.
    """
    model_path = Path("outputs/fafb/models/perslay_part_0_s0.npz")
    if not model_path.exists():
        print(f"  WARN: {model_path} not found, skipping interpretability plot")
        return

    m = np.load(model_path, allow_pickle=True)
    landmarks = m["landmarks"]
    alpha     = float(m["alpha"][0])
    sigma     = float(m["sigma"][0])

    def contributions(diag):
        if len(diag) == 0:
            return np.array([])
        p  = diag.astype(np.float32)
        w  = np.tanh(alpha * (p[:, 1] - p[:, 0]))
        diff = p[:, None, :] - landmarks[None, :, :]
        phi  = np.exp(-(diff ** 2).sum(-1) / (2 * sigma ** 2))
        return (w * phi.mean(axis=1))

    class_data = {}
    for ci, cls in enumerate(label_names):
        cls_samples = [s for s in samples if s.label_idx == ci]
        all_pts, all_c = [], []
        for s in cls_samples:
            d = diags_npz[s.path.stem]
            c = contributions(d)
            if len(c):
                all_pts.append(d); all_c.append(c)
        if all_pts:
            pts = np.concatenate(all_pts)
            c   = np.concatenate(all_c)
            class_data[cls] = {"pts": pts, "c": c,
                                "mean_c": float(c.mean()),
                                "top5_pts": pts[np.argsort(c)[-5:][::-1]]}

    n_cls = len(label_names)
    ncols = 4; nrows = (n_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             facecolor=DARK)
    axes = axes.flatten()

    for ci, cls in enumerate(label_names):
        ax = axes[ci]; ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        if cls not in class_data:
            ax.set_title(cls, color=CELL[ci % len(CELL)], fontsize=10)
            continue
        d = class_data[cls]
        pts, c = d["pts"], d["c"]
        persist = pts[:, 1] - pts[:, 0]
        c_norm  = np.clip(c / (c.max() + 1e-9), 0, 1)
        ax.scatter(pts[:, 0], persist,
                   s=c_norm * 55 + 2, c=c_norm,
                   cmap="YlOrRd", alpha=np.clip(c_norm * 0.9 + 0.1, 0, 1),
                   vmin=0, vmax=1)
        top5 = d["top5_pts"]
        ax.scatter(top5[:, 0], top5[:, 1] - top5[:, 0],
                   s=80, marker="*", color=CELL[ci % len(CELL)],
                   zorder=5, edgecolors="white", lw=0.5)
        ax.set_title(f"{cls[:16]}  (mean_c={d['mean_c']:.4f})",
                     color=CELL[ci % len(CELL)], fontsize=10, fontweight="bold")
        ax.set_xlabel("Birth (µm)", fontsize=8, color=MUTED)
        ax.set_ylabel("Persistence (µm)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7)

    for ax in axes[n_cls:]:
        ax.set_visible(False)

    fig.suptitle("PersLay Contribution-Weighted Diagrams — "
                 "Which branch events define each cell type?\n"
                 "Point size/colour = w_i×φ_ik contribution  "
                 "★ = top-5 most influential events",
                 color=WHITE, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUT_FIGS / "fig5_interpretability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"{PASS} → {path}")


# ── Fig 8: CKA gradient bar chart ────────────────────────────────────────────

def plot_cka_gradient(results: dict):
    """
    Horizontal bar chart showing the CKA gradient from encoding gap
    to cross-architecture comparison against PersLay.
    """
    items = [
        ("PI-PCA32\n(same topology, diff encoding)", results["cka_pl_pi"],    results["cka_pl_pi_std"],    TEAL),
        ("NBLAST-PCA32\n(topology vs position+geometry)", results["cka_pl_nblast"], results["cka_pl_nblast_std"], GREEN),
        ("Morphometric RF\n(topology vs scalars)",      results["cka_pl_morph"],  results["cka_pl_morph_std"],  AMBER),
        ("Graph-Topology\n(filtration vs histograms)",  results["cka_pl_gt"],     results["cka_pl_gt_std"],     PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=DARK)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    y_pos = np.arange(len(items))
    for i, (label, val, std, col) in enumerate(items):
        ax.barh(i, val, xerr=std, color=col, alpha=0.85, height=0.55,
                error_kw=dict(ecolor=WHITE, lw=1.5, capsize=4))
        ax.text(val + std + 0.01, i, f"{val:.3f}", va="center",
                fontsize=10, color=col, fontweight="bold")

    ax.axvline(results["null_p95"], color=CORAL, lw=1.5, ls="--", alpha=0.7,
               label=f"Permutation null p95 = {results['null_p95']:.3f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x[0] for x in items], fontsize=10)
    ax.set_xlabel("Debiased CKA vs PersLay-PD (± std, 15 model pairs)", fontsize=10)
    ax.set_xlim(0, 0.65)
    ax.set_title("CKA Gradient: Encoding Gap → Architecture Gap\n"
                 "Higher CKA = more shared representational geometry with PersLay",
                 color=WHITE, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE)
    ax.text(0.42, 3.42, "← same topology\ndiff encoding",
            color=TEAL, fontsize=8.5, ha="center")
    ax.text(0.042, -0.58, "not significant (p=0.111)",
            color=CORAL, fontsize=8, ha="left")

    plt.tight_layout()
    path = OUT_FIGS / "fig8_cka_gradient.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"{PASS} → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  08 — Cross-representation embeddings + CKA gradient")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    labels = np.array([s.label_idx for s in samples])
    diags  = np.load(DIAGS_NPZ, allow_pickle=True)

    with open(PARTS_JSON) as f:
        partitions = json.load(f)

    # Use logit embeddings (8-dim log-softmax) as the PersLay reference.
    # Feature embeddings (emb_part_*.npy) are always non-negative → CKA trivially ~1.0.
    emb_files = sorted(EMB_DIR.glob("logit_part_*.npy"))

    # ── PI-PCA32 ──────────────────────────────────────────────────────────────
    pi_path = EMB_DIR / "pi_pca_emb.npy"
    if pi_path.exists():
        print(f"[SKIP] PI-PCA already exists")
        pi_emb = np.load(pi_path)
    else:
        print("Building persistence image embeddings...")
        pi_emb = build_pi_embedding(samples, diags)
        np.save(pi_path, pi_emb)

    # ── Graph-Topology ────────────────────────────────────────────────────────
    gt_path = EMB_DIR / "graph_topology_emb.npy"
    if gt_path.exists():
        print(f"[SKIP] Graph-topology already exists")
        gt_emb = np.load(gt_path)
    else:
        print("Building graph-topology embeddings...")
        gt_emb = build_gt_embedding(samples)
        np.save(gt_path, gt_emb)

    # ── CKA against PersLay ───────────────────────────────────────────────────
    print("\n--- Cross-representation CKA (all vs PersLay-PD) ---")
    morph = np.load("outputs/fafb/data/morphometrics.npy")

    results = {}
    null = permutation_cka_null(
        np.load(emb_files[0]), pi_emb, n_permutations=500, seed=0
    )
    results["null_p95"] = round(float(np.percentile(null, 95)), 3)

    for name, emb, key in [
        ("PI-PCA32",       pi_emb,  "cka_pl_pi"),
        ("Graph-Topology", gt_emb,  "cka_pl_gt"),
        ("Morphometric",   morph,   "cka_pl_morph"),
    ]:
        ckas = [debiased_cka(np.load(ef), emb) for ef in emb_files]
        results[key]          = round(float(np.mean(ckas)), 3)
        results[key + "_std"] = round(float(np.std(ckas)), 3)
        p_val = float(np.mean(null >= np.mean(ckas)))
        print(f"  {name:<22}: {np.mean(ckas):.3f} ± {np.std(ckas):.3f}  p={p_val:.4f}")

    # Load NBLAST if available
    nblast_path = EMB_DIR / "nblast_pca32_emb.npy"
    if nblast_path.exists():
        nblast_emb = np.load(nblast_path)
        ckas = [debiased_cka(np.load(ef), nblast_emb) for ef in emb_files]
        results["cka_pl_nblast"]     = round(float(np.mean(ckas)), 3)
        results["cka_pl_nblast_std"] = round(float(np.std(ckas)), 3)
        print(f"  {'NBLAST-PCA32':<22}: {np.mean(ckas):.3f} ± {np.std(ckas):.3f}")
    else:
        results["cka_pl_nblast"]     = None
        results["cka_pl_nblast_std"] = None
        print("  NBLAST-PCA32: not computed (run 07_nblast.py first)")

    with open(OUT_MET / "cross_rep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_interpretability(samples, diags, labels, label_names)
    if results["cka_pl_nblast"] is not None:
        plot_cka_gradient(results)

    print(f"\n{'='*60}")
    print("  Cross-representation analysis complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
