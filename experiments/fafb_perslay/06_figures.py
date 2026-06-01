"""
06_figures.py
=============
Generates all 4 publication figures from outputs/fafb/metrics/ and embeddings/.

Figures:
  fig1_umap_grid.png         — 3×3 UMAP grid (partitions × seeds)
  fig2_cka_heatmap.png       — 15×15 CKA heatmap with partition blocks
  fig3_silhouette.png        — silhouette comparison + null band
  fig4_knn_stability.png     — kNN Jaccard distribution (within vs cross)

Run:
  python experiments/fafb_perslay/06_figures.py
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not available — using PCA for 2D projection")

from sklearn.decomposition import PCA

EMB_DIR  = Path("outputs/fafb/embeddings")
MET_DIR  = Path("outputs/fafb/metrics")
OUT      = Path("outputs/fafb/figures")
OUT.mkdir(parents=True, exist_ok=True)

DARK_BG  = "#0A1628"
CARD_BG  = "#0F2040"
CARD2_BG = "#152B55"
MUTED    = "#6B8BAF"
PALE     = "#A8DCE7"
WHITE    = "#E8F0F6"

def dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "axes.edgecolor":   CARD2_BG,
        "axes.labelcolor":  MUTED,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "text.color":       WHITE,
        "grid.color":       CARD2_BG,
        "grid.alpha":       0.5,
        "font.family":      "sans-serif",
    })

dark_style()

# Colorblind-safe palette for cell types (max 8)
CELL_COLORS = [
    "#0B8FAC", "#F59E0B", "#EF6351", "#10B981",
    "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
]


def project_2d(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """UMAP or PCA fallback for 2D projection."""
    if HAS_UMAP and emb.shape[0] >= 10:
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=min(10, emb.shape[0]-1),
                            min_dist=0.1)
        return reducer.fit_transform(emb)
    else:
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(emb)


def load_emb_models():
    """Load all embedding files, sorted by partition then seed."""
    files = sorted(EMB_DIR.glob("emb_part_*.npy"))
    result = []
    for f in files:
        parts = f.stem.split("_")
        part_id = int(parts[2])
        seed    = int(parts[3][1:])
        result.append({
            "file": f, "part_id": part_id, "seed": seed,
            "label": f"P{part_id}S{seed}",
        })
    return result


# ── Figure 1: UMAP Grid ───────────────────────────────────────────────────────

def fig1_umap_grid(models, labels, label_names):
    print("  Generating Fig 1: UMAP grid...")

    parts = sorted(set(m["part_id"] for m in models))
    seeds_show = sorted(set(m["seed"] for m in models))[:3]  # show 3 seeds
    n_rows, n_cols = len(parts), len(seeds_show)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2),
                             facecolor=DARK_BG)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    proj_method = "UMAP" if HAS_UMAP else "PCA"

    for ri, part_id in enumerate(parts):
        for ci, seed in enumerate(seeds_show):
            ax = axes[ri, ci]
            ax.set_facecolor(CARD_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(CARD2_BG)

            match = [m for m in models if m["part_id"] == part_id and m["seed"] == seed]
            if not match:
                ax.text(0.5, 0.5, "no data", ha='center', va='center',
                        color=MUTED, transform=ax.transAxes)
                continue

            emb = np.load(match[0]["file"])
            proj = project_2d(emb, seed=seed)

            for ci2, cls_idx in enumerate(range(len(label_names))):
                mask = labels == cls_idx
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=CELL_COLORS[ci2 % len(CELL_COLORS)],
                           s=18, alpha=0.75, edgecolors='none',
                           label=label_names[cls_idx] if (ri == 0 and ci == 0) else "")

            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"Seed {seed}", color=PALE, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"Partition {part_id}", color=PALE, fontsize=10)

    # Shared legend
    handles = [Patch(color=CELL_COLORS[i], label=label_names[i])
               for i in range(len(label_names))]
    fig.legend(handles=handles, loc='lower center', ncol=min(4, len(label_names)),
               framealpha=0.15, labelcolor=WHITE, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"{proj_method} Embeddings — PersLay across Partitions & Seeds\n"
                 "Consistent cluster structure supports representational stability",
                 color=WHITE, fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUT / "fig1_umap_grid.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    → {path}")


# ── Figure 2: CKA Heatmap ─────────────────────────────────────────────────────

def fig2_cka_heatmap(models):
    print("  Generating Fig 2: CKA heatmap...")

    cka = np.load(MET_DIR / "cka_logit_matrix.npy")
    with open(MET_DIR / "cka_labels.json") as f:
        meta = json.load(f)

    n = len(meta["labels"])
    part_ids = np.array(meta["part_ids"])
    model_labels = meta["labels"]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(cka, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.035, label="CKA")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka[i,j]:.2f}", ha='center', va='center',
                    fontsize=7, color=WHITE if cka[i,j] < 0.7 else DARK_BG)

    # Partition boundary lines
    parts_sorted = sorted(set(part_ids))
    cumsum = 0
    for p in parts_sorted[:-1]:
        cumsum += (part_ids == p).sum()
        for xy in [ax.axhline, ax.axvline]:
            xy(cumsum - 0.5, color="#F59E0B", lw=2)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8, color=MUTED)
    ax.set_yticklabels(model_labels, fontsize=8, color=MUTED)

    # Annotate block means
    within_vals = cka[part_ids[:, None] == part_ids[None, :]]
    cross_vals  = cka[part_ids[:, None] != part_ids[None, :]]
    np.fill_diagonal(cka, np.nan)

    ax.set_title(f"Pairwise Debiased CKA — {n} Models\n"
                 f"Within-partition: {np.nanmean(within_vals[within_vals<1]):.3f}  "
                 f"Cross-partition: {cross_vals.mean():.3f}",
                 color=WHITE, fontsize=11)

    plt.tight_layout()
    path = OUT / "fig2_cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    → {path}")


# ── Figure 3: Silhouette Comparison ──────────────────────────────────────────

def fig3_silhouette():
    print("  Generating Fig 3: silhouette comparison...")

    sil_df = pd.read_csv(MET_DIR / "silhouette.csv")
    morph_path = EMB_DIR / "morphometric_emb.npy"

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    parts = sorted(sil_df["partition"].unique())
    n_parts = len(parts)
    bar_w = 0.30
    x = np.arange(n_parts)

    # PersLay bars (mean ± std per partition)
    for pi, part in enumerate(parts):
        part_sils = sil_df[sil_df["partition"] == part]["silhouette"].values
        mean_s = part_sils.mean()
        std_s  = part_sils.std()
        ax.bar(x[pi] - bar_w/2, mean_s, bar_w,
               color="#0B8FAC", alpha=0.85, label="PersLay" if pi == 0 else "")
        ax.errorbar(x[pi] - bar_w/2, mean_s, std_s,
                    fmt='none', color=WHITE, capsize=4, lw=1.5)

        # Null band
        null_p95 = sil_df[sil_df["partition"] == part]["null_p95"].values.mean()
        null_mean = sil_df[sil_df["partition"] == part]["null_mean"].values.mean()
        ax.fill_between([x[pi] - bar_w - 0.1, x[pi] + bar_w + 0.05],
                        null_mean, null_p95,
                        alpha=0.25, color="#EF6351",
                        label="Permutation null (5th–95th)" if pi == 0 else "")

    # Morphometric baseline bars
    if morph_path.exists():
        from sklearn.metrics import silhouette_score as ss
        from sklearn.preprocessing import StandardScaler
        with open(Path("outputs/fafb/data/partitions.json")) as f:
            partitions = json.load(f)
        labels = np.load(EMB_DIR / "labels.npy")
        morph_emb = np.load(morph_path)

        for pi, part_str in enumerate([f"part_{p}" for p in parts]):
            if part_str not in partitions:
                continue
            test_idx = partitions[part_str]["test"]
            morph_sil = float(ss(morph_emb[test_idx], labels[test_idx]))
            ax.bar(x[pi] + bar_w/2, morph_sil, bar_w,
                   color="#F59E0B", alpha=0.85,
                   label="Morphometric RF" if pi == 0 else "")

    ax.axhline(0, color=MUTED, lw=0.8, ls='--', alpha=0.5, label="Chance (0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Partition {p}" for p in parts], color=MUTED)
    ax.set_ylabel("Silhouette Score", color=MUTED)
    ax.set_ylim(-0.15, 1.0)
    ax.legend(framealpha=0.15, labelcolor=WHITE, fontsize=9)
    ax.set_title("Cell-Type Separability: PersLay vs Morphometric Baseline\n"
                 "Error bars = ±1 std across seeds  ·  Red band = permutation null",
                 color=WHITE, fontsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor(CARD2_BG)

    plt.tight_layout()
    path = OUT / "fig3_silhouette.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    → {path}")


# ── Figure 4: kNN Jaccard Stability ──────────────────────────────────────────

def fig4_knn_stability():
    print("  Generating Fig 4: kNN Jaccard stability...")

    jac_df = pd.read_csv(MET_DIR / "knn_jaccard.csv")
    # column is jaccard_logit (primary) or legacy jaccard
    jac_col = "jaccard_logit" if "jaccard_logit" in jac_df.columns else "jaccard"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=DARK_BG)

    within = jac_df[jac_df["type"] == "within_partition"][jac_col].values
    cross  = jac_df[jac_df["type"] == "cross_partition"][jac_col].values

    for ax, vals, col, label in [
        (ax1, within, "#0B8FAC", "Within-partition\n(same data, diff seed)"),
        (ax2, cross,  "#F59E0B", "Cross-partition\n(diff data, diff seed)"),
    ]:
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(CARD2_BG)

        if len(vals) > 0:
            ax.violinplot([vals], positions=[0], showmedians=True, widths=0.6)
            ax.scatter(np.zeros(len(vals)) + np.random.uniform(-0.08, 0.08, len(vals)),
                       vals, s=30, alpha=0.6, color=col, zorder=3)
            ax.axhline(vals.mean(), color=col, lw=2, ls='--',
                       label=f"Mean = {vals.mean():.3f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Jaccard Similarity (k=5)", color=MUTED)
        ax.set_xticks([])
        ax.set_title(label, color=col, fontsize=12)
        if len(vals):
            ax.legend(framealpha=0.15, labelcolor=WHITE, fontsize=10)

    fig.suptitle("kNN Embedding Stability — Same Neighbor Structure Across Runs?\n"
                 "High Jaccard = representations are reproducible",
                 color=WHITE, fontsize=11)
    plt.tight_layout()
    path = OUT / "fig4_knn_stability.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"    → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  Generating publication figures")
    print(f"{'='*60}\n")

    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)

    models = load_emb_models()
    if not models:
        print("No embeddings found. Run 04_extract_embeddings.py first.")
        return

    fig1_umap_grid(models, labels, label_names)
    fig2_cka_heatmap(models)
    fig3_silhouette()
    fig4_knn_stability()

    print(f"\n{'='*60}")
    print(f"  All figures → {OUT}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
