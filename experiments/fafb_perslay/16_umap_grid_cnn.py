"""
16_umap_grid_cnn.py
===================
CNNet counterpart to fig1_umap_grid.png — a grid of UMAP projections of CNNet
feature embeddings across partitions × seeds, coloured by cell type. Shows that
CNNet's geometric class separation is consistent across independently trained
models (not a cherry-picked single run).

NO retraining — reads existing cnn_emb_part_*_s*.npy from outputs/fafb/embeddings/.

Output:
  outputs/fafb/figures/fig1_umap_grid_cnn.png

Run:
  sbatch umap_compare.slurm   # after pointing it at this script, or:
  python experiments/fafb_perslay/16_umap_grid_cnn.py
"""

import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fig06", str(Path(__file__).parent / "06_figures.py"))
fig06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig06)

DARK_BG, CARD_BG, CARD2_BG = fig06.DARK_BG, fig06.CARD_BG, fig06.CARD2_BG
MUTED, PALE, WHITE = fig06.MUTED, fig06.PALE, fig06.WHITE
project_2d, _get_colors = fig06.project_2d, fig06._get_colors

EMB_DIR = Path("outputs/fafb/embeddings")
OUT     = Path("outputs/fafb/figures")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS_SHOW = [0, 1, 2]   # 3 seeds per partition (matches PersLay grid)


def load_cnn_models():
    """Load CNNet feature embeddings, sorted by partition then seed."""
    files = sorted(EMB_DIR.glob("cnn_emb_part_*.npy"))
    result = []
    for f in files:
        parts = f.stem.split("_")          # cnn_emb_part_{P}_s{S}
        part_id = int(parts[3])
        seed    = int(parts[4][1:])
        result.append({"file": f, "part_id": part_id, "seed": seed})
    return result


def main():
    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)
    n_classes = len(label_names)
    colors = _get_colors(n_classes)

    models = load_cnn_models()
    parts = sorted(set(m["part_id"] for m in models))
    seeds = [s for s in SEEDS_SHOW if any(m["seed"] == s for m in models)]
    n_rows, n_cols = len(parts), len(seeds)
    proj_method = "UMAP" if fig06.HAS_UMAP else "PCA"

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2),
                             facecolor=DARK_BG)
    axes = np.atleast_2d(axes)

    for ri, part_id in enumerate(parts):
        for ci, seed in enumerate(seeds):
            ax = axes[ri, ci]
            ax.set_facecolor(CARD_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(CARD2_BG)
            match = [m for m in models if m["part_id"] == part_id and m["seed"] == seed]
            if not match:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            emb = np.load(match[0]["file"])
            print(f"  P{part_id}S{seed}: {emb.shape} → {proj_method}...", flush=True)
            proj = project_2d(emb, seed=seed)
            for cls in range(n_classes):
                m = labels == cls
                ax.scatter(proj[m, 0], proj[m, 1], c=colors[cls],
                           s=8, alpha=0.6, edgecolors="none")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"Seed {seed}", color=PALE, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"Partition {part_id}", color=PALE, fontsize=10)

    handles = [Patch(color=colors[i], label=label_names[i]) for i in range(n_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, n_classes),
               framealpha=0.15, labelcolor=WHITE, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"{proj_method} Embeddings — CNNet across Partitions & Seeds "
        f"(n={len(labels):,}, {n_classes} classes)\n"
        "Consistent cluster structure across independently trained models",
        color=WHITE, fontsize=12, y=1.01)
    plt.tight_layout()
    path = OUT / "fig1_umap_grid_cnn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {path}")


if __name__ == "__main__":
    main()
