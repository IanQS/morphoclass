"""
15_umap_compare.py
==================
Side-by-side UMAP of PersLay (CorianderNet) vs CNNet feature embeddings,
coloured by cell type. Visualises the geometric-separation contrast behind the
silhouette numbers (PersLay -0.18 vs CNNet +0.03): PersLay features live in the
positive orthant and intermingle; CNNet features separate classes more cleanly.

NO retraining — reads existing feature embeddings from outputs/fafb/embeddings/.
A representative model (partition 0, seed 0) is shown for each architecture.

Input:
  outputs/fafb/embeddings/emb_part_0_s0.npy       — PersLay feat (N, 32)
  outputs/fafb/embeddings/cnn_emb_part_0_s0.npy   — CNNet   feat (N, 192)
  outputs/fafb/embeddings/labels.npy, label_names.json
  outputs/fafb/metrics/progress_report.json, progress_report_cnn.json  (silhouette)

Output:
  outputs/fafb/figures/fig1_umap_compare.png

Run (UMAP on 17k points is light but use sbatch to stay off the login node):
  sbatch --export=ALL,SCRIPT=experiments/fafb_perslay/15_umap_compare.py figures.slurm
  # or, if a quick interactive node is available:
  python experiments/fafb_perslay/15_umap_compare.py
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
# Reuse the exact palette / style / projection helpers from 06_figures.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fig06", str(Path(__file__).parent / "06_figures.py"))
fig06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig06)

DARK_BG, CARD_BG, CARD2_BG = fig06.DARK_BG, fig06.CARD_BG, fig06.CARD2_BG
MUTED, PALE, WHITE = fig06.MUTED, fig06.PALE, fig06.WHITE
project_2d, _get_colors = fig06.project_2d, fig06._get_colors

EMB_DIR = Path("outputs/fafb/embeddings")
MET_DIR = Path("outputs/fafb/metrics")
OUT     = Path("outputs/fafb/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Representative model shown for each architecture
PART, SEED = 0, 0


def _silhouette(report_json):
    try:
        return json.loads((MET_DIR / report_json).read_text())["silhouette"]["mean"]
    except Exception:
        return None


def main():
    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)
    n_classes = len(label_names)
    colors = _get_colors(n_classes)

    panels = [
        ("PersLay (CorianderNet)", f"emb_part_{PART}_s{SEED}.npy",
         _silhouette("progress_report.json")),
        ("CNNet (persistence image)", f"cnn_emb_part_{PART}_s{SEED}.npy",
         _silhouette("progress_report_cnn.json")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=DARK_BG)
    proj_method = "UMAP" if fig06.HAS_UMAP else "PCA"

    for ax, (name, fname, sil) in zip(axes, panels):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(CARD2_BG)
        emb = np.load(EMB_DIR / fname)
        print(f"  {name}: {fname} {emb.shape} → projecting with {proj_method}...",
              flush=True)
        proj = project_2d(emb, seed=SEED)
        for ci in range(n_classes):
            m = labels == ci
            ax.scatter(proj[m, 0], proj[m, 1], c=colors[ci], s=8, alpha=0.6,
                       edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        sil_txt = f"silhouette = {sil:+.3f}" if sil is not None else ""
        ax.set_title(f"{name}\n{sil_txt}", color=WHITE, fontsize=12)

    handles = [Patch(color=colors[i], label=label_names[i]) for i in range(n_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, n_classes),
               framealpha=0.15, labelcolor=WHITE, fontsize=8,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        f"{proj_method} of feature embeddings — PersLay vs CNNet "
        f"(partition {PART}, seed {SEED}; n={len(labels):,}, {n_classes} classes)\n"
        "CNNet separates cell types more cleanly; PersLay features intermingle "
        "(positive-orthant constraint)",
        color=WHITE, fontsize=12, y=1.04)
    plt.tight_layout()
    path = OUT / "fig1_umap_compare.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {path}")


if __name__ == "__main__":
    main()
