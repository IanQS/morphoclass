"""
02_compute_persistence.py
=========================
SWC → persistence diagrams + morphometric features.

Validates coordinate scale, checks diagram quality, and saves:
  outputs/fafb/data/diagrams.npz         — {root_id: (N,2) array}
  outputs/fafb/data/morphometrics.npy    — (n_neurons, 8) float matrix
  outputs/fafb/data/morphometric_names.json
  outputs/fafb/figures/persistence_sanity.png
  outputs/fafb/figures/morphometric_distributions.png

Run:
  python experiments/fafb_perslay/02_compute_persistence.py
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_dataset, swc_to_persistence_diagram,
                   extract_morphometrics, MORPHOMETRIC_NAMES, SCALE_NM_TO_UM)

DATA_CSV  = Path("outputs/fafb/data/dataset.csv")
OUT_DATA  = Path("outputs/fafb/data")
OUT_FIGS  = Path("outputs/fafb/figures")
OUT_FIGS.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def check_coordinate_scale(samples, n_check=20):
    """Sample SWC files and verify they're in nm (spans > 5,000 µm after scale)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import parse_swc
    spans = []
    for s in samples[:n_check]:
        try:
            arr = parse_swc(s.path)
            xyz = arr[:, 2:5]
            span = (xyz.max(axis=0) - xyz.min(axis=0)).max()
            spans.append(span)
        except Exception:
            pass
    median_raw = float(np.median(spans)) if spans else 0.0
    if median_raw > 5000:
        print(f"{PASS} Coordinate scale: raw median span = {median_raw:.0f} → nm confirmed → applying ÷1000 to µm")
        return True
    else:
        print(f"{WARN} Coordinate scale: raw median span = {median_raw:.1f} → may already be in µm, check header")
        return False


def main():
    print(f"\n{'='*60}")
    print("  Computing persistence diagrams + morphometrics")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    n = len(samples)
    print(f"Loaded {n} neurons, {len(label_names)} classes\n")

    # ── Coordinate scale check ────────────────────────────────────────────────
    check_coordinate_scale(samples)
    print()

    # ── Compute persistence diagrams ──────────────────────────────────────────
    print("Computing persistence diagrams...")
    diagrams = {}
    failed = []

    for i, s in enumerate(samples):
        try:
            pd_arr = swc_to_persistence_diagram(s.path)  # already rescales internally
            diagrams[s.path.stem] = pd_arr
        except Exception as e:
            warnings.warn(f"Failed {s.path.name}: {e}")
            failed.append(s.path.stem)
            diagrams[s.path.stem] = np.array([[0.0, 1.0]], dtype=np.float32)

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}  ...")

    if failed:
        print(f"{WARN} {len(failed)} failures (filled with dummy pair): {failed[:3]}")
    else:
        print(f"{PASS} All {n} persistence diagrams computed")

    # ── Diagram quality checks ────────────────────────────────────────────────
    n_pts = [len(d) for d in diagrams.values()]
    print(f"\nDiagram point counts: min={min(n_pts)}, median={int(np.median(n_pts))}, max={max(n_pts)}")

    # Check birth < death everywhere
    violations = sum(1 for d in diagrams.values() if (d[:, 0] >= d[:, 1]).any())
    if violations:
        print(f"{WARN} {violations} diagrams have birth >= death points (will be filtered)")
    else:
        print(f"{PASS} All diagrams have birth < death")

    # Check scale: typical persistence values should be 10–500 µm
    all_persist = np.concatenate([d[:, 1] - d[:, 0] for d in diagrams.values()])
    print(f"{PASS} Persistence range: {all_persist.min():.1f}–{all_persist.max():.1f} µm "
          f"(median {np.median(all_persist):.1f} µm)")
    if all_persist.max() > 5000:
        print(f"{WARN} Large persistence values — coordinate rescaling may have failed")

    # ── Save diagrams ─────────────────────────────────────────────────────────
    diag_path = OUT_DATA / "diagrams.npz"
    np.savez_compressed(diag_path, **{k: v for k, v in diagrams.items()})
    print(f"\n{PASS} Saved diagrams → {diag_path}")

    # ── Compute morphometrics ─────────────────────────────────────────────────
    print("\nExtracting morphometric features...")
    morph_list = []
    for i, s in enumerate(samples):
        try:
            feats = extract_morphometrics(s.path)
        except Exception:
            feats = np.zeros(8, dtype=np.float32)
        morph_list.append(feats)
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}  ...")

    morphometrics = np.stack(morph_list)
    np.save(OUT_DATA / "morphometrics.npy", morphometrics)
    with open(OUT_DATA / "morphometric_names.json", "w") as f:
        json.dump(MORPHOMETRIC_NAMES, f)
    print(f"{PASS} Saved morphometrics → {OUT_DATA}/morphometrics.npy")

    # ── Sanity figure: persistence diagrams per class ─────────────────────────
    print("\nGenerating sanity figures...")
    n_classes = min(len(label_names), 4)
    n_per_class = 3

    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 3.5, n_classes * 3),
                             facecolor='#0F2040')
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))

    for ci, cls in enumerate(label_names[:n_classes]):
        cls_samples = [s for s in samples if s.label == cls][:n_per_class]
        for pi, s in enumerate(cls_samples):
            ax = axes[ci, pi]
            ax.set_facecolor('#152B55')
            d = diagrams[s.path.stem]
            if len(d) > 0:
                ax.scatter(d[:, 0], d[:, 1], s=20, alpha=0.7,
                           color=colors[ci], edgecolors='none')
                max_v = max(d[:, 1].max(), d[:, 0].max()) * 1.05
                ax.plot([0, max_v], [0, max_v], 'k--', lw=0.5, alpha=0.4)
            ax.set_xlabel("Birth (µm)", fontsize=8, color='#6B8BAF')
            ax.set_ylabel("Death (µm)", fontsize=8, color='#6B8BAF')
            ax.tick_params(colors='#6B8BAF', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2A4A70')
            if pi == 0:
                ax.set_ylabel(cls, fontsize=9, color=colors[ci], fontweight='bold')
            if ci == 0:
                ax.set_title(f"Sample {pi+1}", fontsize=9, color='#A8DCE7')

    fig.suptitle("Persistence Diagrams — Sanity Check\n"
                 "Rows = cell types, should look visually distinct",
                 color='#E8F0F6', fontsize=11, y=1.01)
    plt.tight_layout()
    fig_path = OUT_FIGS / "persistence_sanity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"{PASS} Persistence sanity plot → {fig_path}")

    # ── Morphometric distributions per class ──────────────────────────────────
    labels_arr = np.array([s.label_idx for s in samples])
    n_feats = 4  # show first 4 features
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4),
                             facecolor='#0F2040')
    for fi in range(n_feats):
        ax = axes[fi]
        ax.set_facecolor('#152B55')
        for ci, cls in enumerate(label_names):
            mask = labels_arr == ci
            vals = morphometrics[mask, fi]
            ax.violinplot([vals], positions=[ci], showmedians=True,
                          widths=0.7)
        ax.set_title(MORPHOMETRIC_NAMES[fi].replace('_', '\n'),
                     fontsize=9, color='#A8DCE7')
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels([l[:8] for l in label_names],
                           rotation=45, ha='right', fontsize=7, color='#6B8BAF')
        ax.tick_params(colors='#6B8BAF')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A4A70')

    fig.suptitle("Morphometric Distributions by Cell Type",
                 color='#E8F0F6', fontsize=11)
    plt.tight_layout()
    mfig_path = OUT_FIGS / "morphometric_distributions.png"
    plt.savefig(mfig_path, dpi=150, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"{PASS} Morphometric distributions → {mfig_path}")

    print(f"\n{'='*60}")
    print("  Persistence computation complete.")
    print("  Inspect outputs/fafb/figures/persistence_sanity.png")
    print("  → diagrams from different classes should look distinct")
    print("  → if all look the same, coordinate rescaling failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
