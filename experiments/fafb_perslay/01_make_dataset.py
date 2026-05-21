"""
01_make_dataset.py
==================
Validates the subset dataset and generates stratified partition indices.

Outputs:
  outputs/fafb/data/dataset.csv        — morphoclass-ready path\tlabel CSV
  outputs/fafb/data/partitions.json    — {part_0: {train, val, test}, ...}
  outputs/fafb/data/label_map.json     — {class_name: int_index}
  outputs/fafb/data/class_distribution.png

Run:
  python experiments/fafb_perslay/01_make_dataset.py \
      --dataset morphoclass/data/fafb_subset/dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ── add experiments dir to path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset

OUT = Path("outputs/fafb/data")
OUT.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def main(dataset_csv: str, n_splits: int = 3, n_seeds: int = 5,
         min_per_class: int = 8):

    print(f"\n{'='*60}")
    print(f"  Dataset preparation: {dataset_csv}")
    print(f"{'='*60}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    samples, label_names = load_dataset(Path(dataset_csv))
    print(f"{PASS} Loaded {len(samples)} neurons across {len(label_names)} classes")

    labels_arr = np.array([s.label_idx for s in samples])
    paths = [s.path for s in samples]

    # ── Validate SWC files exist ──────────────────────────────────────────────
    missing = [s.path for s in samples if not s.path.exists()]
    if missing:
        print(f"{FAIL} {len(missing)} SWC files missing. First: {missing[0]}")
        sys.exit(1)
    print(f"{PASS} All SWC files exist on disk")

    # ── Class distribution ────────────────────────────────────────────────────
    counts = pd.Series(labels_arr).value_counts().sort_index()
    named = {label_names[i]: int(counts.get(i, 0)) for i in range(len(label_names))}

    print(f"\nClass distribution:")
    for name, cnt in sorted(named.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 2)
        flag = f"  {WARN} < {min_per_class}" if cnt < min_per_class else ""
        print(f"  {name:35s}  {cnt:4d}  {bar}{flag}")

    small_classes = [n for n, c in named.items() if c < min_per_class]
    if small_classes:
        print(f"\n{WARN} Classes with < {min_per_class} neurons (may cause stratification issues):")
        for c in small_classes:
            print(f"       {c}: {named[c]}")

    imbalance = max(named.values()) / max(1, min(named.values()))
    if imbalance > 10:
        print(f"{WARN} Imbalance ratio {imbalance:.1f}x — consider class-weighted loss")
    else:
        print(f"{PASS} Imbalance ratio: {imbalance:.1f}x")

    # ── Save class distribution figure ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    names_sorted = sorted(named.keys(), key=lambda n: -named[n])
    counts_sorted = [named[n] for n in names_sorted]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names_sorted)))
    ax.bar(range(len(names_sorted)), counts_sorted, color=colors)
    ax.axhline(min_per_class, color='red', linestyle='--', alpha=0.5,
               label=f'min={min_per_class}')
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel("Neuron count")
    ax.set_title(f"Class distribution (N={len(samples)})")
    ax.legend()
    plt.tight_layout()
    fig_path = OUT / "class_distribution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n{PASS} Saved class distribution → {fig_path}")

    # ── Copy dataset CSV to outputs ───────────────────────────────────────────
    out_csv = OUT / "dataset.csv"
    import shutil
    shutil.copy(dataset_csv, out_csv)
    print(f"{PASS} Dataset CSV → {out_csv}")

    # ── Generate stratified partitions ────────────────────────────────────────
    print(f"\nGenerating {n_splits} stratified partitions × {n_seeds} seeds...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(samples))
    partitions = {}

    for part_id, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels_arr)):
        # Inner stratified split for train/val (80/20 of train_val)
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_labels = labels_arr[train_val_idx]
        train_rel, val_rel = next(inner_skf.split(train_val_idx, inner_labels))

        train_idx = train_val_idx[train_rel]
        val_idx   = train_val_idx[val_rel]

        partitions[f"part_{part_id}"] = {
            "train": train_idx.tolist(),
            "val":   val_idx.tolist(),
            "test":  test_idx.tolist(),
        }

        # Verify class distribution preserved
        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_counts = np.bincount(labels_arr[split_idx], minlength=len(label_names))
            print(f"  part_{part_id} {split_name:5s}: {dict(zip(label_names, split_counts.tolist()))}")

        # Verify disjointness
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"

    # Verify cross-partition disjointness
    all_test = []
    for p in partitions.values():
        all_test.extend(p["test"])
    assert len(all_test) == len(set(all_test)), "Duplicate test indices across partitions!"
    print(f"\n{PASS} All partitions are disjoint (no neuron appears in multiple test sets)")

    # ── Save partitions ───────────────────────────────────────────────────────
    part_path = OUT / "partitions.json"
    with open(part_path, "w") as f:
        json.dump(partitions, f, indent=2)
    print(f"{PASS} Partitions → {part_path}")

    # ── Save label map ────────────────────────────────────────────────────────
    label_map = {name: i for i, name in enumerate(label_names)}
    label_path = OUT / "label_map.json"
    with open(label_path, "w") as f:
        json.dump({"label_names": label_names, "label2idx": label_map}, f, indent=2)
    print(f"{PASS} Label map → {label_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  READY: {len(samples)} neurons, {len(label_names)} classes")
    print(f"  {n_splits} partitions × {n_seeds} seeds = {n_splits * n_seeds} training runs")
    print(f"  Next: python experiments/fafb_perslay/02_compute_persistence.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="morphoclass/data/fafb_subset/dataset.csv")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_seeds",  type=int, default=5)
    parser.add_argument("--min_per_class", type=int, default=8)
    args = parser.parse_args()
    main(args.dataset, args.n_splits, args.n_seeds, args.min_per_class)
