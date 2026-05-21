"""
04_extract_embeddings.py
========================
Loads all trained models and collects embeddings + labels into aligned matrices.

Outputs:
  outputs/fafb/embeddings/emb_{part_id}_s{seed}.npy   — (N, D) float
  outputs/fafb/embeddings/labels.npy                  — (N,) int
  outputs/fafb/embeddings/label_names.json
  outputs/fafb/embeddings/morphometric_emb.npy         — (N, 8) baseline

Run:
  python experiments/fafb_perslay/04_extract_embeddings.py
"""

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset

DATA_CSV  = Path("outputs/fafb/data/dataset.csv")
MODEL_DIR = Path("outputs/fafb/models")
OUT       = Path("outputs/fafb/embeddings")
OUT.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def main():
    print(f"\n{'='*60}")
    print("  Extracting and validating embeddings")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    labels = np.array([s.label_idx for s in samples])

    # Save labels and label names once
    np.save(OUT / "labels.npy", labels)
    with open(OUT / "label_names.json", "w") as f:
        json.dump(label_names, f)
    print(f"{PASS} Labels saved: {len(labels)} neurons, {len(label_names)} classes")

    # ── Load embeddings from saved model files ────────────────────────────────
    model_files = sorted(MODEL_DIR.glob("perslay_part_*.npz"))
    if not model_files:
        print(f"{FAIL} No model files found in {MODEL_DIR}")
        print("  Run 03_train_perslay.py first")
        return

    print(f"\nFound {len(model_files)} model files:")
    for mf in model_files:
        data = np.load(mf, allow_pickle=True)
        emb  = data["embeddings"]  # (N, D)

        # Parse part/seed from filename
        stem = mf.stem  # e.g. perslay_part_0_s2
        parts = stem.split("_")
        part_id = f"part_{parts[2]}"
        seed    = int(parts[3][1:])  # s2 → 2

        out_path = OUT / f"emb_{part_id}_s{seed}.npy"
        np.save(out_path, emb)

        # Basic validation
        norm_mean = float(np.linalg.norm(emb, axis=1).mean())
        zero_rows  = int((np.abs(emb).sum(axis=1) == 0).sum())
        print(f"  {mf.name}: shape={emb.shape}  norm_mean={norm_mean:.2f}  zero_rows={zero_rows}")

        if zero_rows > len(samples) * 0.1:
            print(f"  {WARN} Many zero rows — check persistence diagram quality")

    # ── Morphometric baseline embedding ──────────────────────────────────────
    morph_path = Path("outputs/fafb/data/morphometrics.npy")
    if morph_path.exists():
        morph = np.load(morph_path)
        # Standardize features
        mu = morph.mean(axis=0)
        std = morph.std(axis=0) + 1e-9
        morph_norm = (morph - mu) / std
        np.save(OUT / "morphometric_emb.npy", morph_norm)
        print(f"\n{PASS} Morphometric baseline embedding: shape={morph_norm.shape}")
    else:
        print(f"\n{WARN} No morphometrics.npy found — run 02_compute_persistence.py first")

    print(f"\n{PASS} All embeddings in {OUT}/")
    print(f"Next: python experiments/fafb_perslay/05_metrics.py\n")


if __name__ == "__main__":
    main()
