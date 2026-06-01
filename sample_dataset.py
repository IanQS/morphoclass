"""
sample_dataset.py
=================
Sample neurons from the full FlyWire dataset (sk_lod1_783_healed.zip)
and write them to data/fafb_sample/ (overwriting the existing mini-sample).

Extracts only the selected SWC files from the zip — does NOT unpack all 13 GB.

Usage:
  python sample_dataset.py                          # 100 per class (default)
  python sample_dataset.py --per-class 200          # 200 per class
  python sample_dataset.py --per-class 50 --seed 7  # reproducible different sample

Reads:
  /gscratch/scrubbed/emazuh/otterpack/data/sk_lod1_783_healed.zip
  /gscratch/scrubbed/emazuh/otterpack/data/classification.csv.gz

Writes:
  data/fafb_sample/swc/<root_id>.swc          (one per sampled neuron)
  data/fafb_sample/classification_subset.csv  (full metadata)
  data/fafb_sample/dataset.csv               (path<TAB>label, for pipeline)
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ZIP_PATH = Path("/gscratch/scrubbed/emazuh/otterpack/data/sk_lod1_783_healed.zip")
CSV_PATH = Path("/gscratch/scrubbed/emazuh/otterpack/data/classification.csv.gz")
OUT_DIR  = Path("data/fafb_sample")
OUT_SWC  = OUT_DIR / "swc"

LABEL_COL = "class"
# All 20 classes with >= 50 neurons in the full FlyWire dataset.
# Ordered roughly by neuron count (descending).
TARGET_CLASSES = [
    "optic_lobe_intrinsic", "visual", "Kenyon_Cell", "CX",
    "mechanosensory", "olfactory", "AN", "ALPN",
    "LHLN", "ALLN", "gustatory", "DAN",
    "bilateral", "TuBu", "unknown_sensory", "brain_motor_neuron",
    "MBON", "mAL", "hygrosensory", "ocellar",
]


def main(per_class: int, seed: int) -> None:
    print(f"\nSampling {per_class} neurons/class × {len(TARGET_CLASSES)} classes "
          f"= up to {per_class * len(TARGET_CLASSES)} neurons  (seed={seed})\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df["root_id"] = df["root_id"].astype(str)
    df = df[df[LABEL_COL].isin(TARGET_CLASSES) & df[LABEL_COL].notna()]

    # ── Index SWC names in zip (no extraction yet) ────────────────────────────
    print("Indexing zip file (this takes ~10 s)…")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zip_stems = {Path(n).stem for n in zf.namelist()}
    print(f"  {len(zip_stems):,} SWC files found in zip\n")

    df = df[df["root_id"].isin(zip_stems)]

    # ── Sample per class ──────────────────────────────────────────────────────
    rows = []
    for cls in TARGET_CLASSES:
        pool = df[df[LABEL_COL] == cls]
        n    = min(per_class, len(pool))
        rows.append(pool.sample(n=n, random_state=seed))
        print(f"  {cls}: sampled {n}/{len(pool)}")
    df_out = pd.concat(rows).reset_index(drop=True)
    print(f"\n  Total: {len(df_out)} neurons\n")

    # ── Prepare output directory ──────────────────────────────────────────────
    if OUT_SWC.exists():
        shutil.rmtree(OUT_SWC)
    OUT_SWC.mkdir(parents=True)

    # ── Extract selected SWC files from zip ───────────────────────────────────
    print("Extracting SWC files from zip…")
    target_names = {f"{rid}.swc" for rid in df_out["root_id"]}
    extracted = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            if member.filename in target_names:
                member.filename = Path(member.filename).name  # strip any path prefix
                zf.extract(member, OUT_SWC)
                extracted += 1
                if extracted % 100 == 0:
                    print(f"  {extracted}/{len(target_names)} extracted…")
    print(f"  Done — {extracted} files extracted\n")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_DIR / "classification_subset.csv", index=False)

    with open(OUT_DIR / "dataset.csv", "w") as f:
        for _, row in df_out.iterrows():
            swc_path = OUT_SWC / f"{row['root_id']}.swc"
            f.write(f"{swc_path}\t{row[LABEL_COL]}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_mb = sum((OUT_SWC / f"{r}.swc").stat().st_size
                   for r in df_out["root_id"]) / 1024 / 1024
    print(f"Output:  {OUT_DIR}/")
    print(f"Neurons: {len(df_out)}")
    print(f"Size:    {total_mb:.1f} MB")
    print("\nClass counts:")
    print(df_out[LABEL_COL].value_counts().to_string())
    print(f"\nNext: re-run the pipeline from step 01 (or step 02 if classes unchanged):")
    print(f"  python experiments/fafb_perslay/01_make_dataset.py")
    print(f"  python experiments/fafb_perslay/02_compute_persistence.py")
    print(f"  bash submit_all.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100,
                        help="Max neurons per class (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()
    main(args.per_class, args.seed)
