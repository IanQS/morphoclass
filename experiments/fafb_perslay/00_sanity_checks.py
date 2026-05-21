"""
00_sanity_checks.py
===================
Run this top-to-bottom before any training.
Each check prints PASS / WARN / FAIL and explains what to do if it fails.
No morphoclass import required — pure stdlib + numpy + pandas + matplotlib.

Usage:
    python 00_sanity_checks.py --swc_dir . --csv classification.csv
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--swc_dir", default=".", help="Directory containing *.swc files")
parser.add_argument("--csv",     default="classification.csv")
parser.add_argument("--label_col", default="class",
                    help="Which label column to use: flow | super_class | class | sub_class")
parser.add_argument("--min_neurons_per_class", type=int, default=50)
parser.add_argument("--max_classes", type=int, default=10)
parser.add_argument("--out_dir", default="outputs/sanity")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

PASS  = "\033[92m[PASS]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
BLOCK = "\033[91m[BLOCKING]\033[0m"

issues = []  # collect all FAIL items for summary at the end

def fail(msg):
    print(f"{FAIL}  {msg}")
    issues.append(msg)

def warn(msg):
    print(f"{WARN}  {msg}")

def ok(msg):
    print(f"{PASS}  {msg}")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 1 — SWC files exist and are non-empty
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 1: SWC file inventory ─────────────────────────────────────")

swc_dir = Path(args.swc_dir)
swc_paths = list(swc_dir.glob("*.swc"))

if not swc_paths:
    fail(f"No .swc files found in {swc_dir}. Check --swc_dir.")
    sys.exit(1)

ok(f"Found {len(swc_paths):,} SWC files")

empty = [p for p in swc_paths if p.stat().st_size == 0]
if empty:
    fail(f"{len(empty)} empty SWC files: {[p.name for p in empty[:5]]}")
else:
    ok("No empty SWC files")

# Build id→path map (stem = root_id)
swc_map = {p.stem: p for p in swc_paths}


# ════════════════════════════════════════════════════════════════════════════
# CHECK 2 — CSV loads and join succeeds
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 2: CSV join ───────────────────────────────────────────────")

df = pd.read_csv(args.csv)
ok(f"CSV loaded: {len(df):,} rows, columns: {list(df.columns)}")

if "root_id" not in df.columns:
    fail("'root_id' column missing from CSV. Cannot join to SWC filenames.")
    sys.exit(1)

df["root_id_str"] = df["root_id"].astype(str)
df["have_swc"] = df["root_id_str"].isin(swc_map)

n_matched  = df["have_swc"].sum()
n_csv_only = (~df["have_swc"]).sum()
csv_ids    = set(df["root_id_str"])
swc_ids    = set(swc_map.keys())
n_swc_only = len(swc_ids - csv_ids)

ok(f"CSV rows matched to SWC: {n_matched:,} / {len(df):,}")

if n_csv_only > 0:
    warn(f"{n_csv_only:,} CSV rows have no matching SWC — will be dropped")
if n_swc_only > 0:
    warn(f"{n_swc_only:,} SWC files have no CSV entry — will be ignored")
if n_matched == 0:
    fail("Zero rows matched. root_id format mismatch? "
         f"CSV sample: {df['root_id_str'].iloc[0]}  "
         f"SWC sample: {list(swc_map.keys())[0]}")
    sys.exit(1)

df_matched = df[df["have_swc"]].copy()


# ════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Label column quality
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Check 3: Label column '{args.label_col}' ─────────────────────────")

if args.label_col not in df_matched.columns:
    fail(f"Column '{args.label_col}' not found. Available: {list(df_matched.columns)}")
    sys.exit(1)

null_frac = df_matched[args.label_col].isna().mean()
ok(f"Null rate in '{args.label_col}': {null_frac:.1%}")

if null_frac > 0.3:
    warn(f"Over 30% nulls — consider a coarser label column")

df_labeled = df_matched[df_matched[args.label_col].notna()].copy()
value_counts = df_labeled[args.label_col].value_counts()

print(f"\n  Class distribution (top 20):")
print(value_counts.head(20).to_string())

# Save full distribution
fig, ax = plt.subplots(figsize=(12, 5))
top = value_counts.head(30)
ax.bar(range(len(top)), top.values, color="#534AB7", alpha=0.8)
ax.set_xticks(range(len(top)))
ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=8)
ax.axhline(args.min_neurons_per_class, color="coral", linestyle="--",
           label=f"min_neurons={args.min_neurons_per_class}")
ax.set_title(f"Class distribution — '{args.label_col}' (top 30)")
ax.set_ylabel("neuron count")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=150)
plt.close()
ok(f"Saved class distribution → {OUT}/class_distribution.png")

# Recommend usable classes
usable = value_counts[value_counts >= args.min_neurons_per_class]
ok(f"Classes with ≥{args.min_neurons_per_class} neurons: {len(usable)} "
   f"({usable.sum():,} neurons total)")

if len(usable) < 3:
    fail(f"Only {len(usable)} usable classes — try a coarser label column or lower --min_neurons_per_class")
elif len(usable) > args.max_classes:
    warn(f"{len(usable)} usable classes — consider capping at {args.max_classes} "
         f"largest for a cleaner first experiment")

# Class imbalance ratio
if len(usable) >= 2:
    ratio = usable.iloc[0] / usable.iloc[-1]
    if ratio > 10:
        warn(f"Imbalance ratio {ratio:.1f}x (largest / smallest). "
             f"Consider stratified sampling or class weights in loss.")
    else:
        ok(f"Imbalance ratio: {ratio:.1f}x (acceptable)")

# Show all label hierarchy columns together for guidance
print("\n  Null rates across all label columns:")
for col in ["flow", "super_class", "class", "sub_class", "hemilineage"]:
    if col in df_matched.columns:
        n_unique = df_matched[col].nunique()
        null_r   = df_matched[col].isna().mean()
        print(f"    {col:20s}  {n_unique:4d} unique  {null_r:.1%} null")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 4 — SWC coordinate scale (THE most important check)
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 4: Coordinate scale ───────────────────────────────────────")

def parse_swc(path):
    """Return (N, 7) array of SWC rows, skipping comment lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                rows.append([float(x) for x in parts[:7]])
    return np.array(rows) if rows else None

# Sample 50 random SWCs
sample_paths = list(swc_map.values())
rng = np.random.default_rng(0)
sample = rng.choice(sample_paths,
                    size=min(50, len(sample_paths)),
                    replace=False)

all_xyz    = []
node_counts = []

for p in sample:
    arr = parse_swc(p)
    if arr is not None and len(arr) > 0:
        all_xyz.append(arr[:, 2:5])   # columns 2,3,4 = x,y,z
        node_counts.append(len(arr))

all_xyz = np.vstack(all_xyz)
x, y, z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]

x_range = x.max() - x.min()
y_range = y.max() - y.min()
z_range = z.max() - z.min()
median_span = np.median([x_range, y_range, z_range])

print(f"  x range: {x.min():.0f} → {x.max():.0f}  (span {x_range:.0f})")
print(f"  y range: {y.min():.0f} → {y.max():.0f}  (span {y_range:.0f})")
print(f"  z range: {z.min():.0f} → {z.max():.0f}  (span {z_range:.0f})")
print(f"  median span: {median_span:.0f}")

# FlyWire FAFB is in nm. Typical neuron spans ~100–500 µm = 100,000–500,000 nm.
# morphoclass persistence was tuned on µm-scale (Allen/BBP) where spans ~100–2000.
if median_span > 10_000:
    scale_unit = "nm"
    scale_factor = 1000.0
    warn(f"Coordinates appear to be in NANOMETERS (span {median_span:.0f}).")
    print(f"  → You MUST divide x,y,z by {scale_factor:.0f} before computing persistence.")
    print(f"  → Add a rescale step in your dataset loader or pre_transform.")
    print(f"  → Without this, persistence image resolution will be wrong and")
    print(f"     PersLay will learn nothing meaningful.")
    issues.append("Coordinate rescaling required (nm → µm): divide by 1000")
elif median_span > 500:
    scale_unit = "µm (possibly)"
    scale_factor = 1.0
    warn(f"Coordinates span {median_span:.0f} — likely µm but verify with lab metadata.")
else:
    scale_unit = "µm"
    scale_factor = 1.0
    ok(f"Coordinates appear to be in µm (span {median_span:.0f}). No rescaling needed.")

# Radii (column 5) — should be positive, biologically ~0.1–10 µm
radii = all_xyz[:, 0]   # reuse buffer — actually load col 5
radii_raw = []
for p in sample[:10]:
    arr = parse_swc(p)
    if arr is not None:
        radii_raw.append(arr[:, 5])
radii_all = np.concatenate(radii_raw)
if radii_all.min() < 0:
    fail("Negative radius values found in SWC files — malformed files.")
if scale_factor == 1000.0:
    radii_um = radii_all / scale_factor
else:
    radii_um = radii_all
if radii_um.max() > 50:
    warn(f"Radii up to {radii_um.max():.1f} µm after rescaling — unusually large, verify.")
else:
    ok(f"Radii range after rescale: {radii_um.min():.3f}–{radii_um.max():.2f} µm (looks OK)")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 5 — SWC structural validity
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 5: SWC tree structure ─────────────────────────────────────")

def check_swc_structure(path):
    """Returns (n_nodes, n_roots, has_cycle, has_orphan)."""
    arr = parse_swc(path)
    if arr is None or len(arr) == 0:
        return 0, 0, False, False
    ids     = set(arr[:, 0].astype(int))
    parents = arr[:, 6].astype(int)
    roots   = np.sum(parents == -1)
    orphans = sum(1 for p in parents if p != -1 and p not in ids)
    # Naive cycle check: walk up from each node, limit depth
    id_to_parent = {int(r[0]): int(r[6]) for r in arr}
    has_cycle = False
    for node_id in list(id_to_parent.keys())[:20]:  # sample
        visited, cur = set(), node_id
        while cur != -1:
            if cur in visited:
                has_cycle = True
                break
            visited.add(cur)
            cur = id_to_parent.get(cur, -1)
    return len(arr), roots, has_cycle, orphans > 0

struct_sample = sample[:20]
multi_root = []
cyclic     = []
orphaned   = []
tiny        = []

for p in struct_sample:
    n, roots, cycle, orphan = check_swc_structure(p)
    if n < 10:
        tiny.append(p.name)
    if roots > 1:
        multi_root.append(p.name)
    if cycle:
        cyclic.append(p.name)
    if orphan:
        orphaned.append(p.name)

node_counts_arr = np.array(node_counts)
ok(f"Node count distribution (sample n={len(node_counts)}): "
   f"min={node_counts_arr.min()}, median={int(np.median(node_counts_arr))}, "
   f"max={node_counts_arr.max()}")

if tiny:
    warn(f"{len(tiny)} SWCs with <10 nodes (degenerate skeletons): {tiny[:3]}")
if multi_root:
    warn(f"{len(multi_root)} SWCs with >1 root node — "
         f"morphoclass expects single-root trees. May need pruning.")
if cyclic:
    fail(f"{len(cyclic)} SWCs appear to have cycles — malformed.")
if orphaned:
    warn(f"{len(orphaned)} SWCs have orphan nodes (parent IDs not in file).")
if not (tiny or multi_root or cyclic or orphaned):
    ok("Tree structure looks clean (single root, no cycles, no orphans)")

# Node count histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(node_counts_arr, bins=40, color="#1D9E75", alpha=0.8, edgecolor="white")
ax.set_xlabel("nodes per SWC")
ax.set_ylabel("count")
ax.set_title("Node count distribution (sample)")
plt.tight_layout()
plt.savefig(OUT / "node_count_hist.png", dpi=150)
plt.close()
ok(f"Saved node count histogram → {OUT}/node_count_hist.png")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Persistence diagram preview (3 classes, 3 neurons each)
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 6: Persistence diagram preview ────────────────────────────")

try:
    from morphoclass.data import MorphologyDataset
    import morphoclass.transforms as T

    # Pick top-3 usable classes, 3 neurons each
    preview_classes = usable.head(3).index.tolist()
    df_preview = df_labeled[df_labeled[args.label_col].isin(preview_classes)]
    preview_rows = []
    for cls in preview_classes:
        rows = df_preview[df_preview[args.label_col] == cls].head(3)
        for _, row in rows.iterrows():
            p = swc_map[row["root_id_str"]]
            preview_rows.append((str(p), cls))

    # Write a temp CSV
    tmp_csv = OUT / "_preview.csv"
    with open(tmp_csv, "w") as f:
        f.write("\n".join(f"{path}\t{label}" for path, label in preview_rows))

    transforms = T.Compose([
        T.BranchingOnlyNeurites(),
        T.PersistenceDiagram(),
    ])
    ds = MorphologyDataset.from_csv(str(tmp_csv), pre_transform=transforms)

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for i, sample_item in enumerate(ds):
        ax = axes[i // 3][i % 3]
        pd_data = sample_item.diagram   # birth/death pairs
        if pd_data is not None and len(pd_data) > 0:
            births = pd_data[:, 0].numpy()
            deaths = pd_data[:, 1].numpy()
            ax.scatter(births, deaths, s=8, alpha=0.6, color="#534AB7")
            max_val = max(deaths.max(), births.max()) * 1.05
            ax.plot([0, max_val], [0, max_val], "k--", lw=0.5, alpha=0.3)
        label = preview_rows[i][1]
        ax.set_title(f"{label}\n({Path(preview_rows[i][0]).stem[:12]}...)",
                     fontsize=8)
        ax.set_xlabel("birth", fontsize=7)
        ax.set_ylabel("death", fontsize=7)

    plt.suptitle("Persistence diagrams — 3 classes × 3 neurons", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "persistence_preview.png", dpi=150)
    plt.close()
    ok(f"Saved persistence diagram preview → {OUT}/persistence_preview.png")
    print("  → Visually verify: diagrams from different classes should look different.")
    print("     If all diagrams look identical or all empty, coordinate scale is wrong.")

except ImportError:
    warn("morphoclass not importable — skipping persistence diagram preview.")
    warn("Run checks 1–5 first; come back to this after env setup.")
except Exception as e:
    warn(f"Persistence preview failed: {e}")
    warn("Most likely cause: coordinate scale. Re-check output of Check 4.")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Produce the final filtered dataset CSV
# ════════════════════════════════════════════════════════════════════════════
print("\n── Check 7: Write filtered dataset CSV ─────────────────────────────")

df_final = df_labeled[df_labeled[args.label_col].isin(usable.index)].copy()

# Cap to top-N classes
top_classes = usable.head(args.max_classes).index
df_final = df_final[df_final[args.label_col].isin(top_classes)]

rows = []
for _, row in df_final.iterrows():
    path = swc_map[row["root_id_str"]]
    label = row[args.label_col]
    rows.append(f"{path}\t{label}")

out_csv = OUT / "fafb_dataset.csv"
with open(out_csv, "w") as f:
    f.write("\n".join(rows))

ok(f"Wrote {len(rows):,} neurons × {df_final[args.label_col].nunique()} classes "
   f"→ {out_csv}")

print("\n  Final class breakdown:")
for cls, cnt in df_final[args.label_col].value_counts().items():
    bar = "█" * (cnt // 20)
    print(f"    {cls:35s}  {cnt:5d}  {bar}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n══ SUMMARY ══════════════════════════════════════════════════════════")
if issues:
    print(f"{BLOCK} {len(issues)} issue(s) must be resolved before training:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    sys.exit(1)
else:
    print(f"{PASS} All checks passed. Safe to proceed to 01_make_dataset.py")
    print(f"     Dataset CSV ready at: {out_csv}")
