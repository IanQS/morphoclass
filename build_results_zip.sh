#!/bin/bash
# build_results_zip.sh
# Assemble a self-contained zip with all PRH experiment results.
# Run from the repo root after all jobs have completed.
# Includes SWC skeleton files used in the 17k experiment.
#
# Output: prh_morphology_results_<date>.zip

set -euo pipefail

DATE=$(date +%Y%m%d)
ZIPNAME="prh_morphology_results_${DATE}.zip"
TMP="prh_results_tmp_${DATE}"

echo "========================================"
echo "  Building PRH results zip: $ZIPNAME"
echo "========================================"

# ── Sanity checks ──────────────────────────────────────────────────────────────
required=(
    "outputs/fafb/metrics/summary.json"
    "outputs/fafb/metrics/progress_report.json"
    "outputs/fafb/figures/report_combined.png"
    "outputs/fafb/figures/fig_crossarch_cka.png"
    "data/fafb_sample/dataset.csv"
    "COMMANDS.md"
    "methodology.tex"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING: $f — run the pipeline first"
        exit 1
    fi
done

# ── Warn if CNNet progress report missing ─────────────────────────────────────
if [[ ! -f "outputs/fafb/metrics/progress_report_cnn.json" ]]; then
    echo "WARN: outputs/fafb/metrics/progress_report_cnn.json not found"
    echo "      CNNet progress report still running? Continuing without it."
fi
if [[ ! -f "outputs/fafb/figures/report_combined_cnn.png" ]]; then
    echo "WARN: outputs/fafb/figures/report_combined_cnn.png not found"
fi

# ── Build zip ─────────────────────────────────────────────────────────────────
echo ""
echo "Compiling zip (SWC files are ~4.7 GB, this may take 10-30 min)..."
echo ""

# Models (exclude .npz — too large; include only .pt checkpoints)
echo "  → models/ (36 .pt files)"
zip -q "$ZIPNAME" outputs/fafb/models/perslay_part_*.pt
zip -q "$ZIPNAME" outputs/fafb/models/cnn_part_*.pt
zip -q "$ZIPNAME" outputs/fafb/models/run_log.csv
zip -q "$ZIPNAME" outputs/fafb/models/cnn_run_log.csv

# Embeddings
echo "  → embeddings/ (logit + feat .npy)"
zip -q "$ZIPNAME" outputs/fafb/embeddings/logit_part_*.npy
zip -q "$ZIPNAME" outputs/fafb/embeddings/emb_part_*.npy
zip -q "$ZIPNAME" outputs/fafb/embeddings/cnn_logit_part_*.npy
zip -q "$ZIPNAME" outputs/fafb/embeddings/cnn_emb_part_*.npy
zip -q "$ZIPNAME" outputs/fafb/embeddings/labels.npy
zip -q "$ZIPNAME" outputs/fafb/embeddings/label_names.json 2>/dev/null || true

# Metrics
echo "  → metrics/"
zip -q "$ZIPNAME" outputs/fafb/metrics/*.json
zip -q "$ZIPNAME" outputs/fafb/metrics/*.npy
zip -q "$ZIPNAME" outputs/fafb/metrics/*.csv

# Figures
echo "  → figures/"
zip -q "$ZIPNAME" outputs/fafb/figures/*.png

# Data manifests (not the 4.7 GB SWC dir yet)
echo "  → data/fafb_sample/ (manifests)"
zip -q "$ZIPNAME" data/fafb_sample/dataset.csv
zip -q "$ZIPNAME" data/fafb_sample/classification_subset.csv
zip -q "$ZIPNAME" data/fafb_sample/root_ids.txt
zip -q "$ZIPNAME" data/fafb_sample/viewer_sample.csv
zip -q "$ZIPNAME" data/fafb_sample/viewer_sample_all_classes.csv

# Partition splits
zip -q "$ZIPNAME" outputs/fafb/data/partitions.json
zip -q "$ZIPNAME" outputs/fafb/data/label_map.json 2>/dev/null || true

# Documentation
echo "  → COMMANDS.md, methodology.tex, CLAUDE.md, README.md"
zip -q "$ZIPNAME" COMMANDS.md methodology.tex CLAUDE.md
[[ -f README.md ]] && zip -q "$ZIPNAME" README.md || true

# Scripts
echo "  → experiment scripts"
zip -q -r "$ZIPNAME" experiments/fafb_perslay/ \
    --exclude "*.pyc" --exclude "__pycache__/*" --exclude "*.pyo"

# SWC files are in a separate zip: prh_swc_files_<date>.zip
# Run build_swc_zip.sh to generate it.

echo ""
echo "========================================"
echo "  Done: $ZIPNAME"
du -sh "$ZIPNAME"
echo "========================================"
