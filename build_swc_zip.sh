#!/bin/bash
# build_swc_zip.sh
# Zip the 17,288 SWC skeleton files used in the 17k/20-class PRH experiment.
# Run from the repo root. This takes 10-30 min (~4.7 GB input, ~2 GB output).
#
# Output: prh_swc_files_<date>.zip

set -euo pipefail

DATE=$(date +%Y%m%d)
ZIPNAME="prh_swc_files_${DATE}.zip"

if [[ ! -d "data/fafb_sample/swc" ]]; then
    echo "ERROR: data/fafb_sample/swc/ not found. Run from repo root."
    exit 1
fi

N=$(ls data/fafb_sample/swc/*.swc 2>/dev/null | wc -l)
echo "========================================"
echo "  Zipping $N SWC files → $ZIPNAME"
echo "========================================"

zip -q -r "$ZIPNAME" data/fafb_sample/swc/
zip -q "$ZIPNAME" data/fafb_sample/dataset.csv
zip -q "$ZIPNAME" data/fafb_sample/root_ids.txt

echo ""
echo "========================================"
echo "  Done: $ZIPNAME"
du -sh "$ZIPNAME"
echo "========================================"
