# Makefile — PRH Morphology Validation / FlyWire FAFB
# Run from repo root. All paths relative to repo root.
#
# Usage:
#   make pipeline          run full pipeline end-to-end
#   make figures           regenerate all figures (no retraining)
#   make data              steps 0–2 only
#   make train             step 3 only (15 PersLay models)
#   make metrics           steps 4–5 only
#   make nblast            step 7 (NBLAST pretrained model)
#   make sweep             step 9 (subset scaling sweep)
#   make commit-1          interactive: stage + commit scaffold files
#   make status            show what outputs exist
#   make clean-figures     delete figures so they regenerate fresh
#   make check             check all dependencies are installed

PYTHON := python
EXP    := experiments/fafb_perslay
OUT    := outputs/fafb

# ── Data paths (override on command line if needed) ──────────────────────────
SWC_DIR := data/fafb/swc
CSV     := data/fafb/classification.csv
DATASET := $(OUT)/data/dataset.csv    # produced by 01_make_dataset.py

# ── Full pipeline ─────────────────────────────────────────────────────────────
.PHONY: pipeline
pipeline: check sanity data train metrics figures nblast cross sweep trees
	@echo ""
	@echo "✓ Full pipeline complete. Figures in $(OUT)/figures/"

# ── Individual stages ─────────────────────────────────────────────────────────

.PHONY: sanity
sanity:
	@echo "→ 00 Sanity checks"
	$(PYTHON) $(EXP)/00_sanity_checks.py \
		--swc_dir $(SWC_DIR) \
		--csv $(CSV)

.PHONY: data
data: $(OUT)/data/dataset.csv $(OUT)/data/diagrams.npz

$(OUT)/data/dataset.csv:
	@echo "→ 01 Make dataset + partitions"
	$(PYTHON) $(EXP)/01_make_dataset.py \
		--dataset $(CSV) \
		--n_splits 3 \
		--n_seeds 5 \
		--min_per_class 50

$(OUT)/data/diagrams.npz: $(OUT)/data/dataset.csv
	@echo "→ 02 Compute persistence diagrams + morphometrics"
	$(PYTHON) $(EXP)/02_compute_persistence.py

.PHONY: train
train: $(OUT)/data/diagrams.npz
	@echo "→ 03 Train PersLay (15 models: 3 partitions × 5 seeds)"
	$(PYTHON) $(EXP)/03_train_perslay.py

# Single model (for debugging / testing):
# make train-one PART=part_0 SEED=0
.PHONY: train-one
train-one:
	$(PYTHON) $(EXP)/03_train_perslay.py --partition $(PART) --seed $(SEED)

.PHONY: embeddings
embeddings:
	@echo "→ 04 Extract embeddings"
	$(PYTHON) $(EXP)/04_extract_embeddings.py

.PHONY: metrics
metrics: embeddings
	@echo "→ 05 Compute CKA, silhouette, kNN Jaccard, RF baseline"
	$(PYTHON) $(EXP)/05_metrics.py

.PHONY: figures
figures:
	@echo "→ 06 Generate Figs 1–4 (UMAP, CKA heatmap, silhouette, kNN)"
	$(PYTHON) $(EXP)/06_figures.py
	@echo "→ 08 Generate Figs 5, 8 (interpretability, CKA gradient)"
	$(PYTHON) $(EXP)/08_cross_representations.py
	@echo "→ 09 Generate Fig 6 (subset sweep) — uses cached JSON if available"
	$(PYTHON) $(EXP)/09_subset_sweep.py
	@echo "→ 10 Generate Fig 7 (tree analysis)"
	$(PYTHON) $(EXP)/10_tree_analysis.py
	@echo ""
	@echo "All figures:"
	@ls $(OUT)/figures/*.png 2>/dev/null || echo "  (none yet)"

.PHONY: nblast
nblast:
	@echo "→ 07 NBLAST pretrained model (navis Drosophila scoring matrix)"
	$(PYTHON) $(EXP)/07_nblast.py

.PHONY: nblast-force
nblast-force:
	@echo "→ 07 NBLAST (force recompute)"
	$(PYTHON) $(EXP)/07_nblast.py --force

.PHONY: cross
cross:
	@echo "→ 08 Cross-representation analysis (PI-PCA, graph-topology, interpretability)"
	$(PYTHON) $(EXP)/08_cross_representations.py

.PHONY: sweep
sweep:
	@echo "→ 09 Subset scaling sweep n=16→96"
	$(PYTHON) $(EXP)/09_subset_sweep.py

.PHONY: sweep-rerun
sweep-rerun:
	@echo "→ 09 Subset sweep (force full rerun)"
	$(PYTHON) $(EXP)/09_subset_sweep.py --rerun

.PHONY: trees
trees:
	@echo "→ 10 Decision tree + RF analysis (CSE 446 Lec 15)"
	$(PYTHON) $(EXP)/10_tree_analysis.py

# ── Git commit helpers ────────────────────────────────────────────────────────
# Each target stages only the relevant files and opens an editor for the message.
# Run `git push` after all commits.

.PHONY: commit-1
commit-1:
	git add $(EXP)/utils.py $(EXP)/requirements.txt README.md CLAUDE.md GIT_COMMITS.md Makefile
	git commit

.PHONY: commit-2
commit-2:
	git add $(EXP)/00_sanity_checks.py \
	        $(EXP)/01_make_dataset.py \
	        $(EXP)/02_compute_persistence.py
	git commit

.PHONY: commit-3
commit-3:
	git add $(EXP)/03_train_perslay.py \
	        $(EXP)/04_extract_embeddings.py \
	        $(EXP)/05_metrics.py \
	        $(EXP)/06_figures.py
	git commit

.PHONY: commit-4
commit-4:
	git add $(EXP)/07_nblast.py
	git commit

.PHONY: commit-5
commit-5:
	git add $(EXP)/08_cross_representations.py
	git commit

.PHONY: commit-6
commit-6:
	git add $(EXP)/09_subset_sweep.py
	git commit

.PHONY: commit-7
commit-7:
	git add $(EXP)/10_tree_analysis.py
	git commit

.PHONY: commit-figures
commit-figures:
	@echo "Staging all figures (not models or large .npy files)"
	git add $(OUT)/figures/*.png
	git commit

# ── Status and checks ─────────────────────────────────────────────────────────

.PHONY: status
status:
	@echo "=== Pipeline status ==="
	@echo ""
	@echo "Data:"
	@for f in \
	    "$(OUT)/data/dataset.csv" \
	    "$(OUT)/data/partitions.json" \
	    "$(OUT)/data/diagrams.npz" \
	    "$(OUT)/data/morphometrics.npy"; do \
	    if [ -f "$$f" ]; then echo "  ✓ $$f"; else echo "  ✗ $$f"; fi; \
	done
	@echo ""
	@echo "Models (expect 15):"
	@count=$$(ls $(OUT)/models/perslay_*.npz 2>/dev/null | wc -l); \
	    echo "  $$count / 15 model files"
	@echo ""
	@echo "Embeddings:"
	@for f in \
	    "$(OUT)/embeddings/labels.npy" \
	    "$(OUT)/embeddings/nblast_pca32_emb.npy" \
	    "$(OUT)/embeddings/pi_pca_emb.npy" \
	    "$(OUT)/embeddings/graph_topology_emb.npy"; do \
	    if [ -f "$$f" ]; then echo "  ✓ $$f"; else echo "  ✗ $$f (run the relevant script)"; fi; \
	done
	@echo ""
	@echo "Metrics:"
	@for f in \
	    "$(OUT)/metrics/summary.json" \
	    "$(OUT)/metrics/nblast_results.json" \
	    "$(OUT)/metrics/cross_rep_results.json" \
	    "$(OUT)/metrics/subset_sweep.json"; do \
	    if [ -f "$$f" ]; then echo "  ✓ $$f"; else echo "  ✗ $$f"; fi; \
	done
	@echo ""
	@echo "Figures:"
	@ls $(OUT)/figures/*.png 2>/dev/null | sed 's/^/  ✓ /' || echo "  (none)"

.PHONY: check
check:
	@echo "=== Checking dependencies ==="
	@$(PYTHON) -c "import numpy; print('  ✓ numpy', numpy.__version__)"
	@$(PYTHON) -c "import sklearn; print('  ✓ scikit-learn', sklearn.__version__)"
	@$(PYTHON) -c "import matplotlib; print('  ✓ matplotlib', matplotlib.__version__)"
	@$(PYTHON) -c "import scipy; print('  ✓ scipy', scipy.__version__)"
	@$(PYTHON) -c "import umap; print('  ✓ umap-learn')" 2>/dev/null || \
	    echo "  ✗ umap-learn  →  pip install umap-learn"
	@$(PYTHON) -c "import navis; print('  ✓ navis', navis.__version__)" 2>/dev/null || \
	    echo "  ✗ navis  →  pip install navis"
	@$(PYTHON) -c "import gudhi; print('  ✓ gudhi')" 2>/dev/null || \
	    echo "  ✗ gudhi  →  pip install gudhi"
	@$(PYTHON) -c "import networkx; print('  ✓ networkx')"
	@$(PYTHON) -c "import morphoclass; print('  ✓ morphoclass')" 2>/dev/null || \
	    echo "  ✗ morphoclass  →  pip install -e . (from repo root)"
	@echo ""
	@echo "Data:"
	@[ -d "$(SWC_DIR)" ] && echo "  ✓ SWC directory exists" || \
	    echo "  ✗ $(SWC_DIR) not found — download from codex.flywire.ai"
	@[ -f "$(CSV)" ] && echo "  ✓ classification.csv exists" || \
	    echo "  ✗ $(CSV) not found"

# ── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean-figures
clean-figures:
	@echo "Deleting figures so they regenerate fresh..."
	rm -f $(OUT)/figures/*.png
	@echo "Done. Run 'make figures' to regenerate."

.PHONY: clean-sweep-cache
clean-sweep-cache:
	rm -f $(OUT)/metrics/subset_sweep.json
	@echo "Sweep cache cleared. Next 'make sweep' will rerun from scratch."

# Danger: removes all outputs (keep data, remove everything computed)
.PHONY: clean-outputs
clean-outputs:
	@echo "WARNING: This deletes all computed outputs (models, embeddings, metrics, figures)."
	@read -p "Are you sure? [y/N] " ans && [ "$$ans" = "y" ]
	rm -rf $(OUT)/models/ $(OUT)/embeddings/ $(OUT)/metrics/ $(OUT)/figures/
	mkdir -p $(OUT)/models $(OUT)/embeddings $(OUT)/metrics $(OUT)/figures
	@echo "Done. Run 'make pipeline' to recompute everything."
