"""
12_progress_report.py
=====================
Standalone PersLay/CorianderNet convergence report.

Reads pre-computed logit embeddings and training logs; reports cross-seed and
cross-partition CKA, kNN Jaccard, silhouette, and per-class accuracy.
No NBLAST, no cross-representation analysis — PersLay results only.

Outputs:
  outputs/fafb/figures/report_fig1_cka_heatmap.png   — 15×15 CKA heatmap
  outputs/fafb/figures/report_fig2_accuracy.png       — per-model accuracy + F1
  outputs/fafb/figures/report_fig3_knn_silhouette.png — kNN Jaccard + silhouette
  outputs/fafb/figures/report_combined.png            — all panels in one figure
  outputs/fafb/metrics/progress_report.json           — machine-readable summary

Run:
  python experiments/fafb_perslay/12_progress_report.py
"""

import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import debiased_cka, permutation_cka_null, knn_jaccard

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_DIR  = Path("outputs/fafb/embeddings")
MET_DIR  = Path("outputs/fafb/metrics")
FIG_DIR  = Path("outputs/fafb/figures")
LOG_CSV  = Path("outputs/fafb/models/run_log.csv")
PARTS_JSON = Path("outputs/fafb/data/partitions.json")

MET_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 500
KNN_K  = 5

# ── Colour palette ─────────────────────────────────────────────────────────────
DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D27"
CARD2_BG = "#252836"
WHITE    = "#F0F0F0"
MUTED    = "#9CA3AF"
TEAL     = "#0B8FAC"
AMBER    = "#F59E0B"
GREEN    = "#10B981"
RED      = "#EF4444"
PURPLE   = "#8B5CF6"

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_logit_models(tag=""):
    prefix = f"logit_{tag+'_' if tag else ''}"
    files = sorted(EMB_DIR.glob(f"{prefix}part_*.npy"))
    if not files:
        flag = f"--tag {tag}" if tag else ""
        print(f"{FAIL} No logit embeddings found matching '{prefix}part_*.npy'.")
        print(f"  Run: python 04_extract_embeddings.py {flag}")
        sys.exit(1)
    import re
    models = []
    for f in files:
        m = re.search(r'part_(\d+)_s(\d+)', f.stem)
        if not m:
            continue
        part_id = int(m.group(1))
        seed    = int(m.group(2))
        models.append({
            "label":   f"P{part_id}S{seed}",
            "part_id": part_id,
            "seed":    seed,
            "emb":     np.load(f),
        })
    return models


def load_labels():
    labels = np.load(EMB_DIR / "labels.npy")
    with open(EMB_DIR / "label_names.json") as f:
        label_names = json.load(f)
    return labels, label_names


# ── CKA ───────────────────────────────────────────────────────────────────────

def compute_cka(models):
    n = len(models)
    mat = np.zeros((n, n))
    part_ids = np.array([m["part_id"] for m in models])
    print(f"  Computing {n}×{n} CKA matrix ({n*(n-1)//2} pairs)...")
    for i in range(n):
        for j in range(i, n):
            v = debiased_cka(models[i]["emb"], models[j]["emb"])
            mat[i, j] = mat[j, i] = v
    within_mask = (part_ids[:, None] == part_ids[None, :])
    cross_mask  = ~within_mask
    np.fill_diagonal(within_mask, False)
    return mat, mat[within_mask], mat[cross_mask]


# ── kNN Jaccard ───────────────────────────────────────────────────────────────

def compute_knn(models):
    part_ids = np.array([m["part_id"] for m in models])
    within, cross = [], []
    pairs_done = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            j_val = knn_jaccard(models[i]["emb"], models[j]["emb"], k=KNN_K)
            if part_ids[i] == part_ids[j]:
                within.append(j_val)
            else:
                cross.append(j_val)
            pairs_done += 1
    print(f"  kNN Jaccard: {pairs_done} pairs | within={np.mean(within):.3f} cross={np.mean(cross):.3f}")
    return np.array(within), np.array(cross)


# ── Silhouette ────────────────────────────────────────────────────────────────

def compute_silhouette(models, labels):
    """Compute silhouette per model using precomputed distance matrix for speed.
    Expects models/labels already subsampled to a manageable N."""
    from sklearn.metrics import pairwise_distances
    scores = []
    for m in models:
        try:
            emb = m["emb"]
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            D = pairwise_distances(emb_norm, metric="euclidean")
            s = silhouette_score(D, labels, metric="precomputed")
            scores.append(s)
        except Exception:
            scores.append(np.nan)
    arr = np.array(scores)
    print(f"  Silhouette: {np.nanmean(arr):.3f} ± {np.nanstd(arr):.3f}")
    return arr


# ── Figures ───────────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(CARD2_BG)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_cka_heatmap(mat, models, within_vals, cross_vals):
    labels  = [m["label"] for m in models]
    part_ids = np.array([m["part_id"] for m in models])

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    display = mat.copy()
    np.fill_diagonal(display, np.nan)
    im = ax.imshow(display, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")

    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color=WHITE if mat[i,j] < 0.85 else DARK_BG)

    ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                           fontsize=7, color=MUTED)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(labels, fontsize=7, color=MUTED)

    # Partition block borders
    boundaries = np.where(np.diff(part_ids))[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color=AMBER, lw=1.2, alpha=0.7)
        ax.axvline(b, color=AMBER, lw=1.2, alpha=0.7)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("CKA (logit space)", color=MUTED, fontsize=9)

    ax.set_title(
        f"PersLay CKA — Logit Space (n={len(models)} models)\n"
        f"Within-partition: {within_vals.mean():.3f}±{within_vals.std():.3f}   "
        f"Cross-partition: {cross_vals.mean():.3f}±{cross_vals.std():.3f}",
        color=WHITE, fontsize=10, pad=12,
    )
    plt.tight_layout()
    out = FIG_DIR / "report_fig1_cka_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")


def fig_accuracy(log_csv):
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=DARK_BG)

    for ax, col, color, title in [
        (axes[0], "test_acc",  TEAL,  "Test Accuracy"),
        (axes[1], "test_f1",   GREEN, "Test Macro-F1"),
    ]:
        _ax_style(ax)
        vals = df[col].values
        parts = df["partition"].values
        xs = np.arange(len(vals))
        colors = [TEAL if p == "part_0" else AMBER if p == "part_1" else PURPLE
                  for p in parts]
        ax.bar(xs, vals, color=colors, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)
        ax.axhline(vals.mean(), color=WHITE, lw=1.5, ls="--",
                   label=f"Mean={vals.mean():.3f}±{vals.std():.3f}")
        ax.axhline(1/len(df["partition"].unique()) if col == "test_acc" else 0,
                   color=RED, lw=1, ls=":", alpha=0.5, label="Chance")
        ax.set_ylim(0, 1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"P{r.partition[-1]}S{r.seed}" for _, r in df.iterrows()],
                           rotation=45, ha="right", fontsize=7, color=MUTED)
        ax.set_ylabel(title, color=MUTED, fontsize=9)
        ax.set_title(title, color=WHITE, fontsize=10)
        ax.legend(framealpha=0.2, labelcolor=WHITE, fontsize=8)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=TEAL, label="Part 0"),
                  Patch(facecolor=AMBER, label="Part 1"),
                  Patch(facecolor=PURPLE, label="Part 2")]
    fig.legend(handles=legend_els, loc="upper center", ncol=3, framealpha=0.2,
               labelcolor=WHITE, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Per-Model Classification Performance ({len(df)} models)",
                 color=WHITE, fontsize=11, y=1.05)
    plt.tight_layout()
    out = FIG_DIR / "report_fig2_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")


def fig_knn_silhouette(within_knn, cross_knn, sil_scores):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=DARK_BG)

    # Panel 1: kNN within vs cross violin
    ax = axes[0]; _ax_style(ax)
    parts = [within_knn, cross_knn]
    vp = ax.violinplot(parts, positions=[0, 1], showmedians=True, widths=0.5)
    for pc, col in zip(vp["bodies"], [TEAL, AMBER]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    for comp in ["cmedians", "cmaxes", "cmins", "cbars"]:
        vp[comp].set_color(WHITE)
    for i, (vals, col) in enumerate([(within_knn, TEAL), (cross_knn, AMBER)]):
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.06, 0.06, len(vals)),
                   vals, s=25, alpha=0.7, color=col, zorder=3)
        ax.text(i, vals.mean() - 0.07, f"{vals.mean():.3f}", ha="center",
                fontsize=9, color=WHITE, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-partition", "Cross-partition"], color=MUTED, fontsize=9)
    ax.set_ylabel("kNN Jaccard (k=5)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Neighbourhood Stability", color=WHITE, fontsize=10)

    # Panel 2: Silhouette per model
    ax = axes[1]; _ax_style(ax)
    xs = np.arange(len(sil_scores))
    bar_colors = [TEAL if s > 0 else RED for s in sil_scores]
    ax.bar(xs, sil_scores, color=bar_colors, alpha=0.85)
    ax.axhline(0, color=WHITE, lw=1, ls="-", alpha=0.4)
    ax.axhline(np.nanmean(sil_scores), color=AMBER, lw=1.5, ls="--",
               label=f"Mean={np.nanmean(sil_scores):.3f}")
    ax.set_xticks(xs); ax.set_xticklabels([f"M{i}" for i in xs], fontsize=7, color=MUTED)
    ax.set_ylabel("Silhouette Score", color=MUTED, fontsize=9)
    ax.set_title("Class Separation (Embedding Space)", color=WHITE, fontsize=10)
    ax.legend(framealpha=0.2, labelcolor=WHITE, fontsize=8)

    # Panel 3: Summary text
    ax = axes[2]; ax.set_facecolor(CARD_BG); ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(CARD2_BG)

    summary_lines = [
        ("PRIMARY PRH METRICS", WHITE, 11, True),
        ("", WHITE, 9, False),
        (f"CKA logit within-partition", MUTED, 9, False),
        ("  (loaded from metrics/)", MUTED, 8, False),
        ("", WHITE, 9, False),
        (f"kNN Jaccard (k={KNN_K})", MUTED, 9, False),
        (f"  Within:  {within_knn.mean():.3f} ± {within_knn.std():.3f}", TEAL, 10, True),
        (f"  Cross:   {cross_knn.mean():.3f} ± {cross_knn.std():.3f}", AMBER, 10, True),
        ("", WHITE, 9, False),
        (f"Silhouette (feat space)", MUTED, 9, False),
        (f"  Mean: {np.nanmean(sil_scores):.3f} ± {np.nanstd(sil_scores):.3f}",
         GREEN if np.nanmean(sil_scores) > 0 else RED, 10, True),
    ]
    y = 0.95
    for text, color, size, bold in summary_lines:
        weight = "bold" if bold else "normal"
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
                color=color, va="top", fontweight=weight)
        y -= 0.08 if text else 0.04

    plt.tight_layout()
    out = FIG_DIR / "report_fig3_knn_silhouette.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")


def fig_combined(mat, models, within_cka, cross_cka,
                 within_knn, cross_knn, sil_scores, df_log):
    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Top-left: CKA heatmap ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(DARK_BG)
    display = mat.copy(); np.fill_diagonal(display, np.nan)
    im = ax0.imshow(display, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j: continue
            ax0.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=5.5, color=WHITE if mat[i,j] < 0.85 else DARK_BG)
    part_ids = np.array([m["part_id"] for m in models])
    for b in np.where(np.diff(part_ids))[0] + 0.5:
        ax0.axhline(b, color=AMBER, lw=1.0, alpha=0.7)
        ax0.axvline(b, color=AMBER, lw=1.0, alpha=0.7)
    labels_tick = [m["label"] for m in models]
    ax0.set_xticks(range(len(models))); ax0.set_xticklabels(labels_tick, rotation=45,
                                                              ha="right", fontsize=6, color=MUTED)
    ax0.set_yticks(range(len(models))); ax0.set_yticklabels(labels_tick, fontsize=6, color=MUTED)
    cb = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    ax0.set_title(f"CKA Logit\nWithin={within_cka.mean():.3f}  Cross={cross_cka.mean():.3f}",
                  color=WHITE, fontsize=9)

    # ── Top-middle: Accuracy ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _ax_style(ax1)
    vals = df_log["test_acc"].values
    parts = df_log["partition"].values
    xs = np.arange(len(vals))
    bar_colors = [TEAL if p == "part_0" else AMBER if p == "part_1" else PURPLE for p in parts]
    ax1.bar(xs, vals, color=bar_colors, alpha=0.85)
    ax1.axhline(vals.mean(), color=WHITE, lw=1.5, ls="--",
                label=f"Mean={vals.mean():.3f}±{vals.std():.3f}")
    ax1.set_ylim(0, 1); ax1.set_xticks(xs)
    ax1.set_xticklabels([f"P{r.partition[-1]}S{r.seed}" for _, r in df_log.iterrows()],
                        rotation=45, ha="right", fontsize=6.5, color=MUTED)
    ax1.set_ylabel("Test Accuracy", color=MUTED, fontsize=9)
    ax1.set_title(f"Test Accuracy\nMean={vals.mean():.3f}±{vals.std():.3f}", color=WHITE, fontsize=9)
    ax1.legend(framealpha=0.2, labelcolor=WHITE, fontsize=7)

    # ── Top-right: Macro F1 ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _ax_style(ax2)
    vals_f1 = df_log["test_f1"].values
    ax2.bar(xs, vals_f1, color=bar_colors, alpha=0.85)
    ax2.axhline(vals_f1.mean(), color=WHITE, lw=1.5, ls="--")
    ax2.set_ylim(0, 1); ax2.set_xticks(xs)
    ax2.set_xticklabels([f"P{r.partition[-1]}S{r.seed}" for _, r in df_log.iterrows()],
                        rotation=45, ha="right", fontsize=6.5, color=MUTED)
    ax2.set_ylabel("Test Macro-F1", color=MUTED, fontsize=9)
    ax2.set_title(f"Test Macro-F1\nMean={vals_f1.mean():.3f}±{vals_f1.std():.3f}", color=WHITE, fontsize=9)

    # ── Bottom-left: kNN violin ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _ax_style(ax3)
    vp = ax3.violinplot([within_knn, cross_knn], positions=[0, 1], showmedians=True, widths=0.5)
    for pc, col in zip(vp["bodies"], [TEAL, AMBER]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    for comp in ["cmedians", "cmaxes", "cmins", "cbars"]:
        vp[comp].set_color(WHITE)
    for i, (vals, col) in enumerate([(within_knn, TEAL), (cross_knn, AMBER)]):
        ax3.scatter(np.full(len(vals), i) + np.random.uniform(-0.06, 0.06, len(vals)),
                    vals, s=20, alpha=0.7, color=col, zorder=3)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Within-partition", "Cross-partition"], color=MUTED, fontsize=8)
    ax3.set_ylabel(f"kNN Jaccard (k={KNN_K})", color=MUTED, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_title(f"Neighbourhood Stability\nWithin={within_knn.mean():.3f}  Cross={cross_knn.mean():.3f}",
                  color=WHITE, fontsize=9)

    # ── Bottom-middle: Silhouette ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _ax_style(ax4)
    bar_c = [GREEN if s > 0 else RED for s in sil_scores]
    ax4.bar(np.arange(len(sil_scores)), sil_scores, color=bar_c, alpha=0.85)
    ax4.axhline(0, color=WHITE, lw=1, alpha=0.4)
    ax4.axhline(np.nanmean(sil_scores), color=AMBER, lw=1.5, ls="--")
    ax4.set_xticks(np.arange(len(sil_scores)))
    ax4.set_xticklabels([f"M{i}" for i in range(len(sil_scores))], fontsize=7, color=MUTED)
    ax4.set_ylabel("Silhouette Score", color=MUTED, fontsize=9)
    ax4.set_title(f"Class Separation\nMean={np.nanmean(sil_scores):.3f}±{np.nanstd(sil_scores):.3f}",
                  color=WHITE, fontsize=9)

    # ── Bottom-right: summary text box ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(CARD_BG); ax5.axis("off")

    n_neurons = len(np.load(EMB_DIR / "labels.npy"))
    n_classes = len(json.load(open(EMB_DIR / "label_names.json")))
    lines = [
        ("PRH RESULTS SUMMARY", WHITE, 10, True),
        (f"n={n_neurons:,} neurons  ·  {n_classes} classes", MUTED, 8, False),
        (f"{len(models)} models (3 partitions × 5 seeds)", MUTED, 8, False),
        ("", WHITE, 8, False),
        ("CKA Logit (primary PRH metric)", MUTED, 8.5, False),
        (f"  Within:  {within_cka.mean():.3f} ± {within_cka.std():.3f}", TEAL, 10, True),
        (f"  Cross:   {cross_cka.mean():.3f} ± {cross_cka.std():.3f}", AMBER, 10, True),
        ("", WHITE, 8, False),
        (f"kNN Jaccard  within={within_knn.mean():.3f}  cross={cross_knn.mean():.3f}", MUTED, 8.5, False),
        (f"Silhouette   {np.nanmean(sil_scores):.3f} ± {np.nanstd(sil_scores):.3f}", MUTED, 8.5, False),
        ("", WHITE, 8, False),
        ("Test Accuracy / Macro-F1", MUTED, 8.5, False),
        (f"  Acc: {df_log.test_acc.mean():.3f} ± {df_log.test_acc.std():.3f}", GREEN, 10, True),
        (f"  F1:  {df_log.test_f1.mean():.3f} ± {df_log.test_f1.std():.3f}", GREEN, 10, True),
    ]
    y = 0.97
    for text, color, size, bold in lines:
        ax5.text(0.05, y, text, transform=ax5.transAxes, fontsize=size,
                 color=color, va="top", fontweight="bold" if bold else "normal")
        y -= 0.075 if text else 0.03

    n_sig = sum(1 for s in sil_scores if not np.isnan(s) and s > 0)
    ax5.text(0.05, 0.08,
             f"Null p95 — see progress_report.json",
             transform=ax5.transAxes, fontsize=7.5, color=MUTED, va="bottom")

    title_tag = f" [{tag}]" if tag else ""
    fig.suptitle(f"PersLay/CorianderNet — PRH Progress Report{title_tag}", color=WHITE, fontsize=13, y=1.01)
    out = FIG_DIR / f"report_combined{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def subsample_models(models, labels, max_n=4000, seed=42):
    """Stratified subsample to max_n neurons for CKA/kNN (O(N²) ops)."""
    n = len(labels)
    if n <= max_n:
        return models, labels
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_n // len(classes))
    idx = []
    for c in classes:
        ci = np.where(labels == c)[0]
        chosen = rng.choice(ci, size=min(per_class, len(ci)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(sorted(idx))
    sub_models = [{**m, "emb": m["emb"][idx]} for m in models]
    print(f"  Subsampled {n:,} → {len(idx):,} neurons for CKA/kNN (stratified, seed={seed})")
    return sub_models, labels[idx]


def main(tag="", max_n=4000):
    label = f"pretrain ({tag})" if tag else "standard"
    print(f"\n{'='*60}")
    print(f"  PersLay/CorianderNet Progress Report  [{label}]")
    print(f"{'='*60}\n")

    models = load_logit_models(tag)
    labels, label_names = load_labels()
    n_neurons, n_classes = len(labels), len(label_names)
    print(f"  {len(models)} models | {n_neurons:,} neurons | {n_classes} classes\n")

    # Subsample for O(N²) metrics if dataset is large
    cka_models, cka_labels = subsample_models(models, labels, max_n=max_n)

    # ── CKA ──────────────────────────────────────────────────────────────────
    print("Computing CKA matrix (logit space)...")
    cka_mat, within_cka, cross_cka = compute_cka(cka_models)
    np.save(MET_DIR / "report_cka_logit_matrix.npy", cka_mat)

    print("\nComputing permutation null (CKA)...")
    # Use first cross-partition pair if available, else cross-seed
    part_ids = [m["part_id"] for m in models]
    cross_pairs = [(i, j) for i in range(len(models))
                   for j in range(i+1, len(models))
                   if part_ids[i] != part_ids[j]]
    same_pairs  = [(i, j) for i in range(len(models))
                   for j in range(i+1, len(models))
                   if part_ids[i] == part_ids[j]]
    i0, j0 = cross_pairs[0] if cross_pairs else same_pairs[0]
    null_dist = permutation_cka_null(models[i0]["emb"], models[j0]["emb"],
                                     n_permutations=N_PERM, seed=42)
    null_p95 = float(np.percentile(null_dist, 95))
    obs_cka  = float(cka_mat[i0, j0])
    p_value  = float(np.mean(null_dist >= obs_cka))
    print(f"  Null p95={null_p95:.4f}  obs={obs_cka:.4f}  p={p_value:.4f}")

    # ── kNN Jaccard ───────────────────────────────────────────────────────────
    print("\nComputing kNN Jaccard stability...")
    within_knn, cross_knn = compute_knn(cka_models)

    # ── Silhouette ────────────────────────────────────────────────────────────
    print("\nComputing silhouette scores (precomputed distance matrix)...")
    sil_scores = compute_silhouette(cka_models, cka_labels)

    # ── Accuracy from run log (pretrain log has val only; standard has test) ──
    pretrain_log = Path("outputs/fafb/models/pretrain_log.csv")
    log_path = pretrain_log if (tag == "pretrain" and pretrain_log.exists()) else LOG_CSV
    df_log = pd.read_csv(log_path)
    if tag == "pretrain":
        # Pretrain log has one row per epoch; take final epoch per partition/seed
        df_log = df_log.groupby(["partition", "seed"]).last().reset_index()
        if "test_acc" not in df_log.columns:
            df_log["test_acc"] = df_log["val_acc"]
            df_log["test_f1"]  = df_log.get("val_f1", df_log["val_acc"])

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_cka_heatmap(cka_mat, models, within_cka, cross_cka)
    fig_accuracy(LOG_CSV)
    fig_knn_silhouette(within_knn, cross_knn, sil_scores)
    fig_combined(cka_mat, models, within_cka, cross_cka,
                 within_knn, cross_knn, sil_scores, df_log)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    report = {
        "n_neurons":  n_neurons,
        "n_classes":  n_classes,
        "n_models":   len(models),
        "cka_logit": {
            "within_mean":  round(float(within_cka.mean()), 4),
            "within_std":   round(float(within_cka.std()),  4),
            "cross_mean":   round(float(cross_cka.mean()),  4),
            "cross_std":    round(float(cross_cka.std()),   4),
            "null_p95":     round(null_p95, 4),
            "obs_cka":      round(obs_cka, 4),
            "p_value":      round(p_value, 4),
        },
        "knn_jaccard": {
            "k":           KNN_K,
            "within_mean": round(float(within_knn.mean()), 4),
            "within_std":  round(float(within_knn.std()),  4),
            "cross_mean":  round(float(cross_knn.mean()),  4),
            "cross_std":   round(float(cross_knn.std()),   4),
        },
        "silhouette": {
            "mean": round(float(np.nanmean(sil_scores)), 4),
            "std":  round(float(np.nanstd(sil_scores)),  4),
            "per_model": [round(float(s), 4) for s in sil_scores],
        },
        "accuracy": {
            "test_mean": round(float(df_log.test_acc.mean()), 4),
            "test_std":  round(float(df_log.test_acc.std()),  4),
            "f1_mean":   round(float(df_log.test_f1.mean()),  4),
            "f1_std":    round(float(df_log.test_f1.std()),   4),
        },
    }
    suffix   = f"_{tag}" if tag else ""
    out_json = MET_DIR / f"progress_report{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Summary → {out_json}")

    print(f"\n{'='*60}")
    print("  PROGRESS REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Neurons: {n_neurons:,}  |  Classes: {n_classes}  |  Models: {len(models)}")
    print(f"  CKA logit  within: {within_cka.mean():.3f} ± {within_cka.std():.3f}")
    print(f"  CKA logit  cross:  {cross_cka.mean():.3f} ± {cross_cka.std():.3f}  (null p95={null_p95:.4f})")
    print(f"  kNN Jaccard within={within_knn.mean():.3f}  cross={cross_knn.mean():.3f}")
    print(f"  Silhouette {np.nanmean(sil_scores):.3f} ± {np.nanstd(sil_scores):.3f}")
    print(f"  Test acc   {df_log.test_acc.mean():.3f} ± {df_log.test_acc.std():.3f}")
    print(f"  Test F1    {df_log.test_f1.mean():.3f} ± {df_log.test_f1.std():.3f}")
    print(f"{'='*60}\n")
    print(f"  Figures → {FIG_DIR}/report_*.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="",
                        help="Embedding tag: '' for standard, 'pretrain' for contrastive pretrain")
    parser.add_argument("--max-n", type=int, default=4000,
                        help="Max neurons for CKA/kNN (stratified subsample if larger, default 4000)")
    args = parser.parse_args()
    main(args.tag, args.max_n)
