"""
10_tree_analysis.py
===================
Decision tree and Random Forest analysis of morphometric features,
implementing CSE 446 Lecture 15 concepts:
  - Bias-variance tradeoff (depth vs accuracy)
  - Information gain per feature
  - Feature importance: single tree vs RF ensemble
  - Decision tree structure (dark-theme hand-drawn)
  - Per-class RF accuracy across partitions
  - Cable length distribution (explains first split)
  - Cross-partition accuracy + ensemble gain

Outputs:
  outputs/fafb/figures/fig7_tree_analysis.png

Run:
  python experiments/fafb_perslay/10_tree_analysis.py
"""

import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset

DATA_CSV   = Path("outputs/fafb/data/dataset.csv")
MORPH_NPY  = Path("outputs/fafb/data/morphometrics.npy")
PARTS_JSON = Path("outputs/fafb/data/partitions.json")
LABELS_NPY = Path("outputs/fafb/embeddings/labels.npy")
LABEL_JSON = Path("outputs/fafb/embeddings/label_names.json")
OUT_FIGS   = Path("outputs/fafb/figures")

DARK = "#0A1628"; CARD = "#0F2040"; CARD2 = "#152B55"; BORDER = "#1A3050"
TEAL = "#0AAFCC"; TEAL_LT = "#3DD6EF"; AMBER = "#F5A623"; CORAL = "#EF6351"
GREEN = "#22C88A"; PURPLE = "#9B7FE8"; MUTED = "#5A7A9A"; WHITE = "#EEF4FA"
CELL = [TEAL_LT, AMBER, CORAL, GREEN, PURPLE, "#F472B6", "#60A5FA", "#A3E635"]

FEAT_SHORT = ["cable_len", "n_branch", "n_tips", "max_order",
              "soma_r", "seg_len", "asymmetry", "max_path"]

PASS = "\033[92m[PASS]\033[0m"


def entropy(arr, n_classes=8):
    counts = np.bincount(arr, minlength=n_classes).astype(float)
    p = counts[counts > 0] / len(arr)
    return -(p * np.log2(p)).sum()


def main():
    print(f"\n{'='*60}")
    print("  10 — Decision Tree & RF analysis (CSE 446 Lec 15)")
    print(f"{'='*60}\n")

    morph  = np.load(MORPH_NPY)
    labels = np.load(LABELS_NPY)
    with open(LABEL_JSON) as f:
        label_names = json.load(f)
    with open(PARTS_JSON) as f:
        partitions = json.load(f)

    splits = partitions["part_0"]
    X_tr = morph[splits["train"]]; y_tr = labels[splits["train"]]
    X_te = morph[splits["test"]];  y_te = labels[splits["test"]]

    plt.rcParams.update({
        "figure.facecolor": DARK, "axes.facecolor": CARD,
        "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": WHITE, "grid.color": CARD2, "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(20, 24), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

    def sp(ax): [s.set_edgecolor(BORDER) for s in ax.spines.values()]

    # ── A: Bias-variance ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(CARD); sp(ax)
    depths = [1, 2, 3, 4, 5, 6, 7, None]
    tr_a, te_a = [], []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
        tr_a.append(accuracy_score(y_tr, dt.predict(X_tr)))
        te_a.append(accuracy_score(y_te, dt.predict(X_te)))
    xs    = list(range(len(depths)))
    xlbls = [str(d) if d else "∞" for d in depths]
    best_i = te_a.index(max(te_a))
    ax.plot(xs, tr_a, "o-", color=TEAL_LT, lw=2, ms=7, label="Train")
    ax.plot(xs, te_a, "s--", color=AMBER,  lw=2, ms=7, label="Test")
    ax.axhline(1/8, color=MUTED, lw=1, ls=":", label="Chance")
    ax.axvspan(best_i + 0.45, len(xs) - 0.55, alpha=0.07, color=CORAL)
    ax.set_xticks(xs); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_xlabel("Max Depth", fontsize=10); ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("A  Bias–Variance Tradeoff\n(Decision Tree Depth)",
                 color=TEAL_LT, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
    ax.annotate(f"Best test\ndepth={depths[best_i] or '∞'}",
                xy=(best_i, te_a[best_i]), xytext=(best_i + 1.5, te_a[best_i] + 0.08),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
                color=AMBER, fontsize=8)

    # ── B: Information gain ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(CARD); sp(ax)
    root_h = entropy(labels)
    igs = []
    for fi in range(8):
        best = 0.0
        for t in np.percentile(morph[:, fi], np.arange(5, 96, 5)):
            L = labels[morph[:, fi] <= t]; R = labels[morph[:, fi] > t]
            if len(L) == 0 or len(R) == 0: continue
            ig = root_h - (len(L)/len(labels))*entropy(L) - (len(R)/len(labels))*entropy(R)
            best = max(best, ig)
        igs.append(best)
    order = np.argsort(igs)[::-1]
    bar_colors = [TEAL_LT if i == order[0] else PURPLE if igs[i] > 0.05 else MUTED
                  for i in order]
    ax.barh(range(8), [igs[i] for i in order], color=bar_colors, alpha=0.85, height=0.65)
    ax.set_yticks(range(8))
    ax.set_yticklabels([FEAT_SHORT[i] for i in order], fontsize=9)
    ax.set_xlabel("Max Information Gain (bits)", fontsize=10)
    ax.set_title("B  Information Gain per Feature\n(optimal threshold)",
                 color=AMBER, fontsize=11, fontweight="bold", pad=6)

    # ── C: Feature importance DT vs RF ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(CARD); sp(ax)
    dt4   = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    rf200 = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
    x = np.arange(8); w = 0.35
    ax.bar(x - w/2, dt4.feature_importances_,  w, color=TEAL_LT, alpha=0.85, label="DT depth=4")
    ax.bar(x + w/2, rf200.feature_importances_, w, color=AMBER,   alpha=0.85, label="RF 200 trees")
    ax.set_xticks(x); ax.set_xticklabels(FEAT_SHORT, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Importance (Gini)", fontsize=10)
    ax.set_title("C  Feature Importance:\nSingle Tree vs. Random Forest",
                 color=GREEN, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=WHITE)

    # ── D: Hand-drawn tree ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2]); ax.set_facecolor(CARD)
    ax.set_xlim(0, 10); ax.set_ylim(0.5, 10); ax.axis("off")
    dt3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
    tree = dt3.tree_

    def node_txt(n):
        fi = tree.feature[n]
        if fi < 0:
            ci = int(np.argmax(tree.value[n][0]))
            return (f"{label_names[ci][:10]}\n"
                    f"gini={tree.impurity[n]:.2f} n={int(tree.n_node_samples[n])}",
                    True, ci)
        return (f"{FEAT_SHORT[fi]}\n≤{tree.threshold[n]:.0f}\n"
                f"gini={tree.impurity[n]:.2f} n={int(tree.n_node_samples[n])}",
                False, -1)

    pos = {0:(5,9.2), 1:(2.5,7.2), 2:(7.5,7.2),
           3:(1.2,5.2), 4:(3.8,5.2), 5:(6.2,5.2), 6:(8.8,5.2),
           7:(0.5,3.2), 8:(1.9,3.2), 9:(3.2,3.2), 10:(4.5,3.2),
           11:(5.6,3.2), 12:(6.9,3.2), 13:(8.2,3.2), 14:(9.5,3.2)}

    for ni, (x, y) in pos.items():
        if ni >= tree.node_count: continue
        txt, is_leaf, ci = node_txt(ni)
        bg  = CELL[ci % 8] if is_leaf else CARD2
        fc  = DARK if is_leaf else WHITE
        ax.text(x, y, txt, ha="center", va="center", fontsize=7, color=fc,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                          edgecolor=TEAL if not is_leaf else bg, lw=0.8))
        lc = tree.children_left[ni]; rc = tree.children_right[ni]
        if lc > 0 and lc in pos:
            lx, ly = pos[lc]
            ax.annotate("", xy=(lx, ly + 0.4), xytext=(x, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
            ax.text((x+lx)/2-0.2, (y+ly)/2, "T", color=GREEN, fontsize=7, fontweight="bold")
        if rc > 0 and rc in pos:
            rx, ry = pos[rc]
            ax.annotate("", xy=(rx, ry + 0.4), xytext=(x, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.1))
            ax.text((x+rx)/2+0.2, (y+ry)/2, "F", color=CORAL, fontsize=7, fontweight="bold")

    ax.set_title("D  Decision Tree Structure (max depth=3) — "
                 "First split: cable_len ≤ 2109 µm",
                 color=TEAL_LT, fontsize=11, fontweight="bold", pad=6)
    patches = [mpatches.Patch(color=CELL[i], label=label_names[i]) for i in range(8)]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              framealpha=0.25, labelcolor=WHITE, ncol=2,
              title="Leaf class", title_fontsize=8)

    # ── E: Per-class accuracy ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(CARD); sp(ax)
    class_accs = {n: [] for n in label_names}
    for sp_, splits_ in partitions.items():
        rf = RandomForestClassifier(200, random_state=0).fit(
            morph[splits_["train"]], labels[splits_["train"]])
        preds = rf.predict(morph[splits_["test"]])
        for ci, name in enumerate(label_names):
            mask = labels[splits_["test"]] == ci
            if mask.sum() > 0:
                class_accs[name].append((preds[mask] == ci).mean())
    means = [np.mean(v) for v in class_accs.values()]
    stds  = [np.std(v)  for v in class_accs.values()]
    cols  = [GREEN if m >= 0.8 else AMBER if m >= 0.5 else CORAL for m in means]
    order_e = np.argsort(means)
    ax.barh(range(8), [means[i] for i in order_e], xerr=[stds[i] for i in order_e],
            color=[cols[i] for i in order_e], alpha=0.85, height=0.62,
            error_kw=dict(ecolor=WHITE, lw=1.5, capsize=3))
    ax.set_yticks(range(8))
    ax.set_yticklabels([label_names[i][:14] for i in order_e], fontsize=9)
    ax.axvline(1/8, color=MUTED, lw=1, ls=":", label="Chance")
    ax.set_xlabel("RF Accuracy (mean ± std)", fontsize=9)
    ax.set_title("E  Per-Class RF Accuracy\n(Easy vs. Hard)", color=PURPLE,
                 fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
    ax.set_xlim(0, 1.15)

    # ── F: Cable length violin ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :2]); ax.set_facecolor(CARD); sp(ax)
    medians_ci = sorted([(np.median(morph[labels == ci, 0]), ci)
                         for ci in range(8)])
    for rank, (_, ci) in enumerate(medians_ci):
        vals = morph[labels == ci, 0]
        vp = ax.violinplot([vals], positions=[rank], showmedians=True, widths=0.72)
        for pc in vp["bodies"]: pc.set_facecolor(CELL[ci]); pc.set_alpha(0.55)
        for k in ["cmedians", "cmins", "cmaxes", "cbars"]: vp[k].set_color(CELL[ci])
        ax.text(rank, -1000, label_names[ci][:10],
                ha="center", va="top", fontsize=8.5, color=CELL[ci], rotation=30)
    thresh = dt3.tree_.threshold[0]  # actual first split threshold
    ax.axhline(thresh, color=TEAL, lw=2.5, ls="--", alpha=0.9,
               label=f"First split ({thresh:.0f} µm)")
    ax.fill_between([-0.5, 7.5], 0, thresh,  alpha=0.06, color=GREEN)
    ax.fill_between([-0.5, 7.5], thresh, 9500, alpha=0.06, color=CORAL)
    ax.text(3.5, 600,  "Small neurons (below threshold)",
            ha="center", color=GREEN, fontsize=8.5)
    ax.text(3.5, thresh + 500, "Large neurons (above threshold)",
            ha="center", color=CORAL, fontsize=8.5)
    ax.set_ylabel("Total Cable Length (µm)", fontsize=10)
    ax.set_xticks([]); ax.set_ylim(-1500, 9800)
    ax.set_title("F  Cable Length Distribution — Explains First Decision Tree Split",
                 color=AMBER, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE)

    # ── G: Cross-partition accuracy ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2]); ax.set_facecolor(CARD); sp(ax)
    dt_p, rf_p = [], []
    for sp_, splits_ in partitions.items():
        dt4p = DecisionTreeClassifier(max_depth=4, random_state=0).fit(
            morph[splits_["train"]], labels[splits_["train"]])
        rfp  = RandomForestClassifier(200, random_state=0).fit(
            morph[splits_["train"]], labels[splits_["train"]])
        dt_p.append(accuracy_score(labels[splits_["test"]], dt4p.predict(morph[splits_["test"]])))
        rf_p.append(accuracy_score(labels[splits_["test"]], rfp.predict(morph[splits_["test"]])))

    x = np.arange(3); w = 0.32
    ax.bar(x - w/2, dt_p, w, color=TEAL_LT, alpha=0.85, label="DT depth=4")
    ax.bar(x + w/2, rf_p, w, color=AMBER,   alpha=0.85, label="RF 200 trees")
    ax.axhline(np.mean(dt_p), color=TEAL_LT, lw=1.5, ls="--", alpha=0.55)
    ax.axhline(np.mean(rf_p), color=AMBER,   lw=1.5, ls="--", alpha=0.55)
    ax.set_xticks(x); ax.set_xticklabels(["Part 0", "Part 1", "Part 2"], fontsize=9)
    ax.set_ylabel("Test Accuracy", fontsize=10); ax.set_ylim(0, 1.05)
    gain = np.mean(rf_p) - np.mean(dt_p)
    ax.text(1, 0.10, f"Ensemble gain:\n+{gain:.3f}",
            ha="center", color=WHITE, fontsize=9,
            bbox=dict(facecolor=CARD2, edgecolor=BORDER, boxstyle="round,pad=0.4"))
    ax.set_title("G  Ensemble Gain: RF vs Single Tree", color=CORAL,
                 fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=WHITE)

    fig.suptitle("Decision Tree & RF Analysis — FlyWire Neuron Morphometrics\n"
                 "CSE 446 Lec 15: Bias-Variance · Information Gain · "
                 "Feature Importance · Ensemble Methods",
                 color=WHITE, fontsize=13.5, fontweight="bold", y=0.997)

    path = OUT_FIGS / "fig7_tree_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"{PASS} → {path}")

    print(f"\n{'='*60}")
    print(f"  First tree split: {FEAT_SHORT[dt3.tree_.feature[0]]} "
          f"≤ {dt3.tree_.threshold[0]:.0f} µm")
    print(f"  RF mean accuracy: {np.mean(rf_p):.3f}")
    print(f"  Ensemble gain over DT: +{gain:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
