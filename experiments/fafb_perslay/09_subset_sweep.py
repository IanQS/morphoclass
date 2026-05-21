"""
09_subset_sweep.py
==================
Runs all experiments across a range of dataset sizes (n=16→96) to predict
behaviour at full scale (n=783) and characterise the Brbić (2026) CKA
inflation effect at small n.

For each subset size:
  - Stratified sample from the full dataset
  - Train PersLay (3 partitions × n_seeds)
  - Train RF morphometric baseline
  - Build graph-topology embedding
  - Compute: CKA within-partition, cross-partition, cross-architecture
  - Compute: kNN Jaccard within + cross partition
  - Compute: permutation null p95 (inflation baseline)

Outputs:
  outputs/fafb/metrics/subset_sweep.json
  outputs/fafb/figures/fig6_subset_sweep.png

Run:
  python experiments/fafb_perslay/09_subset_sweep.py
"""

import sys, json, time, warnings
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_dataset, parse_swc, swc_to_tree,
                   debiased_cka, permutation_cka_null, knn_jaccard)

DATA_CSV   = Path("outputs/fafb/data/dataset.csv")
DIAGS_NPZ  = Path("outputs/fafb/data/diagrams.npz")
MORPH_NPY  = Path("outputs/fafb/data/morphometrics.npy")
OUT_MET    = Path("outputs/fafb/metrics")
OUT_FIGS   = Path("outputs/fafb/figures")

# Palette
DARK = "#0A1628"; CARD = "#0F2040"; CARD2 = "#152B55"; BORDER = "#1A3050"
TEAL = "#0AAFCC"; AMBER = "#F5A623"; CORAL = "#EF6351"; GREEN = "#22C88A"
PURPLE = "#9B7FE8"; WHITE = "#EEF4FA"; MUTED = "#5A7A9A"

N_CLASSES = 8
PASS = "\033[92m[PASS]\033[0m"

# Subset sizes and matching configs (n_total, n_splits, n_seeds)
CONFIGS = [
    (16, 2, 3),
    (24, 3, 3),
    (32, 3, 4),
    (48, 3, 5),
    (64, 3, 5),
    (80, 3, 5),
    (96, 3, 5),
]
N_PERM = 200   # permutations per null estimate (use 1000 for final paper)


# ── Minimal PersLay (numpy, no torch) ────────────────────────────────────────

class PersLayNumpy:
    def __init__(self, K=32, sigma=10.0, alpha=0.3, seed=0):
        self.K = K; self.sigma = sigma; self.alpha = alpha
        self.rng = np.random.default_rng(seed); self.lm = None

    def fit(self, diagram_list):
        pts = np.vstack([d for d in diagram_list if len(d) > 0])
        self.lm = pts[self.rng.choice(len(pts),
                                       size=min(self.K, len(pts)),
                                       replace=False)].astype("f4")
        return self

    def transform_one(self, d):
        if self.lm is None or len(d) == 0:
            return np.zeros(self.K, "f4")
        p = d.astype("f4")
        w = np.tanh(self.alpha * (p[:, 1] - p[:, 0]))
        phi = np.exp(-((p[:, None, :] - self.lm[None, :, :]) ** 2).sum(-1)
                     / (2 * self.sigma ** 2))
        return ((w[:, None] * phi).sum(0) / (w.sum() + 1e-9)).astype("f4")

    def transform(self, diagram_list):
        return np.stack([self.transform_one(d) for d in diagram_list])


# ── Graph topology embedding (quick version) ──────────────────────────────────

def gt_emb_fn(path, scale=1/1000, n_bins=16):
    arr = parse_swc(path); arr[:, 2:5] *= scale
    ch = swc_to_tree(arr)
    id2r = {int(r[0]): r for r in arr}
    roots = [int(r[0]) for r in arr if int(r[6]) == -1]
    if not roots: return np.zeros(n_bins * 4, "f4")
    root = roots[0]; order = []; q = deque([root])
    while q:
        n = q.popleft(); order.append(n)
        for c in ch.get(n, []): q.append(c)
    degrees = np.array([len(ch.get(n, [])) for n in id2r])
    hd, _ = np.histogram(degrees, bins=n_bins, range=[0, 20])
    st = {n: 1 for n in id2r}
    for n in reversed(order):
        for c in ch.get(n, []): st[n] += st[c]
    sz = np.array(list(st.values()), "f4")
    hs, _ = np.histogram(sz / (sz.max() + 1e-9), bins=n_bins, range=[0, 1])
    hop = {root: 0}
    for n in order[1:]: p = int(id2r[n][6]); hop[n] = hop.get(p, 0) + 1
    hops = np.array(list(hop.values()), "f4")
    hh, _ = np.histogram(hops / (hops.max() + 1e-9), bins=n_bins, range=[0, 1])
    bo = {root: 0}
    for n in order[1:]:
        p = int(id2r[n][6])
        bo[n] = bo.get(p, 0) + (1 if len(ch.get(p, [])) >= 2 else 0)
    bov = np.array(list(bo.values()), "f4")
    hbo, _ = np.histogram(bov / (bov.max() + 1e-9), bins=n_bins, range=[0, 1])
    f = np.concatenate([hd, hs, hh, hbo]).astype("f4")
    return f / (f.sum() + 1e-9)


def manual_stratified_split(n_total, sub_labels, n_splits, seed=42):
    rng = np.random.default_rng(seed)
    n_cls = len(np.unique(sub_labels))
    by_cls = [np.where(sub_labels == c)[0] for c in range(n_cls)]
    for g in by_cls: rng.shuffle(g)
    fold_asgn = np.zeros(n_total, dtype=int)
    for g in by_cls:
        for i, idx in enumerate(g):
            fold_asgn[idx] = i % n_splits
    return [(np.where(fold_asgn != f)[0], np.where(fold_asgn == f)[0])
            for f in range(n_splits)]


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(samples, labels, morph, diagrams, gt_raw):
    results = []
    rng_g = np.random.default_rng(42)

    for n_total, n_splits, n_seeds in CONFIGS:
        npc = n_total // N_CLASSES
        t0 = time.time()

        # Stratified sample
        sub_idx = []
        for ci in range(N_CLASSES):
            ci_idx = np.where(labels == ci)[0]
            sub_idx.extend(rng_g.choice(ci_idx, size=npc, replace=False).tolist())
        sub_idx = np.array(sub_idx)

        sl   = labels[sub_idx]
        sm   = morph[sub_idx]
        sgr  = gt_raw[sub_idx]
        sd   = [diagrams[samples[i].path.stem] for i in sub_idx]

        pca_gt = PCA(n_components=min(16, n_total - 1), random_state=0)
        sgt = pca_gt.fit_transform(sgr).astype("f4")

        folds = manual_stratified_split(n_total, sl, n_splits)
        all_pl = {}
        pl_accs, gt_accs, rf_accs = [], [], []

        for pi, (tr, te) in enumerate(folds):
            ytr, yte = sl[tr], sl[te]
            if len(np.unique(ytr)) < 2: continue

            # RF
            rf = RandomForestClassifier(100, random_state=0).fit(sm[tr], ytr)
            rf_accs.append(accuracy_score(yte, rf.predict(sm[te])))

            # Graph-topology
            clf_gt = LogisticRegression(max_iter=500, C=0.1, random_state=0)
            clf_gt.fit(sgt[tr], ytr)
            gt_accs.append(accuracy_score(yte, clf_gt.predict(sgt[te])))

            for seed in range(n_seeds):
                pl = PersLayNumpy(K=32, sigma=10.0,
                                  alpha=0.3 + seed * 0.1, seed=seed)
                pl.fit([sd[i] for i in tr])
                emb = pl.transform(sd)
                all_pl[(pi, seed)] = emb

                clf = LogisticRegression(max_iter=500, C=0.1,
                                         random_state=seed).fit(emb[tr], ytr)
                pl_accs.append(accuracy_score(yte, clf.predict(emb[te])))

        def pairs(pis, seeds):
            return [(all_pl[(p, s1)], all_pl[(p, s2)])
                    for p in pis for s1 in seeds for s2 in seeds
                    if s1 < s2 and (p, s1) in all_pl and (p, s2) in all_pl]

        within_cka = [debiased_cka(a, b)
                      for a, b in pairs(range(n_splits), range(n_seeds))]
        cross_cka  = [debiased_cka(all_pl[(p1, s1)], all_pl[(p2, s2)])
                      for p1 in range(n_splits) for p2 in range(p1+1, n_splits)
                      for s1 in range(n_seeds) for s2 in range(n_seeds)
                      if (p1, s1) in all_pl and (p2, s2) in all_pl]
        cross_arch = [debiased_cka(all_pl[(pi, seed)], sgt)
                      for pi in range(n_splits) for seed in range(n_seeds)
                      if (pi, seed) in all_pl]

        k_nn = max(2, min(5, npc - 1))
        within_jac = [knn_jaccard(all_pl[(p, s1)], all_pl[(p, s2)], k=k_nn)
                      for p in range(n_splits) for s1 in range(n_seeds)
                      for s2 in range(s1+1, n_seeds)
                      if (p, s1) in all_pl and (p, s2) in all_pl]
        cross_jac  = [knn_jaccard(all_pl[(p1, s)], all_pl[(p2, s)], k=k_nn)
                      for p1 in range(n_splits) for p2 in range(p1+1, n_splits)
                      for s in range(n_seeds)
                      if (p1, s) in all_pl and (p2, s) in all_pl]

        null_p95 = 0.0
        if len(folds) >= 2 and (0, 0) in all_pl and (1, 0) in all_pl:
            null = permutation_cka_null(all_pl[(0, 0)], all_pl[(1, 0)],
                                        n_permutations=N_PERM, seed=0)
            null_p95 = float(np.percentile(null, 95))

        row = dict(
            n=n_total, n_per_class=npc,
            pl_acc=float(np.mean(pl_accs)) if pl_accs else 0.0,
            pl_acc_std=float(np.std(pl_accs)) if pl_accs else 0.0,
            rf_acc=float(np.mean(rf_accs)) if rf_accs else 0.0,
            gt_acc=float(np.mean(gt_accs)) if gt_accs else 0.0,
            chance=1.0 / N_CLASSES,
            within_cka=float(np.mean(within_cka)) if within_cka else 0.0,
            within_cka_std=float(np.std(within_cka)) if within_cka else 0.0,
            cross_cka=float(np.mean(cross_cka)) if cross_cka else 0.0,
            cross_cka_std=float(np.std(cross_cka)) if cross_cka else 0.0,
            cross_arch_cka=float(np.mean(cross_arch)) if cross_arch else 0.0,
            cross_arch_cka_std=float(np.std(cross_arch)) if cross_arch else 0.0,
            within_jac=float(np.mean(within_jac)) if within_jac else 0.0,
            cross_jac=float(np.mean(cross_jac)) if cross_jac else 0.0,
            null_p95=null_p95,
            elapsed=float(time.time() - t0),
        )
        results.append(row)
        print(f"  n={n_total:3d} ({npc}/cls) | "
              f"PL={row['pl_acc']:.3f} RF={row['rf_acc']:.3f} | "
              f"w-CKA={row['within_cka']:.3f} x-CKA={row['cross_cka']:.3f} "
              f"arch={row['cross_arch_cka']:.3f} | "
              f"null={row['null_p95']:.3f} | {row['elapsed']:.1f}s")

    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_sweep(results):
    plt.rcParams.update({
        "figure.facecolor": DARK, "axes.facecolor": CARD,
        "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": WHITE, "grid.color": CARD2, "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    ns       = np.array([r["n"] for r in results])
    w_cka    = np.array([r["within_cka"] for r in results])
    x_cka    = np.array([r["cross_cka"] for r in results])
    a_cka    = np.array([r["cross_arch_cka"] for r in results])
    null95   = np.array([r["null_p95"] for r in results])
    pl_acc   = np.array([r["pl_acc"] for r in results])
    rf_acc   = np.array([r["rf_acc"] for r in results])
    gt_acc   = np.array([r["gt_acc"] for r in results])
    w_jac    = np.array([r["within_jac"] for r in results])
    x_jac    = np.array([r["cross_jac"] for r in results])
    w_std    = np.array([r["within_cka_std"] for r in results])
    x_std    = np.array([r["cross_cka_std"] for r in results])
    pl_std   = np.array([r["pl_acc_std"] for r in results])

    fig = plt.figure(figsize=(20, 13), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    def sp(ax): [s.set_edgecolor(BORDER) for s in ax.spines.values()]

    # A: accuracy
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(CARD); sp(ax)
    ax.plot(ns, rf_acc, "o-", color=AMBER, lw=2, ms=7, label="RF morphometric")
    ax.plot(ns, pl_acc, "s-", color=TEAL,  lw=2, ms=7, label="PersLay-PD")
    ax.plot(ns, gt_acc, "^-", color=PURPLE, lw=2, ms=7, label="Graph-Topology")
    ax.fill_between(ns, pl_acc - pl_std, pl_acc + pl_std, alpha=0.14, color=TEAL)
    ax.axhline(1/N_CLASSES, color=MUTED, lw=1, ls=":", label="Chance")
    z = np.polyfit(ns, pl_acc, 2)
    xs_ext = np.linspace(96, 800, 50)
    ax.plot(xs_ext, np.clip(np.polyval(z, xs_ext), 0, 1),
            "--", color=TEAL, lw=1, alpha=0.5, label="PL trend (extrapolated)")
    ax.axvline(783, color=CORAL, lw=1, ls=":", alpha=0.6)
    ax.text(795, 0.15, "783\n(full)", color=CORAL, fontsize=8)
    ax.set_xlim(0, 850); ax.set_ylim(0, 1.0)
    ax.set_xlabel("n neurons", fontsize=10); ax.set_ylabel("Test accuracy", fontsize=10)
    ax.set_title("A  Accuracy vs dataset size", color=TEAL, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)

    # B: CKA with inflation band
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(CARD); sp(ax)
    ax.fill_between(ns, 0, null95, alpha=0.20, color=CORAL, label="Permutation null p95")
    ax.plot(ns, w_cka, "o-", color=TEAL,   lw=2, ms=7, label="Within-partition")
    ax.fill_between(ns, w_cka - w_std, w_cka + w_std, alpha=0.12, color=TEAL)
    ax.plot(ns, x_cka, "s-", color=AMBER,  lw=2, ms=7, label="Cross-partition")
    ax.fill_between(ns, x_cka - x_std, x_cka + x_std, alpha=0.12, color=AMBER)
    ax.plot(ns, a_cka, "^-", color=PURPLE, lw=2, ms=7, label="Cross-architecture")
    ax.axvline(783, color=CORAL, lw=1, ls=":", alpha=0.6)
    ax.annotate("inflation\nzone", xy=(20, 0.92), xytext=(55, 0.77),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2),
                color=CORAL, fontsize=8)
    ax.set_xlim(0, 850); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("n neurons", fontsize=10); ax.set_ylabel("Debiased CKA", fontsize=10)
    ax.set_title("B  CKA vs dataset size\n(null band = Brbić inflation)",
                 color=AMBER, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)

    # C: Jaccard
    ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(CARD); sp(ax)
    ax.plot(ns, w_jac, "o-", color=TEAL,  lw=2, ms=7, label="Within-partition")
    ax.plot(ns, x_jac, "s-", color=AMBER, lw=2, ms=7, label="Cross-partition")
    ax.fill_between(ns, x_jac, w_jac, alpha=0.12, color=GREEN, label="Stability gap")
    ax.axvline(783, color=CORAL, lw=1, ls=":", alpha=0.6)
    ax.set_xlim(0, 850); ax.set_ylim(0, 1.0)
    ax.set_xlabel("n neurons", fontsize=10)
    ax.set_ylabel("kNN Jaccard (k=2–5)", fontsize=10)
    ax.set_title("C  kNN Jaccard vs dataset size\n(Aristotelian metric, no inflation)",
                 color=GREEN, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=WHITE)

    # D: PRH gap
    ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor(CARD); sp(ax)
    gap_cka = w_cka - x_cka; gap_jac = w_jac - x_jac
    ax.plot(ns, gap_cka, "o-", color=TEAL,  lw=2, ms=7, label="CKA gap")
    ax.plot(ns, gap_jac, "s-", color=AMBER, lw=2, ms=7, label="Jaccard gap")
    ax.fill_between(ns, 0, np.maximum(gap_cka, 0), alpha=0.10, color=CORAL)
    ax.axhline(0, color=MUTED, lw=1, ls="--", alpha=0.5)
    ax.axvline(783, color=CORAL, lw=1, ls=":", alpha=0.6)
    ax.set_xlim(0, 850)
    ax.set_xlabel("n neurons", fontsize=10)
    ax.set_ylabel("Within − cross gap", fontsize=10)
    ax.set_title("D  PRH convergence gap\n(gap→0 = representations converge)",
                 color=CORAL, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=WHITE)

    # E: null-corrected signal
    ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(CARD); sp(ax)
    above_null = x_cka - null95
    ax.plot(ns, null95,     "o-", color=CORAL,  lw=2, ms=7, label="Null p95 (inflation)")
    ax.plot(ns, above_null, "s-", color=TEAL,   lw=2, ms=7, label="Cross-partition above null")
    ax.plot(ns, a_cka,      "^-", color=PURPLE, lw=2, ms=7, label="Cross-architecture CKA")
    ax.fill_between(ns, 0, above_null, where=above_null > 0, alpha=0.12, color=TEAL)
    ax.axhline(0, color=MUTED, lw=1, ls="--", alpha=0.4)
    ax.axvline(783, color=CORAL, lw=1, ls=":", alpha=0.6)
    ax.set_xlim(0, 850)
    ax.set_xlabel("n neurons", fontsize=10); ax.set_ylabel("CKA value", fontsize=10)
    ax.set_title("E  Null-corrected CKA\n(real signal = above null band)",
                 color=PURPLE, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)

    # F: summary table
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(CARD); ax.axis("off")
    ax.text(0.5, 1.03, "F  Subset sweep summary", transform=ax.transAxes,
            ha="center", color=TEAL, fontsize=11, fontweight="bold")
    cols = ["n", "npc", "PL", "RF", "GT", "w-CKA", "x-CKA", "arch", "x-Jac", "null"]
    xs_tab = np.array([0, 0.07, 0.14, 0.21, 0.28, 0.36, 0.45, 0.54, 0.64, 0.74])
    cols_clr = [MUTED, MUTED, TEAL, AMBER, PURPLE, TEAL, AMBER, PURPLE, GREEN, CORAL]
    for i, (col, x) in enumerate(zip(cols, xs_tab)):
        ax.text(x, 0.95, col, transform=ax.transAxes, fontsize=9,
                color=AMBER, fontweight="bold")
    for ri, r in enumerate(results):
        y = 0.85 - ri * 0.115
        vals = [str(r["n"]), str(r["n_per_class"]),
                f"{r['pl_acc']:.3f}", f"{r['rf_acc']:.3f}", f"{r['gt_acc']:.3f}",
                f"{r['within_cka']:.3f}", f"{r['cross_cka']:.3f}",
                f"{r['cross_arch_cka']:.3f}", f"{r['cross_jac']:.3f}",
                f"{r['null_p95']:.3f}"]
        for ci, (val, x) in enumerate(zip(vals, xs_tab)):
            ax.text(x, y, val, transform=ax.transAxes,
                    fontsize=8.5, color=cols_clr[ci])

    fig.suptitle("Subset Scaling Sweep — FlyWire FAFB Connectome\n"
                 "n=16→96 (8 classes) · all metrics vs dataset size",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    path = OUT_FIGS / "fig6_subset_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"\n[PASS] → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  09 — Subset scaling sweep (n=16→96)")
    print(f"{'='*60}\n")

    sweep_path = OUT_MET / "subset_sweep.json"
    if sweep_path.exists():
        print(f"Loading cached sweep results from {sweep_path}")
        with open(sweep_path) as f:
            results = json.load(f)
    else:
        samples, label_names = load_dataset(DATA_CSV)
        labels   = np.array([s.label_idx for s in samples])
        morph    = np.load(MORPH_NPY)
        diagrams = np.load(DIAGS_NPZ, allow_pickle=True)

        print("Pre-computing graph-topology embeddings...")
        gt_raw = np.stack([gt_emb_fn(s.path) for s in samples])

        print(f"Running sweep: {[c[0] for c in CONFIGS]} neurons...\n")
        results = run_sweep(samples, labels, morph, diagrams, gt_raw)

        with open(sweep_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[PASS] Saved → {sweep_path}")

    print("\nGenerating Fig 6 (subset sweep)...")
    plot_sweep(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Key trends:")
    print(f"  PL accuracy:  {results[0]['pl_acc']:.3f} (n=16) → {results[-1]['pl_acc']:.3f} (n=96)")
    print(f"  Null p95:     {results[0]['null_p95']:.3f} (n=16) → {results[-1]['null_p95']:.3f} (n=96)")
    print(f"  Cross-arch:   {results[0]['cross_arch_cka']:.3f} → {results[-1]['cross_arch_cka']:.3f} (stable)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rerun", action="store_true",
                   help="Rerun sweep even if cached results exist")
    args = p.parse_args()
    if args.rerun:
        (OUT_MET / "subset_sweep.json").unlink(missing_ok=True)
    main()
