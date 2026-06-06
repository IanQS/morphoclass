"""
12_diagnostics.py
=================
Three diagnostic experiments on already-computed embeddings — NO retraining,
NO NBLAST. Everything reads from outputs/fafb/.

Experiment 1 — Residualize PersLay against morphometrics
  Q: How much of the PersLay representation (and its cross-partition PRH
     convergence) is explained by 8 simple morphometric features?

Experiment 2 — CKA vs accuracy scatter
  Q: Do higher-accuracy models converge more (higher cross-arch CKA)?
     Also checks whether P1S4 is a within-PersLay outlier.

Experiment 3 — Random-init CKA baseline (architecture-matched)
  Q: How much CKA is "free" from architecture alone vs learned from data?
     learned fraction = (trained_CKA - random_CKA) / (1 - random_CKA)
     Two like-for-like baselines:
       within-partition  → random CorianderNet ↔ random CorianderNet
       cross-arch        → random CorianderNet ↔ random CNNet

Outputs:
  outputs/fafb/metrics/residualization_results.json
  outputs/fafb/metrics/cka_accuracy_correlation.json
  outputs/fafb/metrics/random_init_baseline.json
  outputs/fafb/figures/fig_diagnostics.png

Run (use sbatch — full-N CKA is memory-heavy):
  sbatch diagnostics.slurm
"""

import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import debiased_cka, permutation_cka_null, load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_DIR  = Path("outputs/fafb/embeddings")
DATA_DIR = Path("outputs/fafb/data")
MODEL_DIR = Path("outputs/fafb/models")
MET_DIR  = Path("outputs/fafb/metrics")
FIG_DIR  = Path("outputs/fafb/figures")
MET_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 200  # permutations for null estimates

# ── Dark theme (matches all existing figures) ──────────────────────────────────
DARK = "#0A1628"; CARD = "#0F2040"; CARD2 = "#152B55"; BORDER = "#1A3050"
TEAL = "#0AAFCC"; AMBER = "#F5A623"; CORAL = "#EF6351"; GREEN = "#22C88A"
PURPLE = "#9B7FE8"; WHITE = "#EEF4FA"; MUTED = "#5A7A9A"; SLATE = "#8BA5BE"
PART_COLORS = [TEAL, AMBER, PURPLE]  # one per partition

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": SLATE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": CARD2, "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


# ── Loaders ─────────────────────────────────────────────────────────────────────

def load_logit_embeddings(prefix=""):
    """Return list of (part, seed, array) for the 15 logit embeds.
    prefix='' → PersLay (CorianderNet), prefix='cnn_' → CNNet."""
    out = []
    for part in range(3):
        for seed in range(5):
            path = EMB_DIR / f"{prefix}logit_part_{part}_s{seed}.npy"
            out.append((part, seed, np.load(path)))
    return out


def residualize(emb, covariates):
    """Remove linear effect of covariates from each embedding dimension.
    Fits one LinearRegression on all N neurons, returns emb - prediction."""
    reg = LinearRegression().fit(covariates, emb)
    return emb - reg.predict(covariates)


def within_cross_cka(embeddings):
    """Split all C(15,2)=105 pairs into within-partition (30) and cross (75)."""
    within, cross = [], []
    for i, (pi, si, ei) in enumerate(embeddings):
        for j, (pj, sj, ej) in enumerate(embeddings):
            if j <= i:
                continue
            cka = debiased_cka(ei, ej)
            (within if pi == pj else cross).append(cka)
    return np.array(within), np.array(cross)


# ── Sanity checks ───────────────────────────────────────────────────────────────

def sanity_checks():
    print("\n" + "=" * 60)
    print("  Sanity checks")
    print("=" * 60)
    pl  = list(EMB_DIR.glob("logit_part_*.npy"))
    cnn = list(EMB_DIR.glob("cnn_logit_part_*.npy"))
    morph = np.load(DATA_DIR / "morphometrics.npy")
    n_logit = np.load(EMB_DIR / "logit_part_0_s0.npy").shape[0]
    ca = json.loads((MET_DIR / "cross_arch_results.json").read_text())
    ca_mean = ca["cross_arch_cka_cori_vs_cnn"]["mean"]

    ok = True
    def check(name, cond, detail):
        nonlocal ok
        tag = PASS if cond else FAIL
        ok = ok and cond
        print(f"  {tag} {name}: {detail}")

    check("len(logit files) == 15", len(pl) == 15, f"{len(pl)}")
    check("len(cnn_logit files) == 15", len(cnn) == 15, f"{len(cnn)}")
    check("morphometrics (N,8), N matches logit", morph.shape == (n_logit, 8),
          f"morph={morph.shape}, logit N={n_logit}")
    check("cross_arch CKA mean == 0.406", round(ca_mean, 3) == 0.406,
          f"{ca_mean}")
    if not ok:
        print(f"  {FAIL} sanity checks failed — aborting")
        sys.exit(1)
    print(f"  {PASS} all sanity checks pass (N={n_logit})")
    return n_logit


# ── Experiment 1: Residualize PersLay against morphometrics ──────────────────

def experiment1():
    print("\n" + "=" * 60)
    print("  Exp 1 — Residualize PersLay against morphometrics")
    print("=" * 60)

    morph = np.load(DATA_DIR / "morphometrics.npy")            # (N, 8)
    morph_std = (morph - morph.mean(0)) / (morph.std(0) + 1e-9)

    embeddings = load_logit_embeddings()

    # CKA(raw logit, morph) — sanity ~0.915 (mean over 15)
    cka_raw = [debiased_cka(emb, morph_std) for _, _, emb in embeddings]
    print(f"  CKA(raw logit, morph):         {np.mean(cka_raw):.4f} ± "
          f"{np.std(cka_raw):.4f}   [sanity ~0.915]")

    # Residualize each logit against morphometrics (fit on all N)
    residuals = [(p, s, residualize(emb, morph_std)) for p, s, emb in embeddings]

    cka_res_morph = [debiased_cka(emb, morph_std) for _, _, emb in residuals]
    print(f"  CKA(residualized, morph):      {np.mean(cka_res_morph):.4f} ± "
          f"{np.std(cka_res_morph):.4f}   [should drop ~0]")

    within_raw, cross_raw = within_cross_cka(embeddings)
    within_res, cross_res = within_cross_cka(residuals)
    print(f"  within raw  ({len(within_raw)} pairs): {np.mean(within_raw):.4f} ± {np.std(within_raw):.4f}")
    print(f"  within res  ({len(within_res)} pairs): {np.mean(within_res):.4f} ± {np.std(within_res):.4f}")
    print(f"  cross  raw  ({len(cross_raw)} pairs): {np.mean(cross_raw):.4f} ± {np.std(cross_raw):.4f}")
    print(f"  cross  res  ({len(cross_res)} pairs): {np.mean(cross_res):.4f} ± {np.std(cross_res):.4f}")

    # PRH survival: residualized cross-partition CKA vs permutation null p95.
    # Null built on a representative residualized cross-partition pair.
    a = residuals[0][2]   # part 0, seed 0 (residualized)
    b = residuals[5][2]   # part 1, seed 0 (residualized) — cross partition
    print(f"  Building permutation null ({N_PERM} perms) on a residualized cross pair...")
    null = permutation_cka_null(a, b, n_permutations=N_PERM, seed=0)
    null_p95 = float(np.percentile(null, 95))
    prh_survives = bool(np.mean(cross_res) > null_p95)
    print(f"  null p95 = {null_p95:.5f}   cross_res mean = {np.mean(cross_res):.4f}")
    print(f"  PRH cross-partition signal survives residualization: {prh_survives}")

    results = {
        "n_neurons": int(morph.shape[0]),
        "n_perm": N_PERM,
        "cka_raw_vs_morph_mean":   round(float(np.mean(cka_raw)), 4),
        "cka_raw_vs_morph_std":    round(float(np.std(cka_raw)), 4),
        "cka_resid_vs_morph_mean": round(float(np.mean(cka_res_morph)), 4),
        "cka_resid_vs_morph_std":  round(float(np.std(cka_res_morph)), 4),
        "within_partition_raw_mean":   round(float(np.mean(within_raw)), 4),
        "within_partition_raw_std":    round(float(np.std(within_raw)), 4),
        "within_partition_resid_mean": round(float(np.mean(within_res)), 4),
        "within_partition_resid_std":  round(float(np.std(within_res)), 4),
        "cross_partition_raw_mean":   round(float(np.mean(cross_raw)), 4),
        "cross_partition_raw_std":    round(float(np.std(cross_raw)), 4),
        "cross_partition_resid_mean": round(float(np.mean(cross_res)), 4),
        "cross_partition_resid_std":  round(float(np.std(cross_res)), 4),
        "perm_null_p95": round(null_p95, 5),
        "prh_signal_survives_residualization": prh_survives,
        "interpretation": (
            f"Morphometrics explain a large share of CKA(PersLay, morph) "
            f"({np.mean(cka_raw):.3f} → {np.mean(cka_res_morph):.3f} after "
            f"residualization). Cross-partition PRH convergence "
            + ("SURVIVES" if prh_survives else "DISAPPEARS")
            + f" (residual cross CKA {np.mean(cross_res):.3f} "
            + (">" if prh_survives else "<=")
            + f" null p95 {null_p95:.4f})."
        ),
    }
    out = MET_DIR / "residualization_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  {PASS} Saved → {out}")

    bundle = dict(
        cka_raw=float(np.mean(cka_raw)),
        cka_res_morph=float(np.mean(cka_res_morph)),
        within_raw=within_raw, within_res=within_res,
        cross_raw=cross_raw, cross_res=cross_res,
        null_p95=null_p95,
    )
    return results, bundle


# ── Experiment 2: CKA vs accuracy scatter ─────────────────────────────────────

def experiment2():
    print("\n" + "=" * 60)
    print("  Exp 2 — CKA vs accuracy scatter (PersLay × CNNet)")
    print("=" * 60)

    pl  = load_logit_embeddings()            # 15 PersLay
    cnn = load_logit_embeddings("cnn_")      # 15 CNNet

    pl_df  = pd.read_csv(MODEL_DIR / "run_log.csv")
    cnn_df = pd.read_csv(MODEL_DIR / "cnn_run_log.csv")

    def key(r): return (int(r["partition"].replace("part_", "")), int(r["seed"]))
    pl_acc  = {key(r): r["test_acc"] for _, r in pl_df.iterrows()}
    cnn_acc = {key(r): r["test_acc"] for _, r in cnn_df.iterrows()}

    # 15×15 cross-arch CKA matrix (rows=PersLay, cols=CNNet)
    print("  Computing 225 cross-arch CKA pairs...")
    X = np.zeros((15, 15))
    for i, (pi, si, ei) in enumerate(pl):
        for j, (pj, sj, ej) in enumerate(cnn):
            X[i, j] = debiased_cka(ei, ej)

    pl_mean  = X.mean(axis=1)   # mean cross-arch CKA per PersLay model
    cnn_mean = X.mean(axis=0)   # mean cross-arch CKA per CNNet model
    pl_accs  = np.array([pl_acc[(pi, si)]  for pi, si, _ in pl])
    cnn_accs = np.array([cnn_acc[(pj, sj)] for pj, sj, _ in cnn])
    pl_labels = [f"P{pi}S{si}" for pi, si, _ in pl]
    pl_parts  = [pi for pi, si, _ in pl]

    r_pl,  p_pl  = pearsonr(pl_accs,  pl_mean)
    r_cnn, p_cnn = pearsonr(cnn_accs, cnn_mean)
    print(f"  Pearson r (PersLay acc vs mean cross-arch CKA): r={r_pl:.4f}  p={p_pl:.4f}")
    print(f"  Pearson r (CNNet   acc vs mean cross-arch CKA): r={r_cnn:.4f}  p={p_cnn:.4f}")
    print(f"  cross-arch CKA overall mean = {X.mean():.4f} ± {X.std():.4f}")

    # P1S4 within-PersLay outlier check.
    # Build 15×15 within-PersLay CKA matrix, mean CKA-with-others per model.
    print("  Computing within-PersLay CKA matrix for P1S4 check...")
    W = np.zeros((15, 15))
    for i in range(15):
        for j in range(i + 1, 15):
            W[i, j] = W[j, i] = debiased_cka(pl[i][2], pl[j][2])
    mean_with_others = np.array(
        [W[i, [k for k in range(15) if k != i]].mean() for i in range(15)]
    )
    p1s4_idx = next(i for i, (pi, si, _) in enumerate(pl) if pi == 1 and si == 4)
    others_idx = [i for i in range(15) if i != p1s4_idx]
    p1s4_val = float(mean_with_others[p1s4_idx])
    others_mean = float(mean_with_others[others_idx].mean())
    others_std  = float(mean_with_others[others_idx].std())
    z = (p1s4_val - others_mean) / (others_std + 1e-12)
    is_outlier = bool(abs(z) > 2.0)
    print(f"  P1S4 mean CKA with other 14 PersLay: {p1s4_val:.4f}")
    print(f"  Remaining 14 models mean: {others_mean:.4f} ± {others_std:.4f}  (z={z:+.2f})")
    print(f"  P1S4 is a >2σ outlier: {is_outlier}")

    scatter_data = [
        {"label": pl_labels[i], "partition": int(pl_parts[i]),
         "pl_test_acc": round(float(pl_accs[i]), 4),
         "mean_cross_arch_cka": round(float(pl_mean[i]), 4)}
        for i in range(15)
    ]

    results = {
        "pearson_r_pl_acc_vs_cka":  round(float(r_pl), 4),
        "pearson_p_pl_acc_vs_cka":  round(float(p_pl), 6),
        "pearson_r_cnn_acc_vs_cka": round(float(r_cnn), 4),
        "pearson_p_cnn_acc_vs_cka": round(float(p_cnn), 6),
        "cross_arch_cka_mean": round(float(X.mean()), 4),
        "cross_arch_cka_std":  round(float(X.std()), 4),
        "p1s4_check": {
            "p1s4_mean_cka_with_other_perslay": round(p1s4_val, 4),
            "remaining_14_mean": round(others_mean, 4),
            "remaining_14_std":  round(others_std, 4),
            "z_score": round(z, 3),
            "is_outlier_gt_2std": is_outlier,
        },
        "scatter_data": scatter_data,
        "cnn_scatter_data": [
            {"partition": int(pj), "seed": int(sj),
             "cnn_test_acc": round(float(cnn_accs[k]), 4),
             "mean_cross_arch_cka": round(float(cnn_mean[k]), 4)}
            for k, (pj, sj, _) in enumerate(cnn)
        ],
        "interpretation": (
            f"PersLay accuracy vs cross-arch CKA: r={r_pl:.3f} (p={p_pl:.4f}); "
            f"CNNet: r={r_cnn:.3f} (p={p_cnn:.4f}). "
            + ("Higher-accuracy models converge more."
               if (p_pl < 0.05 and r_pl > 0) else
               "No significant accuracy↔convergence link.")
            + f" P1S4 {'IS' if is_outlier else 'is NOT'} a within-PersLay outlier (z={z:+.2f})."
        ),
    }
    out = MET_DIR / "cka_accuracy_correlation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  {PASS} Saved → {out}")

    bundle = dict(pl_accs=pl_accs, pl_mean=pl_mean, pl_labels=pl_labels,
                  pl_parts=pl_parts, r_pl=r_pl, p_pl=p_pl,
                  p1s4_idx=p1s4_idx)
    return results, bundle


# ── Experiment 3: Random-init CKA baseline ────────────────────────────────────

def _build_coriander_data_objects(diagrams, labels, scale, torch, Data):
    return [
        Data(diagram=torch.tensor(d / scale, dtype=torch.float32),
             y=torch.tensor(int(y), dtype=torch.long), num_nodes=len(d))
        for d, y in zip(diagrams, labels)
    ]


def _rasterize(diag, bins, max_val):
    """Persistence-image rasterization — mirrors 14_train_cnn.rasterize."""
    img = np.zeros((bins, bins), dtype=np.float32)
    if len(diag) == 0:
        return img
    b = np.clip(diag[:, 0] / max_val, 0, 1 - 1e-6)
    d = np.clip(diag[:, 1] / max_val, 0, 1 - 1e-6)
    w = d - b
    bi = (b * bins).astype(int)
    di = (d * bins).astype(int)
    np.add.at(img, (di, bi), w)
    return img


def _build_cnn_data_objects(diagrams, labels, max_val, bins, torch, Data):
    out = []
    for diag, y in zip(diagrams, labels):
        img = _rasterize(diag, bins=bins, max_val=max_val)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out.append(Data(image=img_t, y=torch.tensor(int(y), dtype=torch.long)))
    return out


def _forward_logits(model, loader, torch):
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out.append(model(batch).cpu().numpy())
    return np.vstack(out)


def experiment3(trained_within_mean, cross_arch_mean):
    print("\n" + "=" * 60)
    print("  Exp 3 — Random-init CKA baseline (architecture-matched)")
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch, Data
    sys.path.insert(0, ".")
    from morphoclass.models import CorianderNet, CNNet

    IMAGE_SIZE = 32   # must match 14_train_cnn.IMAGE_SIZE

    samples, label_names = load_dataset(DATA_DIR / "dataset.csv")
    labels = np.array([s.label_idx for s in samples])
    n_classes = len(label_names)

    raw = np.load(DATA_DIR / "diagrams.npz", allow_pickle=True)
    diagrams = [raw[s.path.stem] if s.path.stem in raw
                else np.array([[0., 1.]], dtype=np.float32) for s in samples]
    all_pts = np.vstack(diagrams)

    # ── CorianderNet (PersLay) random-init logits ────────────────────────────
    scale = np.array([[
        max(abs(all_pts[:, 0].max()), abs(all_pts[:, 0].min())),
        max(abs(all_pts[:, 1].max()), abs(all_pts[:, 1].min())),
    ]])
    cori_data = _build_coriander_data_objects(diagrams, labels, scale, torch, Data)

    def cori_collate(dl): return Batch.from_data_list(dl, follow_batch=["diagram"])
    cori_loader = DataLoader(cori_data, batch_size=256, shuffle=False,
                             collate_fn=cori_collate)

    n_models = 3
    cori_logits = []
    for rs in range(n_models):
        torch.manual_seed(rs)
        model = CorianderNet(n_classes=n_classes, n_features=32)
        L = _forward_logits(model, cori_loader, torch)
        cori_logits.append(L)
        print(f"  random CorianderNet {rs}: logit {L.shape}  mean={L.mean():.4f}")

    # ── CNNet (persistence-image CNN) random-init logits ─────────────────────
    cnn_max_val = float(max(abs(all_pts[:, 0].max()), abs(all_pts[:, 1].max())))
    cnn_data = _build_cnn_data_objects(diagrams, labels, cnn_max_val, IMAGE_SIZE,
                                       torch, Data)

    def cnn_collate(dl): return Batch.from_data_list(dl)
    cnn_loader = DataLoader(cnn_data, batch_size=256, shuffle=False,
                            collate_fn=cnn_collate)

    cnn_logits = []
    for rs in range(n_models):
        torch.manual_seed(100 + rs)   # distinct seeds from CorianderNet
        model = CNNet(n_classes=n_classes, image_size=IMAGE_SIZE, bn=True)
        L = _forward_logits(model, cnn_loader, torch)
        cnn_logits.append(L)
        print(f"  random CNNet {rs}: logit {L.shape}  mean={L.mean():.4f}")

    # ── within-arch random baseline: CorianderNet ↔ CorianderNet (3 pairs) ────
    within_pairs = [debiased_cka(cori_logits[i], cori_logits[j])
                    for i in range(n_models) for j in range(i + 1, n_models)]
    random_within_mean = float(np.mean(within_pairs))
    random_within_std  = float(np.std(within_pairs))
    print(f"  Random within-arch CKA  (CorianderNet↔CorianderNet, "
          f"{len(within_pairs)} pairs): {random_within_mean:.4f} ± {random_within_std:.4f}")

    # ── cross-arch random baseline: CorianderNet ↔ CNNet (9 pairs) ────────────
    cross_pairs = [debiased_cka(cori_logits[i], cnn_logits[j])
                   for i in range(n_models) for j in range(n_models)]
    random_cross_mean = float(np.mean(cross_pairs))
    random_cross_std  = float(np.std(cross_pairs))
    print(f"  Random cross-arch CKA   (CorianderNet↔CNNet, "
          f"{len(cross_pairs)} pairs): {random_cross_mean:.4f} ± {random_cross_std:.4f}")

    def learned_fraction(trained, baseline):
        return (trained - baseline) / (1.0 - baseline)

    lf_within = learned_fraction(trained_within_mean, random_within_mean)
    lf_cross  = learned_fraction(cross_arch_mean,     random_cross_mean)
    print(f"  trained within-partition CKA = {trained_within_mean:.4f} "
          f"(baseline {random_within_mean:.4f}) → learned fraction = {lf_within:.4f}")
    print(f"  trained cross-arch CKA       = {cross_arch_mean:.4f} "
          f"(baseline {random_cross_mean:.4f}) → learned fraction = {lf_cross:.4f}")

    results = {
        "n_random_models": n_models,
        "within_arch_baseline": {
            "comparison": "random CorianderNet vs random CorianderNet",
            "cka_pairs": [round(float(c), 4) for c in within_pairs],
            "cka_mean":  round(random_within_mean, 4),
            "cka_std":   round(random_within_std, 4),
        },
        "cross_arch_baseline": {
            "comparison": "random CorianderNet vs random CNNet",
            "cka_pairs": [round(float(c), 4) for c in cross_pairs],
            "cka_mean":  round(random_cross_mean, 4),
            "cka_std":   round(random_cross_std, 4),
        },
        "trained_within_partition_cka": round(float(trained_within_mean), 4),
        "trained_cross_arch_cka":       round(float(cross_arch_mean), 4),
        "learned_fraction_within_partition": round(float(lf_within), 4),
        "learned_fraction_cross_arch":       round(float(lf_cross), 4),
        "interpretation": (
            f"Architecture-matched random-init baselines: CorianderNet↔CorianderNet "
            f"share CKA={random_within_mean:.3f}, CorianderNet↔CNNet share "
            f"CKA={random_cross_mean:.3f} from architecture/topology alone. "
            f"Of the trained within-partition CKA ({trained_within_mean:.3f}), "
            f"{lf_within*100:.1f}% is learned beyond the within-arch baseline; "
            f"of the cross-arch CKA ({cross_arch_mean:.3f}), {lf_cross*100:.1f}% "
            f"is learned beyond the like-for-like cross-arch baseline."
        ),
    }
    out = MET_DIR / "random_init_baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  {PASS} Saved → {out}")

    return results, dict(
        random_within_mean=random_within_mean, random_within_std=random_within_std,
        within_pairs=within_pairs,
        random_cross_mean=random_cross_mean, random_cross_std=random_cross_std,
        cross_pairs=cross_pairs,
        lf_within=float(lf_within), lf_cross=float(lf_cross),
    )


# ── Figure ──────────────────────────────────────────────────────────────────────

def make_figure(b1, b2, b3, res1, res2):
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.6), facecolor=DARK)
    fig.suptitle("Diagnostic Experiments — PRH Morphology (n=17,288, 20 classes)",
                 color=WHITE, fontsize=13, y=1.03)

    # ── Panel A: morphometric residualization (horizontal bars) ───────────────
    ax = axes[0]; ax.set_facecolor(CARD)
    labels = ["raw logit\nvs morph", "resid logit\nvs morph",
              "within-part\n(raw)", "within-part\n(resid)",
              "cross-part\n(raw)", "cross-part\n(resid)"]
    vals = [b1["cka_raw"], b1["cka_res_morph"],
            float(np.mean(b1["within_raw"])), float(np.mean(b1["within_res"])),
            float(np.mean(b1["cross_raw"])),  float(np.mean(b1["cross_res"]))]
    colors = [AMBER, TEAL, AMBER, TEAL, AMBER, TEAL]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CKA", color=SLATE)
    ax.axvline(b1["null_p95"], color=MUTED, linestyle="--", linewidth=0.9,
               label=f"null p95={b1['null_p95']:.3f}")
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", ha="left",
                color=WHITE, fontsize=8, fontweight="bold")
    ax.set_title("A  Morphometric residualization\n(amber=raw, teal=residualized)",
                 color=WHITE, fontsize=10)
    ax.legend(fontsize=7, labelcolor=MUTED, framealpha=0, loc="lower right")
    ax.grid(axis="x", alpha=0.3); ax.tick_params(colors=MUTED)

    # ── Panel B: CKA vs accuracy scatter (colour by partition) ────────────────
    ax = axes[1]; ax.set_facecolor(CARD)
    accs = b2["pl_accs"]; ckas = b2["pl_mean"]; parts = b2["pl_parts"]
    for p in range(3):
        m = np.array(parts) == p
        ax.scatter(accs[m], ckas[m], color=PART_COLORS[p], s=70, zorder=4,
                   edgecolor=DARK, linewidth=0.6, label=f"partition {p}")
    # trendline
    mm, bb = np.polyfit(accs, ckas, 1)
    xs = np.linspace(accs.min(), accs.max(), 50)
    ax.plot(xs, mm * xs + bb, color=WHITE, linewidth=1.3, linestyle="--", alpha=0.6)
    # label P1S4
    i = b2["p1s4_idx"]
    ax.scatter([accs[i]], [ckas[i]], facecolor="none", edgecolor=CORAL,
               s=200, linewidth=1.8, zorder=5)
    ax.annotate("P1S4", (accs[i], ckas[i]), xytext=(7, 6),
                textcoords="offset points", color=CORAL, fontsize=9, fontweight="bold")
    ax.set_xlabel("PersLay test accuracy", color=SLATE)
    ax.set_ylabel("Mean cross-arch CKA (vs CNNet)", color=SLATE)
    ax.set_title(f"B  Accuracy vs convergence\nr={b2['r_pl']:.3f}  p={b2['p_pl']:.4f}",
                 color=WHITE, fontsize=10)
    ax.legend(fontsize=7, labelcolor=WHITE, framealpha=0, loc="best")
    ax.grid(alpha=0.3); ax.tick_params(colors=MUTED)

    # ── Panel C: trained vs architecture-matched random-init (strip plot) ─────
    ax = axes[2]; ax.set_facecolor(CARD)
    trained_w = np.asarray(b1["within_raw"])
    rand_w = np.asarray(b3["within_pairs"])
    rand_x = np.asarray(b3["cross_pairs"])
    rng = np.random.default_rng(0)
    # x=0 trained within-part; x=1 random within-arch; x=2 random cross-arch
    ax.scatter(rng.normal(0, 0.05, len(trained_w)), trained_w, color=TEAL, s=42,
               alpha=0.85, zorder=4, label=f"trained within-part (n={len(trained_w)})")
    ax.scatter(rng.normal(1, 0.04, len(rand_w)), rand_w, color=MUTED, s=70,
               marker="D", zorder=4,
               label=f"random within-arch (n={len(rand_w)})")
    ax.scatter(rng.normal(2, 0.04, len(rand_x)), rand_x, color=PURPLE, s=55,
               marker="D", alpha=0.85, zorder=4,
               label=f"random cross-arch (n={len(rand_x)})")
    # mean bars
    ax.plot([-0.18, 0.18], [trained_w.mean()] * 2, color=WHITE, linewidth=2, zorder=5)
    ax.plot([0.82, 1.18], [rand_w.mean()] * 2, color=WHITE, linewidth=2, zorder=5)
    ax.plot([1.82, 2.18], [rand_x.mean()] * 2, color=WHITE, linewidth=2, zorder=5)
    ax.axhline(b1["null_p95"], color=CORAL, linestyle="--", linewidth=1.0,
               label=f"perm null p95={b1['null_p95']:.3f}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Trained\nwithin", "Random\nwithin-arch", "Random\ncross-arch"],
                       fontsize=8)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("CKA (logit space)", color=SLATE)
    ax.set_title("C  CKA vs architecture-matched\nrandom-init baselines",
                 color=WHITE, fontsize=10)
    ax.legend(fontsize=6.5, labelcolor=WHITE, framealpha=0, loc="center right")
    ax.grid(axis="y", alpha=0.3); ax.tick_params(colors=MUTED)

    # ── Panel D: summary table of key CKA values before/after residualization ──
    ax = axes[3]; ax.set_facecolor(CARD); ax.axis("off")
    rows = ["CKA(logit, morph)", "within-partition", "cross-partition",
            "perm null p95", "PRH survives?"]
    before = [f"{b1['cka_raw']:.3f}",
              f"{np.mean(b1['within_raw']):.3f}",
              f"{np.mean(b1['cross_raw']):.3f}",
              "—", "—"]
    after = [f"{b1['cka_res_morph']:.3f}",
             f"{np.mean(b1['within_res']):.3f}",
             f"{np.mean(b1['cross_res']):.3f}",
             f"{b1['null_p95']:.4f}",
             "YES" if res1["prh_signal_survives_residualization"] else "NO"]
    cell_text = [[before[i], after[i]] for i in range(len(rows))]
    tbl = ax.table(cellText=cell_text, rowLabels=rows,
                   colLabels=["Before", "After"],
                   cellLoc="center", rowLoc="left",
                   bbox=[0.30, 0.18, 0.66, 0.62])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:                      # header row
            cell.set_facecolor(CARD2); cell.set_text_props(color=WHITE, fontweight="bold")
        elif c == -1:                   # row labels
            cell.set_facecolor(CARD); cell.set_text_props(color=SLATE)
        else:
            cell.set_facecolor(DARK); cell.set_text_props(color=WHITE)
    ax.set_title("D  Key CKA values\nbefore vs after residualization",
                 color=WHITE, fontsize=10, y=0.92)
    ax.text(0.30, 0.10,
            f"within-arch floor = {b3['random_within_mean']:.3f}   "
            f"learned frac = {b3['lf_within']:.3f}",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.text(0.30, 0.02,
            f"cross-arch floor = {b3['random_cross_mean']:.3f}   "
            f"learned frac = {b3['lf_cross']:.3f}",
            transform=ax.transAxes, color=MUTED, fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig_diagnostics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"\n  {PASS} Figure → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  12 — Diagnostic experiments (no retraining, no NBLAST)")
    print("=" * 60)

    sanity_checks()

    res1, b1 = experiment1()
    res2, b2 = experiment2()

    trained_within = b1["within_raw"].mean()          # computed within-partition CKA
    cross_arch = res2["cross_arch_cka_mean"]           # computed cross-arch CKA
    res3, b3 = experiment3(trained_within, cross_arch)

    make_figure(b1, b2, b3, res1, res2)

    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Exp1  CKA(PersLay,morph): {b1['cka_raw']:.3f} → {b1['cka_res_morph']:.3f}; "
          f"cross-part resid {np.mean(b1['cross_res']):.3f} "
          f"(null p95 {b1['null_p95']:.4f}, PRH survives="
          f"{res1['prh_signal_survives_residualization']})")
    print(f"  Exp2  r(acc,cross-arch CKA)={res2['pearson_r_pl_acc_vs_cka']:.3f} "
          f"p={res2['pearson_p_pl_acc_vs_cka']:.4f}; "
          f"P1S4 outlier={res2['p1s4_check']['is_outlier_gt_2std']}")
    print(f"  Exp3  random floor: within-arch={res3['within_arch_baseline']['cka_mean']:.3f} "
          f"cross-arch={res3['cross_arch_baseline']['cka_mean']:.3f}; "
          f"learned fraction within={res3['learned_fraction_within_partition']:.3f} "
          f"cross-arch={res3['learned_fraction_cross_arch']:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
