"""Generate cross-architecture CKA heatmap (CorianderNet vs CNNet)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MET_DIR = Path("outputs/fafb/metrics")
FIG_DIR = Path("outputs/fafb/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0F1117"; CARD_BG = "#1A1D27"; WHITE = "#F0F0F0"
MUTED = "#9CA3AF"; AMBER = "#F59E0B"; TEAL = "#0B8FAC"

mat = np.load(MET_DIR / "cross_arch_cka_matrix.npy")   # (15, 15) cori × cnn
ca  = json.load(open(MET_DIR / "cross_arch_results.json"))
cka_labels = json.load(open(MET_DIR / "cka_labels.json"))
model_labels = cka_labels["labels"]   # CorianderNet model labels

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

# ── Left: cross-arch heatmap ──────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(DARK_BG)
im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(15)); ax.set_xticklabels(model_labels, rotation=45, ha="right",
                                               fontsize=6, color=MUTED)
ax.set_yticks(range(15)); ax.set_yticklabels(model_labels, fontsize=6, color=MUTED)
ax.set_xlabel("CNNet models", color=MUTED, fontsize=9)
ax.set_ylabel("CorianderNet models", color=MUTED, fontsize=9)
for i in range(15):
    for j in range(15):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                fontsize=5, color=WHITE if mat[i,j] < 0.6 else DARK_BG)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=MUTED, labelsize=7)
cb.set_label("CKA", color=MUTED, fontsize=8)
mean_cka = ca["cross_arch_cka_cori_vs_cnn"]["mean"]
std_cka  = ca["cross_arch_cka_cori_vs_cnn"]["std"]
mean_rsa = ca["cross_arch_rsa_cori_vs_cnn"]["mean"]
ax.set_title(f"Cross-Architecture CKA\nCorianderNet (PD) × CNNet (PI)\n"
             f"Mean={mean_cka:.3f}±{std_cka:.3f}  RSA={mean_rsa:.3f}",
             color=WHITE, fontsize=9)

# ── Right: summary bar comparison ────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(CARD_BG)
for sp in ax.spines.values(): sp.set_edgecolor("#252836")
ax.tick_params(colors=MUTED, labelsize=8)

metrics = [
    ("CorianderNet\nwithin-partition", 0.890, 0.132, TEAL),
    ("CorianderNet\ncross-partition",  0.899, 0.103, TEAL),
    ("CNNet\nwithin-partition",        ca["cnn_within_partition_cka"]["mean"],
                                       ca["cnn_within_partition_cka"]["std"], AMBER),
    ("CNNet\ncross-partition",         ca["cnn_cross_partition_cka"]["mean"],
                                       ca["cnn_cross_partition_cka"]["std"], AMBER),
    ("Cross-arch\n(Cori × CNN)",       mean_cka, std_cka, "#8B5CF6"),
]
labels_bar = [m[0] for m in metrics]
vals  = [m[1] for m in metrics]
errs  = [m[2] for m in metrics]
cols  = [m[3] for m in metrics]
xs = range(len(metrics))
ax.bar(xs, vals, color=cols, alpha=0.85, yerr=errs,
       error_kw={"ecolor": WHITE, "capsize": 4, "linewidth": 1.5})
ax.set_ylim(0, 1.05)
ax.set_xticks(xs); ax.set_xticklabels(labels_bar, fontsize=7.5, color=MUTED)
ax.set_ylabel("CKA (logit space)", color=MUTED, fontsize=9)
ax.axhline(0.5, color=WHITE, lw=0.8, ls="--", alpha=0.3)
for x, v, e in zip(xs, vals, errs):
    ax.text(x, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=7.5,
            color=WHITE, fontweight="bold")
ax.set_title("CKA Summary: Within vs Cross vs Cross-Arch", color=WHITE, fontsize=9)

fig.suptitle("PRH Cross-Architecture Test: CorianderNet (Persistence Diagrams) "
             "vs CNNet (Persistence Images)", color=WHITE, fontsize=10, y=1.01)
plt.tight_layout()
out = FIG_DIR / "fig_crossarch_cka.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"Saved → {out}")
