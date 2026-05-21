"""
11_hparam_sweep.py
==================
Hyperparameter sweep for CorianderNet training on the FAFB sample dataset.
Sweeps learning rate and batch size on a single partition/seed to diagnose
the training collapse observed at lr=5e-3 (12/15 models stuck at chance).

Reads:  outputs/fafb/data/{dataset.csv, partitions.json, diagrams.npz}
Saves:  outputs/fafb/metrics/hparam_sweep.json
        outputs/fafb/figures/fig_hparam_sweep.png

Run:
  python experiments/fafb_perslay/11_hparam_sweep.py
  python experiments/fafb_perslay/11_hparam_sweep.py --partition part_0 --seed 0
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset
from morphoclass.models import CorianderNet

DATA_CSV     = Path("outputs/fafb/data/dataset.csv")
PARTITIONS   = Path("outputs/fafb/data/partitions.json")
DIAGRAMS_NPZ = Path("outputs/fafb/data/diagrams.npz")
OUT_METRICS  = Path("outputs/fafb/metrics")
OUT_FIGURES  = Path("outputs/fafb/figures")
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

N_FEATURES = 32

# ── Sweep grid ────────────────────────────────────────────────────────────────
LR_VALUES         = [5e-4, 1e-3, 2e-3, 5e-3]
BATCH_SIZES       = [16, 32, 64]
EPOCHS_VALUES     = [200, 500]


def collate(data_list):
    return Batch.from_data_list(data_list, follow_batch=["diagram"])


def make_loader(data_list, indices, batch_size, shuffle=False):
    subset = [data_list[i] for i in indices]
    # Full-batch if batch_size >= subset size
    bs = min(batch_size, len(subset))
    return DataLoader(subset, batch_size=bs, shuffle=shuffle, collate_fn=collate)


def build_data_objects(diagrams, labels, scale):
    return [
        Data(
            diagram=torch.tensor(d / scale, dtype=torch.float32),
            y=torch.tensor(int(y), dtype=torch.long),
            num_nodes=len(d),
        )
        for d, y in zip(diagrams, labels)
    ]


def run_single(lr, batch_size, n_epochs, seed, splits, data_list, n_classes, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = splits["train"]
    val_idx   = splits["val"]
    test_idx  = splits["test"]

    model     = CorianderNet(n_classes=n_classes, n_features=N_FEATURES).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_loader = make_loader(data_list, train_idx, batch_size, shuffle=True)

    acc_history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()

        # Record train acc every 10 epochs for the learning curve
        if (epoch + 1) % 10 == 0:
            model.eval()
            preds, true = [], []
            with torch.no_grad():
                for batch in make_loader(data_list, train_idx, batch_size):
                    batch = batch.to(device)
                    preds.extend(model(batch).argmax(1).cpu().tolist())
                    true.extend(batch.y.cpu().tolist())
            acc_history.append(accuracy_score(true, preds))

    elapsed = time.time() - t0

    # Final evaluation
    def evaluate(idx):
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in make_loader(data_list, idx, batch_size):
                batch = batch.to(device)
                preds.extend(model(batch).argmax(1).cpu().tolist())
                true.extend(batch.y.cpu().tolist())
        return (accuracy_score(true, preds),
                f1_score(true, preds, average="macro", zero_division=0))

    val_acc,  val_f1  = evaluate(val_idx)
    test_acc, test_f1 = evaluate(test_idx)

    return {
        "lr": lr, "batch_size": batch_size, "n_epochs": n_epochs,
        "val_acc": val_acc, "test_acc": test_acc,
        "val_f1": val_f1, "test_f1": test_f1,
        "elapsed_s": elapsed,
        "train_acc_history": acc_history,
    }


def plot_sweep(results, n_classes, out_path):
    chance = 1.0 / n_classes
    epochs_list = sorted(set(r["n_epochs"] for r in results))

    fig, axes = plt.subplots(
        len(epochs_list), 2,
        figsize=(13, 5 * len(epochs_list)),
        squeeze=False,
    )
    fig.suptitle(
        "CorianderNet Hyperparameter Sweep\n"
        f"(part_0, seed=0, n_classes={n_classes}, chance={chance:.3f})",
        fontsize=13, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(LR_VALUES)))

    for row, n_epochs in enumerate(epochs_list):
        ax_bar  = axes[row, 0]
        ax_curve = axes[row, 1]

        subset = [r for r in results if r["n_epochs"] == n_epochs]
        subset.sort(key=lambda r: (r["batch_size"], r["lr"]))

        # ── Bar chart: test_acc by (batch_size, lr) ───────────────────────────
        labels_bar, test_accs, val_accs = [], [], []
        for r in subset:
            labels_bar.append(f"bs={r['batch_size']}\nlr={r['lr']:.0e}")
            test_accs.append(r["test_acc"])
            val_accs.append(r["val_acc"])

        x = np.arange(len(labels_bar))
        w = 0.35
        ax_bar.bar(x - w/2, test_accs, w, label="Test acc", color="#4C72B0")
        ax_bar.bar(x + w/2, val_accs,  w, label="Val acc",  color="#DD8452", alpha=0.8)
        ax_bar.axhline(chance, color="red", ls="--", lw=1.2, label=f"Chance ({chance:.3f})")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels_bar, fontsize=8)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_title(f"n_epochs={n_epochs} — test & val accuracy")
        ax_bar.legend(fontsize=8)

        # ── Learning curves: train acc over epochs for each lr (best bs) ──────
        best_bs = max(BATCH_SIZES)  # show curves for largest batch size
        for i, lr in enumerate(LR_VALUES):
            match = [r for r in subset if r["lr"] == lr and r["batch_size"] == best_bs]
            if not match:
                continue
            r = match[0]
            epochs_x = [(e + 1) * 10 for e in range(len(r["train_acc_history"]))]
            ax_curve.plot(epochs_x, r["train_acc_history"],
                          color=colors[i], label=f"lr={lr:.0e}", lw=1.8)
        ax_curve.axhline(chance, color="red", ls="--", lw=1.2, label="Chance")
        ax_curve.set_xlabel("Epoch")
        ax_curve.set_ylabel("Train accuracy")
        ax_curve.set_title(f"n_epochs={n_epochs} — learning curves (bs={best_bs})")
        ax_curve.legend(fontsize=8)
        ax_curve.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Figure → {out_path}")


def print_table(results, n_classes):
    chance = 1.0 / n_classes
    print(f"\n{'='*72}")
    print(f"  Hyperparameter Sweep Results  (chance={chance:.3f})")
    print(f"{'='*72}")
    print(f"{'LR':>8}  {'BS':>4}  {'Epochs':>6}  {'Val':>6}  {'Test':>6}  {'F1':>6}  {'Time':>6}")
    print(f"{'-'*72}")

    sorted_r = sorted(results, key=lambda r: -r["test_acc"])
    for r in sorted_r:
        marker = " ★" if r["test_acc"] > chance * 1.5 else ""
        print(f"{r['lr']:>8.0e}  {r['batch_size']:>4d}  {r['n_epochs']:>6d}  "
              f"{r['val_acc']:>6.3f}  {r['test_acc']:>6.3f}  "
              f"{r['test_f1']:>6.3f}  {r['elapsed_s']:>5.1f}s{marker}")
    print(f"{'='*72}")

    best = sorted_r[0]
    print(f"\n  Best: lr={best['lr']:.0e}  batch_size={best['batch_size']}  "
          f"n_epochs={best['n_epochs']}  → test_acc={best['test_acc']:.3f}")


def main(partition="part_0", seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CorianderNet hyperparameter sweep")
    print(f"  Partition: {partition}  Seed: {seed}  Device: {device}")
    print(f"  Grid: {len(LR_VALUES)} LRs × {len(BATCH_SIZES)} batch sizes "
          f"× {len(EPOCHS_VALUES)} epoch counts = "
          f"{len(LR_VALUES)*len(BATCH_SIZES)*len(EPOCHS_VALUES)} runs")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    n_classes = len(label_names)

    raw = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = [raw[s.path.stem] if s.path.stem in raw
                else np.array([[0., 1.]], dtype=np.float32)
                for s in samples]

    all_pts = np.vstack(diagrams)
    scale = np.array([[
        max(abs(all_pts[:, 0].max()), abs(all_pts[:, 0].min())),
        max(abs(all_pts[:, 1].max()), abs(all_pts[:, 1].min())),
    ]])

    labels    = np.array([s.label_idx for s in samples])
    data_list = build_data_objects(diagrams, labels, scale)

    with open(PARTITIONS) as f:
        partitions = json.load(f)
    splits = partitions[partition]

    results = []
    total = len(LR_VALUES) * len(BATCH_SIZES) * len(EPOCHS_VALUES)
    i = 0
    for n_epochs, batch_size, lr in product(EPOCHS_VALUES, BATCH_SIZES, LR_VALUES):
        i += 1
        print(f"[{i:2d}/{total}] lr={lr:.0e}  batch_size={batch_size:3d}  "
              f"n_epochs={n_epochs}", end="  ", flush=True)
        r = run_single(lr, batch_size, n_epochs, seed, splits,
                       data_list, n_classes, device)
        results.append(r)
        print(f"test_acc={r['test_acc']:.3f}  val_acc={r['val_acc']:.3f}  "
              f"({r['elapsed_s']:.1f}s)")

    # Save JSON (exclude history for brevity in summary)
    out_json = OUT_METRICS / "hparam_sweep.json"
    summary  = [{k: v for k, v in r.items() if k != "train_acc_history"}
                for r in results]
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PASS] Metrics → {out_json}")

    print_table(results, n_classes)

    out_fig = OUT_FIGURES / "fig_hparam_sweep.png"
    plot_sweep(results, n_classes, out_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="part_0")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()
    main(args.partition, args.seed)
