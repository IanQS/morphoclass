"""
03_train_perslay.py
===================
Trains CorianderNet (morphoclass PersLay) across all partitions × seeds.
Reads:  outputs/fafb/data/{dataset.csv, partitions.json, diagrams.npz}
Saves:  outputs/fafb/models/perslay_{part_id}_s{seed}.npz  (embeddings + metrics)
        outputs/fafb/models/perslay_{part_id}_s{seed}.pt   (model state dict)
        outputs/fafb/models/run_log.csv

Uses morphoclass.models.CorianderNet — the official PyTorch PersLay implementation
with learnable Gaussian landmarks, trained end-to-end with Adam + NLL loss.

Run:
  python experiments/fafb_perslay/03_train_perslay.py
  python experiments/fafb_perslay/03_train_perslay.py --partition part_0 --seed 2
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

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
OUT          = Path("outputs/fafb/models")
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH    = OUT / "run_log.csv"
SEEDS       = [0, 1, 2, 3, 4]
N_FEATURES  = 32    # PersLay embedding dim — must match downstream CKA code
N_EPOCHS    = 200
BATCH_SIZE  = 8
LR          = 5e-3
WEIGHT_DECAY = 5e-4

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def build_data_objects(diagrams, labels, scale):
    """Wrap each neuron's persistence diagram as a torch_geometric Data object.

    Normalises to (0, 1) by the global per-dimension max absolute value so
    that CorianderNet's GaussianPointTransformer sample_points (init uniform
    in [0,1]) are in the same range as the input.
    """
    data_list = []
    for diag, y in zip(diagrams, labels):
        d_norm = torch.tensor(diag / scale, dtype=torch.float32)
        data_list.append(Data(
            diagram=d_norm,
            y=torch.tensor(y, dtype=torch.long),
            num_nodes=len(d_norm),
        ))
    return data_list


def _collate(data_list):
    return Batch.from_data_list(data_list, follow_batch=["diagram"])


def make_loader(data_list, indices, batch_size, shuffle=False):
    subset = [data_list[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate)


def evaluate(model, data_list, indices, batch_size, device):
    loader = make_loader(data_list, indices, batch_size)
    preds, true = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.argmax(1).cpu().tolist())
            true.extend(batch.y.cpu().tolist())
    acc = accuracy_score(true, preds)
    f1  = f1_score(true, preds, average="macro", zero_division=0)
    return acc, f1


def extract_embeddings(model, data_list, batch_size, device):
    """Run all neurons through the PersLay feature extractor → (N, N_FEATURES)."""
    all_idx = list(range(len(data_list)))
    loader  = make_loader(data_list, all_idx, batch_size)
    parts   = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.feature_extractor(batch.diagram, batch.diagram_batch)
            parts.append(emb.cpu().numpy())
    return np.vstack(parts)


def train_one(seed, part_id, splits, data_list, n_classes, all_labels, device):
    out_path   = OUT / f"perslay_{part_id}_s{seed}.npz"
    model_path = OUT / f"perslay_{part_id}_s{seed}.pt"

    if out_path.exists() and model_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        d = np.load(out_path, allow_pickle=True)
        return {
            "partition": part_id, "seed": seed,
            "val_acc":  float(d["val_acc"]),
            "test_acc": float(d["test_acc"]),
            "val_f1":   float(d["val_f1"]),
            "test_f1":  float(d["test_f1"]),
        }

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = splits["train"]
    val_idx   = splits["val"]
    test_idx  = splits["test"]

    model     = CorianderNet(n_classes=n_classes, n_features=N_FEATURES).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = make_loader(data_list, train_idx, BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            val_acc, _ = evaluate(model, data_list, val_idx, BATCH_SIZE, device)
            print(f"    epoch {epoch+1:3d}  val_acc={val_acc:.3f}")

    elapsed = time.time() - t0

    val_acc,  val_f1  = evaluate(model, data_list, val_idx,  BATCH_SIZE, device)
    test_acc, test_f1 = evaluate(model, data_list, test_idx, BATCH_SIZE, device)

    all_embeddings = extract_embeddings(model, data_list, BATCH_SIZE, device)

    torch.save(model.state_dict(), model_path)
    np.savez_compressed(
        out_path,
        embeddings=all_embeddings,
        labels=np.array(all_labels),
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array(test_idx),
        val_acc=np.array([val_acc]),
        test_acc=np.array([test_acc]),
        val_f1=np.array([val_f1]),
        test_f1=np.array([test_f1]),
        elapsed=np.array([elapsed]),
    )

    return {
        "partition": part_id, "seed": seed,
        "val_acc": val_acc, "test_acc": test_acc,
        "val_f1":  val_f1,  "test_f1":  test_f1,
        "elapsed_s": elapsed,
    }


def main(only_partition=None, only_seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CorianderNet (PersLay) training sweep")
    print(f"  Device: {device}")
    print(f"  n_features={N_FEATURES}  n_epochs={N_EPOCHS}  lr={LR}")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    n_classes = len(label_names)

    raw = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = []
    for s in samples:
        key = s.path.stem
        diagrams.append(raw[key] if key in raw else np.array([[0.0, 1.0]], dtype=np.float32))

    # Global normalisation to (0, 1) per dimension — matches GaussianPointTransformer init
    all_pts = np.vstack(diagrams)
    scale = np.array([[
        max(abs(all_pts[:, 0].max()), abs(all_pts[:, 0].min())),
        max(abs(all_pts[:, 1].max()), abs(all_pts[:, 1].min())),
    ]])
    print(f"Diagram scale (µm): birth_max={scale[0,0]:.1f}  death_max={scale[0,1]:.1f}")

    all_labels = np.array([s.label_idx for s in samples])
    data_list  = build_data_objects(diagrams, all_labels, scale)

    with open(PARTITIONS) as f:
        partitions = json.load(f)

    log_exists = LOG_PATH.exists()
    log_file   = open(LOG_PATH, "a", newline="")
    writer     = csv.DictWriter(log_file, fieldnames=[
        "partition", "seed", "val_acc", "test_acc", "val_f1", "test_f1", "elapsed_s"
    ])
    if not log_exists:
        writer.writeheader()

    all_results = []
    for part_id, splits in partitions.items():
        if only_partition and part_id != only_partition:
            continue
        for seed in SEEDS:
            if only_seed is not None and seed != only_seed:
                continue
            print(f"Training {part_id} seed={seed}")
            result = train_one(seed, part_id, splits, data_list,
                               n_classes, all_labels, device)
            writer.writerow({k: result.get(k, "") for k in writer.fieldnames})
            log_file.flush()
            print(f"  val_acc={result['val_acc']:.3f}  "
                  f"test_acc={result['test_acc']:.3f}  "
                  f"elapsed={result.get('elapsed_s', 0):.1f}s")
            all_results.append(result)

    log_file.close()

    if all_results:
        val_accs  = [r["val_acc"]  for r in all_results]
        test_accs = [r["test_acc"] for r in all_results]
        chance    = 1.0 / n_classes
        print(f"\n{'='*60}")
        print(f"  Ran {len(all_results)} models  (chance={chance:.3f})")
        print(f"  Val  accuracy:  {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
        print(f"  Test accuracy:  {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
        if np.mean(test_accs) > chance * 1.5:
            print(f"{PASS} Model learns above chance")
        else:
            print(f"{WARN} Near chance — check diagram normalisation or increase n_epochs")
        print(f"{'='*60}\n")

    print(f"Run log → {LOG_PATH}")
    print(f"Next:    python experiments/fafb_perslay/04_extract_embeddings.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=None,
                        help="Run only this partition (e.g. part_0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed (0–4)")
    args = parser.parse_args()
    main(args.partition, args.seed)
