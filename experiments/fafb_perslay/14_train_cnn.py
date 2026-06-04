"""
14_train_cnn.py
===============
Trains CNNet (persistence image CNN) across all partitions × seeds —
same experimental design as 03_train_perslay.py.

CNNet uses a different input encoding (rasterized 2D persistence image)
and different architecture (CNN) from CorianderNet (PersLay set-function).
CKA(CorianderNet-logit, CNNet-logit) is a cross-architecture PRH test:
high CKA means the same topological structure is recovered regardless of
how the persistence diagram is encoded.

Input:  persistence diagrams from outputs/fafb/data/diagrams.npz
Output:
  outputs/fafb/models/cnn_part_{0,1,2}_s{0..4}.pt   — 15 model weights
  outputs/fafb/embeddings/cnn_emb_part_*_s*.npy      — (N, feat_dim) features
  outputs/fafb/embeddings/cnn_logit_part_*_s*.npy    — (N, n_classes) logits
  outputs/fafb/models/cnn_run_log.csv

Run:
  python experiments/fafb_perslay/14_train_cnn.py
  python experiments/fafb_perslay/14_train_cnn.py --partition part_0 --seed 0
"""

import argparse, csv, json, sys, time
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
from morphoclass.models import CNNet

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV     = Path("outputs/fafb/data/dataset.csv")
PARTITIONS   = Path("outputs/fafb/data/partitions.json")
DIAGRAMS_NPZ = Path("outputs/fafb/data/diagrams.npz")
MODEL_OUT    = Path("outputs/fafb/models")
EMB_OUT      = Path("outputs/fafb/embeddings")
MODEL_OUT.mkdir(parents=True, exist_ok=True)
EMB_OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH   = MODEL_OUT / "cnn_run_log.csv"
SEEDS      = [0, 1, 2, 3, 4]
IMAGE_SIZE = 32      # persistence image resolution (32×32 — fast, sufficient detail)
N_EPOCHS   = 500
BATCH_SIZE = 32
LR         = 5e-4
WEIGHT_DECAY = 5e-4

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


# ── Rasterization ─────────────────────────────────────────────────────────────

def rasterize(diag: np.ndarray, bins: int, max_val: float) -> np.ndarray:
    """Rasterize a persistence diagram to a (bins, bins) image using a linear weight."""
    img = np.zeros((bins, bins), dtype=np.float32)
    if len(diag) == 0:
        return img
    b = np.clip(diag[:, 0] / max_val, 0, 1 - 1e-6)
    d = np.clip(diag[:, 1] / max_val, 0, 1 - 1e-6)
    w = d - b                                   # persistence weight
    bi = (b * bins).astype(int)
    di = (d * bins).astype(int)
    np.add.at(img, (di, bi), w)
    return img


def build_data_objects(diagrams, labels, max_val, bins=IMAGE_SIZE):
    data_list = []
    for diag, y in zip(diagrams, labels):
        img = rasterize(diag, bins=bins, max_val=max_val)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        data_list.append(Data(
            image=img_t,
            y=torch.tensor(int(y), dtype=torch.long),
        ))
    return data_list


# ── Data helpers ──────────────────────────────────────────────────────────────

def _collate(data_list):
    return Batch.from_data_list(data_list)


def make_loader(data_list, indices, batch_size, shuffle=False):
    subset = [data_list[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate)


def evaluate(model, data_list, indices, device):
    loader = make_loader(data_list, indices, BATCH_SIZE)
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


def extract_embeddings(model, data_list, device):
    loader = make_loader(data_list, list(range(len(data_list))), BATCH_SIZE)
    feats, logits = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = model.feature_extractor(batch)      # (N, feat_dim)
            l = model(batch)                        # (N, n_classes) log-softmax
            feats.append(f.cpu().numpy())
            logits.append(l.cpu().numpy())
    return np.vstack(feats), np.vstack(logits)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(seed, part_id, splits, data_list, n_classes, device, writer, log_file):
    model_path = MODEL_OUT / f"cnn_{part_id}_s{seed}.pt"
    emb_path   = EMB_OUT   / f"cnn_emb_{part_id}_s{seed}.npy"
    logit_path = EMB_OUT   / f"cnn_logit_{part_id}_s{seed}.npy"

    if model_path.exists() and emb_path.exists():
        print(f"  SKIP (exists): {model_path.name}")
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = splits["train"]
    val_idx   = splits["val"]
    test_idx  = splits["test"]

    model     = CNNet(n_classes=n_classes, image_size=IMAGE_SIZE, bn=True).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader    = make_loader(data_list, train_idx, BATCH_SIZE, shuffle=True)

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            val_acc, _ = evaluate(model, data_list, val_idx, device)
            print(f"    epoch {epoch+1:3d}  val_acc={val_acc:.3f}", flush=True)

    elapsed = time.time() - t0
    val_acc,  val_f1  = evaluate(model, data_list, val_idx,  device)
    test_acc, test_f1 = evaluate(model, data_list, test_idx, device)

    feat_emb, logit_emb = extract_embeddings(model, data_list, device)
    torch.save(model.state_dict(), model_path)
    np.save(emb_path,   feat_emb)
    np.save(logit_path, logit_emb)

    result = {
        "partition": part_id, "seed": seed,
        "val_acc": val_acc, "test_acc": test_acc,
        "val_f1":  val_f1,  "test_f1":  test_f1,
        "elapsed_s": elapsed,
    }
    writer.writerow({k: result.get(k, "") for k in writer.fieldnames})
    log_file.flush()
    print(f"  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  elapsed={elapsed:.0f}s",
          flush=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(only_partition=None, only_seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CNNet (persistence image CNN) training sweep")
    print(f"  Device:     {device}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}  |  epochs: {N_EPOCHS}  |  lr: {LR}")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    n_classes = len(label_names)

    raw = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = [raw[s.path.stem] if s.path.stem in raw
                else np.array([[0., 1.]], dtype=np.float32) for s in samples]

    all_pts = np.vstack(diagrams)
    max_val = float(max(abs(all_pts[:, 0].max()), abs(all_pts[:, 1].max())))
    print(f"Persistence diagram max value (µm): {max_val:.1f}")
    print(f"Rasterizing {len(diagrams):,} diagrams to {IMAGE_SIZE}×{IMAGE_SIZE} images...")

    all_labels = np.array([s.label_idx for s in samples])
    data_list  = build_data_objects(diagrams, all_labels, max_val)
    print(f"Done.\n")

    with open(PARTITIONS) as f:
        partitions = json.load(f)

    log_exists = LOG_PATH.exists()
    log_file   = open(LOG_PATH, "a", newline="")
    writer     = csv.DictWriter(log_file, fieldnames=[
        "partition", "seed", "val_acc", "test_acc", "val_f1", "test_f1", "elapsed_s"
    ])
    if not log_exists:
        writer.writeheader()

    results = []
    for part_id, splits in partitions.items():
        if only_partition and part_id != only_partition:
            continue
        for seed in SEEDS:
            if only_seed is not None and seed != only_seed:
                continue
            print(f"Training CNNet {part_id} seed={seed}", flush=True)
            r = train_one(seed, part_id, splits, data_list, n_classes, device,
                          writer, log_file)
            if r:
                results.append(r)

    log_file.close()

    if results:
        accs = [r["test_acc"] for r in results]
        print(f"\n{'='*60}")
        print(f"  CNNet mean test accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Next: run 05_metrics.py to compute CKA(CorianderNet, CNNet)")
        print(f"{'='*60}\n")

    print(f"Log → {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.partition, args.seed)
