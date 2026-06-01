"""
13_pretrain_contrastive.py
==========================
Pretrains CorianderNet (PersLay) jointly with Supervised Contrastive Loss
(Khosla et al. 2020) + Cross-Entropy on the FAFB dataset.

One pretrained model per partition (3 total); no seed sweep — pretraining is
meant as a stable initialisation, not a variance study.

Intended workflow:
  1. Sample up to 10k/class (unbalanced ok):
       python sample_dataset.py --per-class 10000 --seed 42
  2. Compute persistence diagrams:
       sbatch prep_data.slurm
  3. Run this pretraining (GPU):
       sbatch pretrain_contrastive.slurm
  4. Fine-tune or evaluate:
       sbatch --export=ALL,INIT_FROM_PRETRAIN=1 train_perslay.slurm

Outputs:
  outputs/fafb/models/perslay_pretrain_part_{0,1,2}.pt  — model state dicts
  outputs/fafb/models/pretrain_log.csv                  — per-epoch metrics

Run:
  python experiments/fafb_perslay/13_pretrain_contrastive.py
  python experiments/fafb_perslay/13_pretrain_contrastive.py --partition part_0
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
from morphoclass.models import CorianderNet

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV     = Path("outputs/fafb/data/dataset.csv")
PARTITIONS   = Path("outputs/fafb/data/partitions.json")
DIAGRAMS_NPZ = Path("outputs/fafb/data/diagrams.npz")
OUT          = Path("outputs/fafb/models")
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "pretrain_log.csv"

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FEATURES   = 32       # must match downstream CKA code
N_EPOCHS     = 300      # pretraining convergence; fine-tune further with 03_train
BATCH_SIZE   = 32       # per-class examples in balanced batch (total = K × n_classes)
LR           = 5e-4
WEIGHT_DECAY = 5e-4
ALPHA        = 0.5      # loss = ALPHA * CE + (1-ALPHA) * SupCon
TEMPERATURE  = 0.07     # SupCon temperature (standard value from Khosla 2020)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


# ── Supervised Contrastive Loss ────────────────────────────────────────────────

def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al. 2020, Eq. 2).

    features : (N, D) unit-normalised embeddings
    labels   : (N,)  integer class labels

    Pulls same-class embeddings together and pushes different-class apart
    using all available positives per anchor in the batch.
    """
    device = features.device
    n = features.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrix scaled by temperature
    sim = torch.matmul(features, features.T) / temperature  # (N, N)

    # Positive mask: same class, exclude self
    labels_col = labels.unsqueeze(0)   # (1, N)
    labels_row = labels.unsqueeze(1)   # (N, 1)
    pos_mask  = (labels_row == labels_col)                            # (N, N)
    self_mask = ~torch.eye(n, dtype=torch.bool, device=device)       # (N, N)
    pos_mask  = pos_mask & self_mask

    n_pos = pos_mask.float().sum(dim=1)   # (N,) — positives per anchor

    # Numerical stability: subtract row max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Log-sum-exp denominator over all non-self pairs
    exp_sim  = torch.exp(sim) * self_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Mean log-prob over positives per anchor
    loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / (n_pos + 1e-12)

    # Only include anchors that have at least one positive in the batch
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_per[valid].mean()


# ── Class-balanced batch sampler ──────────────────────────────────────────────

class ClassBalancedSampler:
    """
    Yields index lists of size (k_per_class × n_classes) per iteration,
    ensuring every class appears k_per_class times.  For small classes,
    samples with replacement.
    """
    def __init__(self, labels: np.ndarray, k_per_class: int, n_batches: int,
                 seed: int = 0):
        self.labels     = labels
        self.k          = k_per_class
        self.n_batches  = n_batches
        self.rng        = np.random.default_rng(seed)
        classes         = np.unique(labels)
        self.cls_idx    = {c: np.where(labels == c)[0] for c in classes}

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for idx in self.cls_idx.values():
                replace = len(idx) < self.k
                chosen  = self.rng.choice(idx, size=self.k, replace=replace)
                batch.extend(chosen.tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ── Data helpers (shared with 03_train_perslay.py) ────────────────────────────

def build_data_objects(diagrams, labels, scale):
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


def make_balanced_loader(data_list, indices, labels, k_per_class,
                         n_batches, seed=0):
    subset    = [data_list[i] for i in indices]
    sub_labels = np.array(labels)[indices]
    sampler   = ClassBalancedSampler(sub_labels, k_per_class, n_batches, seed)

    def balanced_collate(batch_indices):
        return _collate([subset[i] for i in batch_indices])

    return DataLoader(
        dataset=list(range(len(subset))),
        batch_sampler=sampler,
        collate_fn=balanced_collate,
    )


def evaluate(model, data_list, indices, batch_size, device):
    loader = make_loader(data_list, indices, batch_size)
    preds, true = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch)
            preds.extend(out.argmax(1).cpu().tolist())
            true.extend(batch.y.cpu().tolist())
    acc = accuracy_score(true, preds)
    f1  = f1_score(true, preds, average="macro", zero_division=0)
    return acc, f1


# ── Training ──────────────────────────────────────────────────────────────────

def pretrain_one(part_id, splits, data_list, all_labels, n_classes, device,
                 writer, log_file):
    model_path = OUT / f"perslay_pretrain_{part_id}.pt"
    if model_path.exists():
        print(f"  SKIP (exists): {model_path.name}")
        return

    torch.manual_seed(42)
    np.random.seed(42)

    train_idx = splits["train"]
    val_idx   = splits["val"]

    # Balanced loader: BATCH_SIZE examples per class per step
    n_classes_present = len(np.unique(np.array(all_labels)[train_idx]))
    n_batches = max(1, len(train_idx) // (BATCH_SIZE * n_classes_present))
    train_loader = make_balanced_loader(
        data_list, train_idx, all_labels,
        k_per_class=BATCH_SIZE,
        n_batches=n_batches * N_EPOCHS,
        seed=42,
    )

    model     = CorianderNet(n_classes=n_classes, n_features=N_FEATURES).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n  Balanced batch: {BATCH_SIZE}/class × {n_classes_present} classes "
          f"= {BATCH_SIZE * n_classes_present} per step  |  {n_batches} steps/epoch")

    t0 = time.time()
    step = 0
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_ce = epoch_sc = epoch_total = 0.0
        n_steps = 0

        for batch in train_loader:
            # Each epoch consumes exactly n_batches steps from the pre-generated loader
            if n_steps >= n_batches:
                break
            batch = batch.to(device)

            # Forward — full model for CE; feature extractor for SupCon
            logits   = model(batch)                                        # log-softmax (N, C)
            features = model.feature_extractor(batch.diagram,
                                               batch.diagram_batch)        # (N, F)
            features_norm = F.normalize(features, p=2, dim=1)             # unit sphere

            ce_loss  = F.nll_loss(logits, batch.y)
            sc_loss  = supcon_loss(features_norm, batch.y)
            loss     = ALPHA * ce_loss + (1.0 - ALPHA) * sc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce    += ce_loss.item()
            epoch_sc    += sc_loss.item()
            epoch_total += loss.item()
            n_steps += 1
            step    += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            val_acc, val_f1 = evaluate(model, data_list, val_idx, BATCH_SIZE * 4, device)
            elapsed = time.time() - t0
            print(f"    epoch {epoch+1:3d}  "
                  f"ce={epoch_ce/n_steps:.3f}  "
                  f"sc={epoch_sc/n_steps:.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"elapsed={elapsed:.0f}s")
            row = {
                "partition": part_id,
                "epoch":     epoch + 1,
                "ce_loss":   round(epoch_ce / n_steps, 4),
                "sc_loss":   round(epoch_sc / n_steps, 4),
                "val_acc":   round(val_acc, 4),
                "val_f1":    round(val_f1, 4),
                "elapsed_s": round(elapsed, 1),
            }
            writer.writerow(row)
            log_file.flush()

    torch.save(model.state_dict(), model_path)
    print(f"  {PASS} Saved → {model_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(only_partition=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CorianderNet contrastive pretraining")
    print(f"  Device:   {device}")
    print(f"  Loss:     {ALPHA:.1f}×CE + {1-ALPHA:.1f}×SupCon  (τ={TEMPERATURE})")
    print(f"  Epochs:   {N_EPOCHS}  |  k/class: {BATCH_SIZE}  |  lr: {LR}")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    n_classes = len(label_names)
    print(f"Dataset: {len(samples):,} neurons  |  {n_classes} classes\n")

    raw = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = []
    for s in samples:
        key = s.path.stem
        diagrams.append(raw[key] if key in raw else np.array([[0.0, 1.0]], dtype=np.float32))

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
        "partition", "epoch", "ce_loss", "sc_loss", "val_acc", "val_f1", "elapsed_s"
    ])
    if not log_exists:
        writer.writeheader()

    for part_id, splits in partitions.items():
        if only_partition and part_id != only_partition:
            continue
        print(f"\nPretraining {part_id}  "
              f"(train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])} held-out)")
        pretrain_one(part_id, splits, data_list, all_labels,
                     n_classes, device, writer, log_file)

    log_file.close()
    print(f"\nPretrain log → {LOG_PATH}")
    print(f"Models      → {OUT}/perslay_pretrain_part_*.pt")
    print(f"\nNext: fine-tune with 03_train_perslay.py or run eval directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=None,
                        help="Run only this partition (e.g. part_0)")
    args = parser.parse_args()
    main(args.partition)
