"""
03_train_perslay.py
===================
Trains PersLay models across all partitions and seeds.
Reads: outputs/fafb/data/{dataset.csv, partitions.json, diagrams.npz}
Saves: outputs/fafb/models/perslay_{part_id}_s{seed}.npz  (landmark params + fit state)
       outputs/fafb/models/run_log.csv

For the subset validation this uses the numpy PersLay + sklearn classifier.
The morphoclass version would swap in the PyTorch PersLay layer but the
interface (fit/predict/embed) is identical.

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset, PersLayNumpy

DATA_CSV     = Path("outputs/fafb/data/dataset.csv")
PARTITIONS   = Path("outputs/fafb/data/partitions.json")
DIAGRAMS_NPZ = Path("outputs/fafb/data/diagrams.npz")
OUT          = Path("outputs/fafb/models")
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run_log.csv"
SEEDS = [0, 1, 2, 3, 4]

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def load_diagrams(samples):
    """Load precomputed persistence diagrams aligned to sample order."""
    data = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = []
    for s in samples:
        key = s.path.stem
        if key in data:
            diagrams.append(data[key])
        else:
            diagrams.append(np.array([[0.0, 1.0]], dtype=np.float32))
    return diagrams


def train_one(seed: int, part_id: str, splits: dict,
              samples, label_names, diagrams,
              n_landmarks: int = 32, sigma: float = 10.0):
    """
    Single training run: fit PersLay landmarks on train set,
    embed all neurons, train logistic regression classifier,
    evaluate on val and test sets.

    Returns: dict of metrics
    """
    out_path = OUT / f"perslay_{part_id}_s{seed}.npz"
    if out_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        # Load and return saved metrics
        d = np.load(out_path, allow_pickle=True)
        return {
            "partition": part_id, "seed": seed,
            "val_acc": float(d["val_acc"]), "test_acc": float(d["test_acc"]),
            "val_f1": float(d["val_f1"]),   "test_f1": float(d["test_f1"]),
        }

    t0 = time.time()

    train_idx = splits["train"]
    val_idx   = splits["val"]
    test_idx  = splits["test"]

    # ── Fit PersLay on training diagrams ──────────────────────────────────────
    # Note: alpha (weight function steepness) varies by seed to test sensitivity
    # In the full morphoclass version this would be a learnable parameter
    alpha = 0.3 + seed * 0.1  # [0.3, 0.4, 0.5, 0.6, 0.7] across seeds
    perslay = PersLayNumpy(
        n_landmarks=n_landmarks,
        sigma=sigma,
        alpha=alpha,
        seed=seed,
    )
    train_diagrams = [diagrams[i] for i in train_idx]
    perslay.fit(train_diagrams)

    # ── Embed all neurons ─────────────────────────────────────────────────────
    all_embeddings = perslay.transform(diagrams)  # (N, n_landmarks)

    # ── Train logistic regression classifier ──────────────────────────────────
    labels = np.array([s.label_idx for s in samples])

    X_train = all_embeddings[train_idx]
    y_train = labels[train_idx]
    X_val   = all_embeddings[val_idx]
    y_val   = labels[val_idx]
    X_test  = all_embeddings[test_idx]
    y_test  = labels[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_train, y_train)

    val_preds  = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    val_acc  = accuracy_score(y_val,  val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    val_f1   = f1_score(y_val,  val_preds,  average='macro', zero_division=0)
    test_f1  = f1_score(y_test, test_preds, average='macro', zero_division=0)

    elapsed = time.time() - t0

    # ── Save model state ──────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        landmarks=perslay.landmarks,
        alpha=np.array([perslay.alpha]),
        sigma=np.array([perslay.sigma]),
        embeddings=all_embeddings,
        labels=labels,
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
        "val_f1": val_f1,   "test_f1": test_f1,
        "elapsed_s": elapsed,
    }


def main(only_partition=None, only_seed=None):
    print(f"\n{'='*60}")
    print("  PersLay training sweep")
    print(f"{'='*60}\n")

    samples, label_names = load_dataset(DATA_CSV)
    diagrams = load_diagrams(samples)

    with open(PARTITIONS) as f:
        partitions = json.load(f)

    # ── Init log file ─────────────────────────────────────────────────────────
    log_exists = LOG_PATH.exists()
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.DictWriter(log_file,
        fieldnames=["partition","seed","val_acc","test_acc","val_f1","test_f1","elapsed_s"])
    if not log_exists:
        writer.writeheader()

    # ── Run sweep ─────────────────────────────────────────────────────────────
    all_results = []
    n_total = 0
    for part_id, splits in partitions.items():
        if only_partition and part_id != only_partition:
            continue
        for seed in SEEDS:
            if only_seed is not None and seed != only_seed:
                continue
            n_total += 1
            print(f"Training {part_id} seed={seed}  (alpha={0.3+seed*0.1:.1f})")
            result = train_one(
                seed=seed, part_id=part_id, splits=splits,
                samples=samples, label_names=label_names,
                diagrams=diagrams,
            )
            writer.writerow(result)
            log_file.flush()
            print(f"  val_acc={result['val_acc']:.3f}  test_acc={result['test_acc']:.3f}"
                  f"  val_f1={result['val_f1']:.3f}  elapsed={result.get('elapsed_s',0):.1f}s")
            all_results.append(result)

    log_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        val_accs  = [r["val_acc"]  for r in all_results]
        test_accs = [r["test_acc"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"  Ran {len(all_results)} models")
        print(f"  Val  accuracy:  {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
        print(f"  Test accuracy:  {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
        chance = 1.0 / len(label_names)
        print(f"  Chance level:   {chance:.3f}  ({len(label_names)} classes)")
        if np.mean(test_accs) > chance * 1.5:
            print(f"{PASS} Model learns above chance")
        else:
            print(f"{WARN} Test accuracy near chance — check diagram quality or n_landmarks")
        print(f"{'='*60}\n")

    print(f"Run log → {LOG_PATH}")
    print(f"Next: python experiments/fafb_perslay/04_extract_embeddings.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.partition, args.seed)
