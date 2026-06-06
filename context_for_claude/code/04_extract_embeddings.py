"""
04_extract_embeddings.py
========================
Loads all trained models and collects two embedding types per model:

  emb_part_X_sY.npy    — (N, 32) PersLay feature-extractor output
  logit_part_X_sY.npy  — (N,  8) log-softmax output (classifier head)

The feature embeddings capture the learned PersLay representation.
The logit embeddings are used for debiased CKA (feature embeddings have an
always-positive constraint that inflates CKA regardless of training).

Outputs:
  outputs/fafb/embeddings/emb_{part_id}_s{seed}.npy
  outputs/fafb/embeddings/logit_{part_id}_s{seed}.npy
  outputs/fafb/embeddings/labels.npy
  outputs/fafb/embeddings/label_names.json
  outputs/fafb/embeddings/morphometric_emb.npy

Run:
  python experiments/fafb_perslay/04_extract_embeddings.py
"""

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_dataset
from morphoclass.models import CorianderNet

DATA_CSV     = Path("outputs/fafb/data/dataset.csv")
DIAGRAMS_NPZ = Path("outputs/fafb/data/diagrams.npz")
MODEL_DIR    = Path("outputs/fafb/models")
OUT          = Path("outputs/fafb/embeddings")
OUT.mkdir(parents=True, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

BATCH_SIZE = 32


def _collate(data_list):
    return Batch.from_data_list(data_list, follow_batch=["diagram"])


def build_data_objects(diagrams, labels, scale):
    return [
        Data(
            diagram=torch.tensor(d / scale, dtype=torch.float32),
            y=torch.tensor(int(y), dtype=torch.long),
            num_nodes=len(d),
        )
        for d, y in zip(diagrams, labels)
    ]


def extract_both(model, data_list, device):
    """Return (feature_emb, logits) each of shape (N, D)."""
    loader = DataLoader(data_list, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=_collate)
    feats, logits = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = model.feature_extractor(batch.diagram, batch.diagram_batch)
            l = model(batch)           # log-softmax (N, n_classes)
            feats.append(f.cpu().numpy())
            logits.append(l.cpu().numpy())
    return np.vstack(feats), np.vstack(logits)


def main(tag=""):
    """
    tag: "" for standard perslay_part_*.pt models → emb_part_*_sN.npy
         "pretrain" for perslay_pretrain_part_*.pt → emb_pretrain_part_*_sN.npy
    """
    model_glob  = f"perslay_{tag+'_' if tag else ''}part_*.pt"
    emb_prefix  = f"{tag+'_' if tag else ''}"

    print(f"\n{'='*60}")
    print(f"  Extracting and validating embeddings  (tag={tag or 'standard'})")
    print(f"  Model glob:  {model_glob}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples, label_names = load_dataset(DATA_CSV)
    labels = np.array([s.label_idx for s in samples])
    n_classes = len(label_names)

    np.save(OUT / "labels.npy", labels)
    with open(OUT / "label_names.json", "w") as f:
        json.dump(label_names, f)
    print(f"{PASS} Labels saved: {len(labels)} neurons, {n_classes} classes\n")

    # ── Load persistence diagrams for forward pass ────────────────────────────
    raw = np.load(DIAGRAMS_NPZ, allow_pickle=True)
    diagrams = [raw[s.path.stem] if s.path.stem in raw
                else np.array([[0., 1.]], dtype=np.float32) for s in samples]
    all_pts = np.vstack(diagrams)
    scale = np.array([[
        max(abs(all_pts[:, 0].max()), abs(all_pts[:, 0].min())),
        max(abs(all_pts[:, 1].max()), abs(all_pts[:, 1].min())),
    ]])
    data_list = build_data_objects(diagrams, labels, scale)

    # ── Process each trained model ────────────────────────────────────────────
    pt_files = sorted(MODEL_DIR.glob(model_glob))
    if not pt_files:
        print(f"{FAIL} No .pt model files matching '{model_glob}' in {MODEL_DIR}")
        script = "13_pretrain_contrastive.py" if tag == "pretrain" else "03_train_perslay.py"
        print(f"  Run {script} first")
        return

    print(f"Found {len(pt_files)} model files:")
    for pt in pt_files:
        # Skip if both output files exist and are newer than the model
        m = re.search(r'part_(\d+)_s(\d+)', pt.stem)
        if m:
            pid, sid = m.group(1), m.group(2)
            emb_out   = OUT / f"emb_{emb_prefix}part_{pid}_s{sid}.npy"
            logit_out = OUT / f"logit_{emb_prefix}part_{pid}_s{sid}.npy"
            if (emb_out.exists() and logit_out.exists() and
                    emb_out.stat().st_mtime > pt.stat().st_mtime):
                print(f"  SKIP (up-to-date): {pt.name}")
                continue
        # Robust parser: extract part number and seed via regex
        # handles both perslay_part_0_s2 and perslay_pretrain_part_0_s2
        m = re.search(r'part_(\d+)_s(\d+)', pt.stem)
        if not m:
            print(f"  {WARN} Cannot parse part/seed from {pt.name} — skipping")
            continue
        part_id = f"part_{m.group(1)}"
        seed    = int(m.group(2))

        model = CorianderNet(n_classes=n_classes, n_features=32).to(device)
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True))

        emb, logit = extract_both(model, data_list, device)

        np.save(OUT / f"emb_{emb_prefix}{part_id}_s{seed}.npy",   emb)
        np.save(OUT / f"logit_{emb_prefix}{part_id}_s{seed}.npy", logit)

        zero_rows = int((np.abs(emb).sum(axis=1) == 0).sum())
        print(f"  {pt.name}: feat={emb.shape}  logit={logit.shape}  "
              f"feat_norm={np.linalg.norm(emb, axis=1).mean():.1f}  "
              f"zero_rows={zero_rows}")
        if zero_rows > len(samples) * 0.1:
            print(f"  {WARN} Many zero rows — check diagram quality")

    # ── Morphometric baseline embedding ──────────────────────────────────────
    morph_path = Path("outputs/fafb/data/morphometrics.npy")
    if morph_path.exists():
        morph = np.load(morph_path)
        mu  = morph.mean(axis=0)
        std = morph.std(axis=0) + 1e-9
        np.save(OUT / "morphometric_emb.npy", (morph - mu) / std)
        print(f"\n{PASS} Morphometric baseline: shape={morph.shape}")
    else:
        print(f"\n{WARN} No morphometrics.npy — run 02_compute_persistence.py first")

    next_step = "12_progress_report.py --tag pretrain" if tag == "pretrain" \
                else "05_metrics.py"
    print(f"\n{PASS} All embeddings in {OUT}/")
    print(f"  emb_{emb_prefix}part_*.npy   — 32-dim PersLay features")
    print(f"  logit_{emb_prefix}part_*.npy — {n_classes}-dim log-softmax (use for CKA)")
    print(f"Next: python experiments/fafb_perslay/{next_step}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="",
                        help="Model tag: '' for standard perslay_part_*.pt, "
                             "'pretrain' for perslay_pretrain_part_*.pt")
    args = parser.parse_args()
    main(args.tag)
