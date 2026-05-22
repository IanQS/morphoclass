"""
utils.py — shared helpers for the FAFB PersLay PRH validation experiment.

Designed to run standalone (no morphoclass required) but mirrors the
morphoclass API so the swap is a one-line import change.

All coordinate inputs are assumed to be in NANOMETERS — rescaling to µm
(divide by 1000) is applied internally before persistence computation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MorphologySample:
    """Mirrors morphoclass's Data object."""
    path: Path
    label: str
    label_idx: int
    # filled after transforms
    diagram: Optional[np.ndarray] = None   # shape (N, 2) birth/death pairs
    features: Optional[np.ndarray] = None  # morphometric feature vector
    embedding: Optional[np.ndarray] = None # PersLay output


# ── SWC parsing ───────────────────────────────────────────────────────────────

def parse_swc(path: Path) -> np.ndarray:
    """
    Parse SWC file → (N, 7) array: [id, type, x, y, z, radius, parent].
    Skips comment lines. Coordinates returned as-is (nanometers for FAFB).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                rows.append([float(p) for p in parts[:7]])
    if not rows:
        raise ValueError(f"Empty SWC: {path}")
    return np.array(rows)


def swc_to_tree(arr: np.ndarray) -> dict[int, list[int]]:
    """Build adjacency list (child→parent direction) from SWC array."""
    id_to_idx = {int(r[0]): i for i, r in enumerate(arr)}
    children: dict[int, list[int]] = {int(r[0]): [] for r in arr}
    for r in arr:
        node_id = int(r[0])
        parent_id = int(r[6])
        if parent_id != -1 and parent_id in children:
            children[parent_id].append(node_id)
    return children


# ── Coordinate rescaling ──────────────────────────────────────────────────────

SCALE_NM_TO_UM = 1.0 / 1000.0


# ── Persistence diagram from SWC ──────────────────────────────────────────────

def swc_to_persistence_diagram(path: Path, scale: float = SCALE_NM_TO_UM) -> np.ndarray:
    """
    Compute H0 persistence diagram from a neuron skeleton using the
    'height filtration on the tree' approach:

    1. Root the tree at the soma node (type=1, parent=-1).
    2. Compute geodesic (path) distance from root to every node.
    3. Each branching subtree gives a (birth, death) pair where:
       - birth  = distance at which the branch point appears
       - death  = distance of the branch's longest tip
    4. Return sorted (birth, death) array in µm.

    This mirrors morphoclass's BranchingOnlyNeurites + PersistenceDiagram
    transform pipeline at a conceptual level.
    """
    arr = parse_swc(path)
    arr[:, 2:5] *= scale  # nm → µm

    id_to_row = {int(r[0]): r for r in arr}
    id_to_children = swc_to_tree(arr)

    # Find root (parent == -1)
    roots = [int(r[0]) for r in arr if int(r[6]) == -1]
    if not roots:
        raise ValueError(f"No root node in {path}")
    root = roots[0]

    # BFS to compute path distances from root
    from collections import deque
    dist = {root: 0.0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        r = id_to_row[node]
        for child in id_to_children.get(node, []):
            cr = id_to_row[child]
            edge_len = float(np.linalg.norm(cr[2:5] - r[2:5]))
            dist[child] = dist[node] + edge_len
            queue.append(child)

    # Precompute max subtree tip distance iteratively (BFS reverse — no recursion)
    from collections import deque as _deque
    bfs_order = []
    _q = _deque([root])
    while _q:
        _n = _q.popleft()
        bfs_order.append(_n)
        for _c in id_to_children.get(_n, []):
            _q.append(_c)

    max_tip: dict[int, float] = {}
    for _n in reversed(bfs_order):
        _ch = id_to_children.get(_n, [])
        max_tip[_n] = dist[_n] if not _ch else max(max_tip.get(_c, dist[_c]) for _c in _ch)

    # Collect persistence pairs: for each branching node, pair its distance
    # with the maximum tip distance in each of its subtrees (H0 style)
    pairs = []

    # Iterative DFS
    stack = [root]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        children = id_to_children.get(node, [])
        if len(children) >= 2:
            tip_dists = sorted([max_tip[c] for c in children], reverse=True)
            birth = dist[node]
            for death in tip_dists[1:]:
                if death > birth:
                    pairs.append((birth, death))
        for c in children:
            stack.append(c)

    if not pairs:
        # Degenerate tree (unbranched path): single pair root→tip
        tip = max(dist.values())
        pairs = [(0.0, tip)]

    return np.array(pairs, dtype=np.float32)


# ── Morphometric features ─────────────────────────────────────────────────────

def extract_morphometrics(path: Path, scale: float = SCALE_NM_TO_UM) -> np.ndarray:
    """
    Extract 8 morphometric features from SWC file.
    Returns 1D numpy array. All lengths in µm.

    Features:
      0  total_cable_length
      1  n_branch_points
      2  n_tips
      3  max_branch_order
      4  soma_radius (µm)
      5  mean_segment_length
      6  asymmetry_index
      7  max_path_length
    """
    arr = parse_swc(path)
    arr[:, 2:5] *= scale
    arr[:, 5] *= scale  # radius too

    id_to_row = {int(r[0]): r for r in arr}
    id_to_children = swc_to_tree(arr)
    roots = [int(r[0]) for r in arr if int(r[6]) == -1]
    root = roots[0] if roots else int(arr[0, 0])

    # 0: total cable length
    total_length = 0.0
    edges = 0
    for r in arr:
        p = int(r[6])
        if p != -1 and p in id_to_row:
            total_length += float(np.linalg.norm(r[2:5] - id_to_row[p][2:5]))
            edges += 1

    # branch points and tips
    n_branch = sum(1 for n, ch in id_to_children.items() if len(ch) >= 2)
    n_tips = sum(1 for n, ch in id_to_children.items() if len(ch) == 0 and n != root)

    # 3: max branch order (BFS)
    from collections import deque
    order = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in id_to_children.get(node, []):
            parent_order = order[node]
            # increment order at branch points
            order[child] = parent_order + (1 if len(id_to_children.get(node, [])) >= 2 else 0)
            queue.append(child)
    max_order = max(order.values()) if order else 0

    # 4: soma radius
    soma_radius = float(id_to_row[root][5]) if root in id_to_row else 0.0

    # 5: mean segment length
    mean_seg = total_length / max(edges, 1)

    # 6: asymmetry index
    # split at root's children: compare longest paths
    root_children = id_to_children.get(root, [])
    if len(root_children) >= 2:
        from collections import deque as dq2
        def max_depth(start):
            best = 0.0
            st = [(start, 0.0)]
            while st:
                n, d = st.pop()
                r_ = id_to_row[n]
                for c in id_to_children.get(n, []):
                    cr = id_to_row[c]
                    new_d = d + float(np.linalg.norm(cr[2:5] - r_[2:5]))
                    best = max(best, new_d)
                    st.append((c, new_d))
            return best
        depths = sorted([max_depth(c) for c in root_children], reverse=True)
        asym = (depths[0] - depths[1]) / (depths[0] + depths[1] + 1e-9)
    else:
        asym = 0.0

    # 7: max path length from root
    dist = {root: 0.0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        r = id_to_row[node]
        for child in id_to_children.get(node, []):
            cr = id_to_row[child]
            dist[child] = dist[node] + float(np.linalg.norm(cr[2:5] - r[2:5]))
            queue.append(child)
    max_path = max(dist.values()) if dist else 0.0

    return np.array([
        total_length,
        float(n_branch),
        float(n_tips),
        float(max_order),
        soma_radius,
        mean_seg,
        asym,
        max_path,
    ], dtype=np.float32)


MORPHOMETRIC_NAMES = [
    "total_cable_length", "n_branch_points", "n_tips",
    "max_branch_order", "soma_radius", "mean_segment_length",
    "asymmetry_index", "max_path_length",
]


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(csv_path: Path) -> tuple[list[MorphologySample], list[str]]:
    """
    Load dataset.csv (path\\tlabel format).
    Returns (samples, label_names) where label_names[i] = class string.
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                rows.append((Path(parts[0]), parts[1]))

    label_names = sorted(set(r[1] for r in rows))
    label2idx = {l: i for i, l in enumerate(label_names)}

    samples = [
        MorphologySample(path=p, label=lbl, label_idx=label2idx[lbl])
        for p, lbl in rows
    ]
    return samples, label_names


# ── PersLay (self-contained numpy implementation) ─────────────────────────────

class PersLayNumpy:
    """
    Minimal differentiable-free PersLay implementation in numpy.

    Architecture:
      For each persistence diagram D = {(b_i, d_i)}:
        1. Compute persistence:  p_i = d_i - b_i
        2. Weight:  w_i = tanh(alpha * p_i)   (learned parameter alpha)
        3. For each Gaussian landmark g_k:
              phi_ik = exp(-||point_i - g_k||² / (2*sigma²))
        4. Aggregate: z_k = sum_i(w_i * phi_ik) / (sum_i w_i + eps)
        5. Output: z = [z_1, ..., z_K]  (embedding vector, dim=K)

    Parameters alpha, sigma, and landmark positions are initialized from data
    and optionally refined by gradient-free optimization (not implemented here
    — for the validation study, fixed parameters are sufficient to test
    whether the representation is stable across seeds/architectures).
    """

    def __init__(
        self,
        n_landmarks: int = 32,
        sigma: float = 5.0,    # µm scale
        alpha: float = 0.5,
        seed: int = 0,
    ):
        self.n_landmarks = n_landmarks
        self.sigma = sigma
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.landmarks: Optional[np.ndarray] = None  # (K, 2)

    def fit(self, diagrams: list[np.ndarray]):
        """Set landmark positions from data (k-means++ init)."""
        all_pts = np.vstack([d for d in diagrams if len(d) > 0])
        # Simple: uniformly sample from all diagram points
        idx = self.rng.choice(len(all_pts), size=min(self.n_landmarks, len(all_pts)), replace=False)
        self.landmarks = all_pts[idx].astype(np.float32)
        return self

    def transform_one(self, diagram: np.ndarray) -> np.ndarray:
        """Embed a single persistence diagram → (n_landmarks,) vector."""
        if self.landmarks is None:
            raise RuntimeError("Call fit() before transform_one()")
        if len(diagram) == 0:
            return np.zeros(self.n_landmarks, dtype=np.float32)

        pts = diagram.astype(np.float32)  # (N, 2)
        persistence = pts[:, 1] - pts[:, 0]  # (N,)
        weights = np.tanh(self.alpha * persistence)  # (N,) ∈ [0,1]

        # Pairwise distances to landmarks: (N, K)
        diff = pts[:, None, :] - self.landmarks[None, :, :]  # (N, K, 2)
        sq_dist = (diff ** 2).sum(axis=-1)                   # (N, K)
        phi = np.exp(-sq_dist / (2 * self.sigma ** 2))        # (N, K)

        # Weighted sum aggregation
        z = (weights[:, None] * phi).sum(axis=0)             # (K,)
        z = z / (weights.sum() + 1e-9)
        return z.astype(np.float32)

    def transform(self, diagrams: list[np.ndarray]) -> np.ndarray:
        """Embed list of diagrams → (N, n_landmarks) matrix."""
        return np.stack([self.transform_one(d) for d in diagrams])


# ── Evaluation metrics ────────────────────────────────────────────────────────

def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Debiased Centered Kernel Alignment (Kornblith et al. 2019).
    Uses the linear kernel K = X X^T directly — no row-wise L2-normalisation,
    which would convert it to a cosine kernel and inflate CKA for always-positive
    embeddings (e.g. raw PersLay features).
    X, Y: (n_samples, d) float arrays.
    Returns scalar in [0, 1].
    """
    n = X.shape[0]
    K = X @ X.T
    L = Y @ Y.T

    def debiased_hsic(A: np.ndarray, B: np.ndarray) -> float:
        """Debiased HSIC estimator — removes diagonal before summing."""
        n_ = A.shape[0]
        np.fill_diagonal(A, 0.0)
        np.fill_diagonal(B, 0.0)
        term1 = np.sum(A * B)
        term2 = np.sum(A) * np.sum(B) / ((n_ - 1) * (n_ - 2))
        term3 = 2.0 * np.sum(A, axis=1) @ np.sum(B, axis=1) / (n_ - 2)
        return (term1 + term2 - term3) / (n_ * (n_ - 3))

    K_, L_ = K.copy(), L.copy()
    hsic_kl = debiased_hsic(K_, L_)
    hsic_kk = debiased_hsic(K.copy(), K.copy())
    hsic_ll = debiased_hsic(L.copy(), L.copy())

    denom = np.sqrt(max(hsic_kk, 0) * max(hsic_ll, 0))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


def permutation_cka_null(X: np.ndarray, Y: np.ndarray,
                          n_permutations: int = 500,
                          seed: int = 0) -> np.ndarray:
    """Shuffle rows of Y, recompute CKA. Returns null distribution array."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(Y))
        null.append(debiased_cka(X, Y[perm]))
    return np.array(null)


def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score without sklearn dependency on embeddings."""
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(X, labels, metric='euclidean'))


def knn_jaccard(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Mean kNN Jaccard similarity between two embedding spaces.
    Requires X and Y to be embeddings of the SAME neurons (same row order).
    """
    from sklearn.neighbors import NearestNeighbors

    def get_neighbors(Z):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        _, idx = nn.kneighbors(Z)
        return [set(row[1:]) for row in idx]  # exclude self

    nbrs_x = get_neighbors(X)
    nbrs_y = get_neighbors(Y)
    jaccards = [
        len(a & b) / len(a | b) if (a | b) else 0.0
        for a, b in zip(nbrs_x, nbrs_y)
    ]
    return float(np.mean(jaccards))
