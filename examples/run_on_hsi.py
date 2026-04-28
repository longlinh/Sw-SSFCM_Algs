"""Template: run Sw-SSFCM on a hyperspectral image.

This script is a *blueprint* — adapt the ``load_hsi`` block to whatever
format your dataset is in. Common HSI benchmarks (Salinas, Indian Pines,
Pavia, Botswana, KSC, WHU-Hi-LongKou) are typically distributed as either:

  * MATLAB .mat files containing 3-D cube ``X (H, W, B)`` + ground-truth
    label image ``Y (H, W)`` with 0 = background, [1..C] = class ids.
  * ENVI / HDF5 / .npy variants thereof.

We expect at the end of ``load_hsi`` to obtain:

    cube : ndarray, shape (H, W, B)   # H rows, W columns, B spectral bands
    gt   : ndarray, shape (H, W)      # int labels, 0 = ignore

The script then:
  1. flattens the cube to ``(H*W, B)`` (row-major),
  2. constructs a partial-label vector with ~`labels_per_class` samples
     per class drawn at random from labeled pixels,
  3. fits Sw-SSFCM,
  4. reports overall ACC + NMI on labeled pixels.

Usage::

    cd <repo-root>
    python examples/run_on_hsi.py /path/to/dataset.mat

Requires ``scipy`` to read ``.mat`` files (already in requirements.txt).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make sibling modules importable when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sw_ssfcm import SwSSFCM                                     # noqa: E402

SEED = 42


# ---------------------------------------------------------------------------
def load_hsi(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load HSI cube + ground truth.

    Adapt this function to your dataset. The default branch handles MATLAB
    ``.mat`` files containing keys ``X`` and ``Y`` (case-insensitive search).
    """
    from scipy.io import loadmat

    mat = loadmat(path)
    cube_key = next(k for k in mat if k.lower() in {"x", "data", "img", "cube"})
    gt_key = next(k for k in mat if k.lower() in {"y", "gt", "labels", "groundtruth"})
    cube = np.asarray(mat[cube_key])
    gt = np.asarray(mat[gt_key]).astype(int)

    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D cube (H, W, B), got shape {cube.shape}")
    if gt.shape != cube.shape[:2]:
        raise ValueError(
            f"Cube spatial shape {cube.shape[:2]} != gt shape {gt.shape}"
        )
    return cube.astype(np.float64), gt


def stratified_partial(gt_flat: np.ndarray, labels_per_class: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Pick at most ``labels_per_class`` labeled samples per non-zero class.

    Background (gt == 0) is treated as unlabeled. Labels in the returned
    vector are 0-indexed (gt class 1 -> 0, class 2 -> 1, ...).
    """
    yp = np.full_like(gt_flat, -1)
    for k in np.unique(gt_flat):
        if k == 0:
            continue
        idx = np.where(gt_flat == k)[0]
        rng.shuffle(idx)
        yp[idx[: labels_per_class]] = k - 1
    return yp


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum()) / len(y_true)


# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")

    print(f"[load] {path}")
    cube, gt = load_hsi(path)
    H, W, B = cube.shape
    C = int(gt.max())
    print(f"  cube={cube.shape}  bands={B}  classes={C}")

    X = cube.reshape(H * W, B)
    gt_flat = gt.flatten()
    rng = np.random.default_rng(SEED)
    y_partial = stratified_partial(gt_flat, labels_per_class=60, rng=rng)

    # Tip: tune `alpha` per dataset (HSI typically benefits from alpha in [10, 100]).
    print("[fit] Sw-SSFCM(alpha=30, radius=2)")
    t0 = time.time()
    model = SwSSFCM(
        n_clusters=C, alpha=30.0, radius=2,
        max_iter=200, tol=1e-4, random_state=SEED,
    ).fit(X, y_partial, image_shape=(H, W))
    elapsed = time.time() - t0
    print(f"  iters={model.n_iter_}  time={elapsed:.1f}s")

    # Evaluate on labeled pixels only.
    mask = gt_flat > 0
    y_true = gt_flat[mask] - 1
    y_pred = model.labels_[mask]
    acc = cluster_acc(y_true, y_pred, C)
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"[eval] ACC={acc:.4f}  NMI={nmi:.4f}")


if __name__ == "__main__":
    main()
