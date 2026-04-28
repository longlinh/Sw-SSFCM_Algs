"""End-to-end demo for SeFCM and Sw-SSFCM.

Run::

    python -m algs.demo
"""
from __future__ import annotations

import time

import numpy as np

try:
    from sklearn.datasets import load_iris
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment
    HAS_SK = True
except Exception:  # pragma: no cover
    HAS_SK = False

from sefcm import SeFCM
from sw_ssfcm import SwSSFCM

SEED = 42


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum()) / len(y_true)


def _make_partial(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Keep ``frac`` labels per class (stratified), mark the rest as -1."""
    yp = np.full_like(y, -1)
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)
        keep = max(1, int(round(frac * len(idx))))
        yp[idx[:keep]] = k
    return yp


def demo_sefcm():
    print("\n=== Demo 1: SeFCM on Iris ===")
    X, y = load_iris(return_X_y=True)
    rng = np.random.default_rng(SEED)
    y_partial = _make_partial(y, frac=0.10, rng=rng)

    t0 = time.time()
    model = SeFCM(n_clusters=3, alpha=2.0, random_state=SEED).fit(X, y_partial)
    elapsed = time.time() - t0

    acc = cluster_acc(y, model.labels_, 3)
    nmi = normalized_mutual_info_score(y, model.labels_)
    print(f"  ACC={acc:.4f}  NMI={nmi:.4f}  iters={model.n_iter_}  time={elapsed:.2f}s")


def demo_sw_ssfcm():
    print("\n=== Demo 2: Sw-SSFCM on synthetic 30x30 image ===")
    H = W = 30
    C = 3
    rng = np.random.default_rng(SEED)

    # Three horizontal stripes of 3-D Gaussian blobs
    gt = np.zeros((H, W), dtype=int)
    gt[H // 3 : 2 * H // 3] = 1
    gt[2 * H // 3 :] = 2
    centers = np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10]], dtype=float)
    X = centers[gt.flatten()] + rng.normal(0, 1.5, (H * W, 3))
    y = gt.flatten()
    y_partial = _make_partial(y, frac=0.05, rng=rng)

    t0 = time.time()
    model = SwSSFCM(
        n_clusters=C, alpha=5.0, radius=1, random_state=SEED
    ).fit(X, y_partial, image_shape=(H, W))
    elapsed = time.time() - t0

    acc = cluster_acc(y, model.labels_, C)
    nmi = normalized_mutual_info_score(y, model.labels_)
    print(f"  ACC={acc:.4f}  NMI={nmi:.4f}  iters={model.n_iter_}  time={elapsed:.2f}s")


if __name__ == "__main__":
    if not HAS_SK:
        raise SystemExit("Demo requires scikit-learn and scipy. "
                         "Install with: pip install -r algs/requirements.txt")
    demo_sefcm()
    demo_sw_ssfcm()
