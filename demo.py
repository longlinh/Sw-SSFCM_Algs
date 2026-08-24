#!/usr/bin/env python3
"""Minimal, self-contained demonstration on a synthetic hyperspectral scene.

No download and no data files: a small scene with spatially contiguous class
regions is generated in-process, a few pixels per class are revealed as labels,
and Sw-SSFCM is run on it. Expected output (deterministic, seed 42):

    Softmax      ACC=0.74xx
    Sw-SSFCM r=2 ACC=0.92xx  (alpha=..., 3 iterations)

    python demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate                       # noqa: E402
from swssfcm import sw_ssfcm                        # noqa: E402


def standardize(X):
    """Zero-mean, unit-variance per band (population std, zero-std bands left alone)."""
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - X.mean(axis=0, keepdims=True)) / std


def synthetic_scene(height=64, width=64, n_bands=30, n_clusters=6,
                    noise=2.0, background_fraction=0.15, seed=42):
    """A synthetic HSI with the structure the spatial term exploits.

    Class regions come from a Voronoi tessellation of random seed points (spatially
    contiguous patches); each class has a smooth Gaussian-absorption spectrum and every
    pixel is that spectrum plus Gaussian noise. A random subset of pixels is left
    unlabelled (background) to mimic the partial ground truth of real scenes.
    Returns ``X (N, d)`` standardised row-major pixels and ``y_true (N,)`` with
    ``-1`` on background.
    """
    rng = np.random.default_rng(seed)
    seeds_yx = rng.integers(0, [height, width], size=(n_clusters * 3, 2))
    seed_class = np.arange(len(seeds_yx)) % n_clusters
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
    dist = ((grid[:, None, 0] - seeds_yx[None, :, 0]) ** 2
            + (grid[:, None, 1] - seeds_yx[None, :, 1]) ** 2)
    labels = seed_class[dist.argmin(axis=1)]

    band_axis = np.linspace(0.0, 1.0, n_bands)[None, :]
    centre = np.linspace(0.15, 0.85, n_clusters)[:, None]
    amplitude = rng.uniform(1.5, 3.0, size=(n_clusters, 1))
    offset = rng.uniform(-0.5, 0.5, size=(n_clusters, 1))
    signatures = offset + amplitude * np.exp(-((band_axis - centre) ** 2) / 0.02)

    X = signatures[labels] + rng.normal(0.0, noise, size=(height * width, n_bands))
    y_true = labels.astype(int)
    y_true[rng.random(height * width) < background_fraction] = -1
    return standardize(X), y_true, height, width, n_clusters


def stratified_labels(y_true, n_per_class, seed=42):
    """Reveal at most ``n_per_class`` labels per class (all of a smaller class), rest -1."""
    rng = np.random.default_rng(seed)
    y = np.full_like(y_true, -1)
    for cls in np.unique(y_true[y_true >= 0]):
        idx = np.where(y_true == cls)[0]
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        y[pick] = cls
    return y


def main():
    X, y_true, H, W, C = synthetic_scene(seed=42)
    y_lab = stratified_labels(y_true, n_per_class=10, seed=42)
    gt = (y_true >= 0) & (y_lab < 0)                    # ground-truth pixels without a label

    res = sw_ssfcm(X, y_lab, H, W, n_clusters=C, theta=0.99, r=2, seed=42,
                   softmax_kw=dict(epochs=300))         # short Softmax keeps the demo < 5 s
    softmax_acc = evaluate(y_true[gt], res["P"].argmax(1)[gt])["acc"]
    sw_acc = evaluate(y_true[gt], res["labels"][gt])["acc"]

    print(f"Softmax      ACC={softmax_acc:.4f}")
    print(f"Sw-SSFCM r=2 ACC={sw_acc:.4f}  "
          f"(alpha={res['alpha']:.1f}, {res['n_iter']} iterations)")
    print("label image:", res["labels"].reshape(H, W).shape,
          "memberships U:", res["U"].shape, "centroids V:", res["V"].shape)


if __name__ == "__main__":
    main()
