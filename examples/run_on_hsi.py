#!/usr/bin/env python3
"""Run Sw-SSFCM on one hyperspectral scene.

Two ways to point the script at data:

1. By registered scene name, for any of the eight benchmarks of the paper. The
   file layout and MATLAB variable names are taken from ``reproduce/datasets.py``::

       python examples/run_on_hsi.py --dataset indian_pines --data-root ~/data/HSI

2. By explicit file paths, for your own scene. Provide the cube file and the
   ground-truth file plus the MATLAB variable name inside each. The cube must be
   ``(H, W, B)`` and the ground truth ``(H, W)`` with 0 marking background::

       python examples/run_on_hsi.py \
           --cube my_scene.mat --cube-key data \
           --gt my_scene_gt.mat --gt-key labels \
           --n-clusters 12

   Pass ``--gt`` the same path as ``--cube`` when both live in one file.

The script samples a stratified subset of labelled pixels, fits Sw-SSFCM and
reports ACC, NMI and ARI over the pixels that carry ground truth.

Sensible starting point for alpha: use ``--tau`` instead, which is scale-free
across scenes; the script converts it with alpha = tau/(1-tau) * d/ln(C).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the repository root importable when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import evaluate                                          # noqa: E402
from reproduce.datasets import (                                      # noqa: E402
    DATASET_KEYS,
    Scene,
    load_scene,
    standardize,
    stratified_labels,
)
from sw_ssfcm import SwSSFCM                                          # noqa: E402

SEED = 42


def load_custom_scene(
    cube_path: str, cube_key: str, gt_path: str, gt_key: str, n_clusters: int | None
) -> Scene:
    """Build a Scene from an arbitrary pair of MATLAB files."""
    from scipy.io import loadmat

    cube = np.asarray(loadmat(cube_path)[cube_key]).astype(np.float64)
    if cube.ndim != 3:
        raise ValueError(f"expected a 3-D cube (H, W, B), got shape {cube.shape}")
    height, width, bands = cube.shape

    gt = np.asarray(loadmat(gt_path)[gt_key]).astype(int)
    if gt.shape != (height, width):
        raise ValueError(f"cube is {height}x{width} but ground truth is {gt.shape}")
    gt = gt.reshape(-1)

    valid_mask = gt > 0
    y_true = np.full(gt.shape, -1, dtype=int)
    y_true[valid_mask] = gt[valid_mask] - 1

    return Scene(
        key="custom",
        name=Path(cube_path).stem,
        X=standardize(cube.reshape(height * width, bands)),
        y_true=y_true,
        valid_mask=valid_mask,
        height=height,
        width=width,
        n_clusters=n_clusters or int(np.unique(y_true[valid_mask]).size),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", choices=DATASET_KEYS, help="registered scene name")
    parser.add_argument("--data-root", help="directory holding registered scenes")
    parser.add_argument("--cube", help="path to a MATLAB file holding the (H, W, B) cube")
    parser.add_argument("--cube-key", default="data", help="variable name of the cube")
    parser.add_argument("--gt", help="path to a MATLAB file holding the (H, W) labels")
    parser.add_argument("--gt-key", default="labels", help="variable name of the labels")
    parser.add_argument(
        "--n-clusters", type=int, help="number of classes (default: inferred)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.9,
        help="normalized guidance strength in [0, 1) (default: 0.9)",
    )
    parser.add_argument("--radius", type=int, default=2, choices=(1, 2))
    parser.add_argument("--labels-per-class", type=int, default=60)
    parser.add_argument("--max-iter", type=int, default=10000)
    args = parser.parse_args(argv)

    if args.dataset:
        if not args.data_root:
            parser.error("--dataset requires --data-root")
    elif not (args.cube and args.gt):
        parser.error("provide either --dataset or both --cube and --gt")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.dataset:
        scene = load_scene(args.dataset, args.data_root)
    else:
        scene = load_custom_scene(
            args.cube, args.cube_key, args.gt, args.gt_key, args.n_clusters
        )

    print(
        f"[load] {scene.name}: {scene.height}x{scene.width}x{scene.n_bands}, "
        f"C={scene.n_clusters}, {int(scene.valid_mask.sum()):,} labelled pixels"
    )

    y_partial = stratified_labels(
        scene.y_true, scene.valid_mask, args.labels_per_class, seed=SEED
    )
    alpha = (args.tau / (1.0 - args.tau)) * (scene.n_bands / np.log(scene.n_clusters))
    print(
        f"[fit ] Sw-SSFCM(tau={args.tau}, alpha={alpha:.2f}, radius={args.radius}) "
        f"on {int(np.sum(y_partial >= 0)):,} labelled samples"
    )

    started = time.perf_counter()
    model = SwSSFCM(
        n_clusters=scene.n_clusters,
        alpha=alpha,
        radius=args.radius,
        max_iter=args.max_iter,
        random_state=SEED,
    ).fit(scene.X, y_partial, image_shape=(scene.height, scene.width))
    elapsed = time.perf_counter() - started
    print(f"       converged in {model.n_iter_} iterations, {elapsed:.1f}s")

    scores = evaluate(scene.y_true[scene.valid_mask], model.labels_[scene.valid_mask])
    print(
        f"[eval] ACC={scores['acc'] * 100:.2f}%  NMI={scores['nmi'] * 100:.2f}%  "
        f"ARI={scores['ari'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
