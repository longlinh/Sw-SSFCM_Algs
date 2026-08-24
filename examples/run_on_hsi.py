#!/usr/bin/env python3
"""Run Sw-SSFCM on one benchmark scene (or your own cube) and print the four columns.

    python examples/run_on_hsi.py --dataset botswana --data-root ~/data/HSI
    python examples/run_on_hsi.py --dataset botswana --data-root ~/data/HSI --budget 60 --seed 42 --epochs 10000
    python examples/run_on_hsi.py --cube my.mat:cube --gt my.mat:gt --data-root .
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce import benchmark_budget                                          # noqa: E402
from reproduce.datasets import DATASET_KEYS, Scene, load_scene, standardize, _load_mat  # noqa: E402


def load_custom(cube_spec, gt_spec, root):
    path, key = cube_spec.split(":")
    cube = _load_mat(Path(root) / path, key).astype(float)
    path, key = gt_spec.split(":")
    gt = _load_mat(Path(root) / path, key).astype(int).reshape(-1)
    H, W, B = cube.shape
    y = np.where(gt > 0, gt - 1, -1)
    return Scene("custom", "custom", standardize(cube.reshape(H * W, B)), y, y >= 0, H, W,
                 int(np.unique(y[y >= 0]).size))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", choices=DATASET_KEYS)
    ap.add_argument("--cube", help="file.mat:variable for your own cube (H, W, B)")
    ap.add_argument("--gt", help="file.mat:variable for your own ground truth (H, W), 0 = background")
    ap.add_argument("--budget", type=int, default=10, help="labels per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theta", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=10000)
    args = ap.parse_args(argv)

    scene = load_custom(args.cube, args.gt, args.data_root) if args.cube else load_scene(args.dataset, args.data_root)
    print(f"[load] {scene.name}: {scene.height}x{scene.width}x{scene.n_bands}, C={scene.n_clusters}, "
          f"{int(scene.valid_mask.sum()):,} labelled pixels")
    rows = benchmark_budget.run_cell(scene, args.budget, args.seed, args.theta, args.epochs)
    print(f"[eval] {args.budget} labels/class, seed {args.seed}, theta {args.theta} — ACC on unlabelled GT pixels")
    for r in rows:
        print(f"  {r['algo']:<12} ACC={r['acc_unl'] * 100:6.2f}%  NMI={r['nmi_unl']:.4f}  "
              f"F1={r['f1_unl']:.4f}  XB={r['xb']:.2f}  iters={r['iters']}  {r['time_s']}s")


if __name__ == "__main__":
    main()
