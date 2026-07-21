#!/usr/bin/env python3
"""Per-variant tau-sweep reproducing the main experiment of the paper.

For every scene and every value of the normalized guidance strength

    tau = alpha * ln(C) / (d + alpha * ln(C))    <=>    alpha = tau / (1 - tau) * d / ln(C)

the script fits SeFCM, Sw-SSFCM (r=1) and Sw-SSFCM (r=2) and records ACC, NMI,
ARI and wall-clock time. The best tau per scene and per variant is the operating
point reported in the paper; a separate sweep per variant is required because
the spatial term shifts the optimum.

Usage
-----
    # Synthetic scene, no download needed (a couple of minutes)
    python reproduce/tau_sweep.py --synthetic

    # One real scene
    python reproduce/tau_sweep.py --data-root ~/data/HSI --datasets indian_pines

    # Full paper protocol, all eight scenes
    python reproduce/tau_sweep.py --data-root ~/data/HSI

Outputs ``tau_sweep.csv`` and ``tau_sweep.json`` under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import evaluate                                        # noqa: E402
from reproduce.datasets import (                                    # noqa: E402
    DATASET_KEYS,
    Scene,
    load_scene,
    make_synthetic_scene,
    stratified_labels,
)
from sefcm import SeFCM                                             # noqa: E402
from sw_ssfcm import SwSSFCM                                        # noqa: E402

# Fixed experimental protocol of the paper.
SEED = 42
FUZZIFIER = 2.0
TOLERANCE = 1e-4
MAX_ITER = 10000
SOFTMAX_LR = 0.01
SOFTMAX_L2 = 1e-4
SOFTMAX_MAX_EPOCH = 10000
SOFTMAX_TOL = 0.0  # 0 disables early stopping, matching the published runs
LABELS_PER_CLASS = 60
TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

VARIANTS = ["SeFCM", "Sw-SSFCM_r1", "Sw-SSFCM_r2"]

CSV_FIELDS = [
    "dataset",
    "variant",
    "tau",
    "alpha",
    "acc",
    "nmi",
    "ari",
    "n_iter",
    "seconds",
    "n_bands",
    "n_clusters",
    "n_pixels",
]


def tau_to_alpha(tau: float, n_bands: int, n_clusters: int) -> float:
    """Invert the normalization tau = alpha*ln(C) / (d + alpha*ln(C))."""
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must lie in [0, 1), got {tau}")
    return (tau / (1.0 - tau)) * (n_bands / np.log(n_clusters))


def fit_variant(
    scene: Scene,
    y_partial: np.ndarray,
    variant: str,
    alpha: float,
    max_iter: int,
) -> tuple:
    """Fit one variant and return ``(labels, n_iter, seconds)``."""
    common = dict(
        n_clusters=scene.n_clusters,
        alpha=alpha,
        m=FUZZIFIER,
        max_iter=max_iter,
        tol=TOLERANCE,
        softmax_lr=SOFTMAX_LR,
        softmax_l2=SOFTMAX_L2,
        softmax_max_epoch=SOFTMAX_MAX_EPOCH,
        softmax_tol=SOFTMAX_TOL,
        random_state=SEED,
    )
    start = time.perf_counter()
    if variant == "SeFCM":
        model = SeFCM(**common).fit(scene.X, y_partial)
    elif variant.startswith("Sw-SSFCM_r"):
        radius = int(variant.rsplit("r", 1)[1])
        model = SwSSFCM(radius=radius, **common).fit(
            scene.X, y_partial, image_shape=(scene.height, scene.width)
        )
    else:
        raise ValueError(f"unknown variant '{variant}'")
    return model.labels_, model.n_iter_, time.perf_counter() - start


def sweep_scene(
    scene: Scene,
    tau_grid: list,
    labels_per_class: int,
    max_iter: int,
    variants: list,
) -> list:
    """Run the full tau grid on one scene. Returns one row per (variant, tau)."""
    y_partial = stratified_labels(
        scene.y_true, scene.valid_mask, labels_per_class, seed=SEED
    )
    n_labelled = int(np.sum(y_partial >= 0))
    print(
        f"\n=== {scene.name} === {scene.height}x{scene.width}x{scene.n_bands}, "
        f"C={scene.n_clusters}, labelled={n_labelled:,} / "
        f"{int(scene.valid_mask.sum()):,} valid pixels"
    )

    y_eval = scene.y_true[scene.valid_mask]
    rows = []
    for tau in tau_grid:
        alpha = tau_to_alpha(tau, scene.n_bands, scene.n_clusters)
        line = [f"  tau={tau:<5.2f} alpha={alpha:>10.2f}"]
        for variant in variants:
            labels, n_iter, seconds = fit_variant(
                scene, y_partial, variant, alpha, max_iter
            )
            scores = evaluate(y_eval, labels[scene.valid_mask])
            rows.append(
                {
                    "dataset": scene.key,
                    "variant": variant,
                    "tau": tau,
                    "alpha": round(alpha, 4),
                    "acc": round(scores["acc"], 4),
                    "nmi": round(scores["nmi"], 4),
                    "ari": round(scores["ari"], 4),
                    "n_iter": n_iter,
                    "seconds": round(seconds, 1),
                    "n_bands": scene.n_bands,
                    "n_clusters": scene.n_clusters,
                    "n_pixels": scene.n_pixels,
                }
            )
            line.append(f"{variant} ACC={scores['acc'] * 100:>6.2f}% ({seconds:>5.1f}s)")
        print(" | ".join(line))
    return rows


def summarize(rows: list) -> dict:
    """Pick the best tau per (dataset, variant) by accuracy."""
    best: dict = {}
    for row in rows:
        key = (row["dataset"], row["variant"])
        if key not in best or row["acc"] > best[key]["acc"]:
            best[key] = row
    summary: dict = {}
    for (dataset, variant), row in best.items():
        summary.setdefault(dataset, {})[variant] = {
            "tau_star": row["tau"],
            "alpha_star": row["alpha"],
            "acc": row["acc"],
            "nmi": row["nmi"],
            "ari": row["ari"],
            "seconds": row["seconds"],
        }
    return summary


def write_outputs(rows: list, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tau_sweep.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "tau_sweep.json"
    with open(json_path, "w") as handle:
        json.dump({"summary": summary, "tracking": rows}, handle, indent=2)
    print(f"\nWrote {csv_path}\nWrote {json_path}")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("Best operating point per scene and variant")
    print("=" * 78)
    print(f"{'Scene':<20} {'Variant':<14} {'tau*':>6} {'alpha*':>10} {'ACC':>8} {'NMI':>8}")
    print("-" * 78)
    for dataset, variants in summary.items():
        for variant, best in variants.items():
            print(
                f"{dataset:<20} {variant:<14} {best['tau_star']:>6.2f} "
                f"{best['alpha_star']:>10.2f} {best['acc'] * 100:>7.2f}% "
                f"{best['nmi'] * 100:>7.2f}%"
            )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        default=None,
        help="directory holding the downloaded HSI scenes",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DATASET_KEYS,
        help=f"scenes to sweep (default: all of {DATASET_KEYS})",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="sweep the built-in synthetic scene instead of real data",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="variants to evaluate (default: all three)",
    )
    parser.add_argument(
        "--tau",
        nargs="+",
        type=float,
        default=TAU_GRID,
        help=f"tau grid (default: {TAU_GRID})",
    )
    parser.add_argument(
        "--labels-per-class",
        type=int,
        default=LABELS_PER_CLASS,
        help=f"labelled pixels sampled per class (default: {LABELS_PER_CLASS})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help=f"maximum FCM iterations (default: {MAX_ITER})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="destination directory for tau_sweep.csv / tau_sweep.json",
    )
    args = parser.parse_args(argv)
    if not args.synthetic and args.data_root is None:
        parser.error("either --synthetic or --data-root is required")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.synthetic:
        scenes = [make_synthetic_scene(seed=SEED)]
    else:
        keys = args.datasets or DATASET_KEYS
        scenes = [load_scene(key, args.data_root) for key in keys]

    print(f"tau grid      : {args.tau}")
    print(f"labels/class  : {args.labels_per_class}")
    print(f"seed          : {SEED}")
    print(f"variants      : {args.variants}")

    rows = []
    for scene in scenes:
        rows.extend(
            sweep_scene(
                scene, args.tau, args.labels_per_class, args.max_iter, args.variants
            )
        )

    summary = summarize(rows)
    print_summary(summary)
    write_outputs(rows, summary, Path(args.out_dir))


if __name__ == "__main__":
    main()
