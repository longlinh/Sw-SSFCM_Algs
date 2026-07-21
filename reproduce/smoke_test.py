#!/usr/bin/env python3
"""End-to-end check of the reproduction pipeline on synthetic data.

Runs the same code path as the paper experiment -- scene loading, stratified
label sampling, tau-sweep over SeFCM and both Sw-SSFCM radii, best-tau
selection, table generation -- on a synthetic scene generated in memory, so it
needs no download and finishes in well under two minutes on a laptop CPU.

It also asserts what the pipeline must satisfy on any sane data: a repeated fit
with the same seed is bit-identical, softmax guidance beats no guidance, and the
spatial term helps on a spatially structured scene.

Usage
-----
    python reproduce/smoke_test.py
    python reproduce/smoke_test.py --keep   # leave the results on disk
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import evaluate                                        # noqa: E402
from reproduce import make_tables, tau_sweep                        # noqa: E402
from reproduce.datasets import make_synthetic_scene, stratified_labels  # noqa: E402

# Reduced protocol: enough tau values to exercise selection, small enough to
# stay fast. The full grid lives in tau_sweep.TAU_GRID.
SMOKE_TAU_GRID = [0.3, 0.6, 0.9]
SMOKE_MAX_ITER = 200
SMOKE_LABELS_PER_CLASS = 20


def check_no_guidance_baseline(scene, y_partial) -> float:
    """Fit with alpha=0, i.e. FCM initialized from the labelled class means."""
    labels, _, _ = tau_sweep.fit_variant(
        scene, y_partial, "SeFCM", alpha=0.0, max_iter=SMOKE_MAX_ITER
    )
    return evaluate(scene.y_true[scene.valid_mask], labels[scene.valid_mask])["acc"]


def check_determinism(scene, y_partial) -> bool:
    """Two fits with the same seed must produce identical hard labels."""
    alpha = tau_sweep.tau_to_alpha(0.6, scene.n_bands, scene.n_clusters)
    first, _, _ = tau_sweep.fit_variant(
        scene, y_partial, "Sw-SSFCM_r1", alpha, SMOKE_MAX_ITER
    )
    second, _, _ = tau_sweep.fit_variant(
        scene, y_partial, "Sw-SSFCM_r1", alpha, SMOKE_MAX_ITER
    )
    return bool(np.array_equal(first, second))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="write results next to the script instead of a temporary directory",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    scene = make_synthetic_scene(seed=tau_sweep.SEED)
    print(
        f"Synthetic scene: {scene.height}x{scene.width}x{scene.n_bands}, "
        f"C={scene.n_clusters}, {int(scene.valid_mask.sum()):,} labelled-capable pixels"
    )

    y_partial = stratified_labels(
        scene.y_true, scene.valid_mask, SMOKE_LABELS_PER_CLASS, seed=tau_sweep.SEED
    )
    acc_no_guidance = check_no_guidance_baseline(scene, y_partial)
    print(f"Reference (alpha=0, no guidance): ACC={acc_no_guidance * 100:.2f}%")

    deterministic = check_determinism(scene, y_partial)
    print(f"Repeated fit with seed {tau_sweep.SEED} is identical: {deterministic}")

    rows = tau_sweep.sweep_scene(
        scene,
        SMOKE_TAU_GRID,
        SMOKE_LABELS_PER_CLASS,
        SMOKE_MAX_ITER,
        tau_sweep.VARIANTS,
    )
    summary = tau_sweep.summarize(rows)
    tau_sweep.print_summary(summary)

    out_dir = (
        Path(__file__).resolve().parent / "results"
        if args.keep
        else Path(tempfile.mkdtemp(prefix="swssfcm-smoke-"))
    )
    tau_sweep.write_outputs(rows, summary, out_dir)
    make_tables.main(["--results", str(out_dir)])

    best = summary["synthetic"]
    acc_sefcm = best["SeFCM"]["acc"]
    acc_r1 = best["Sw-SSFCM_r1"]["acc"]
    acc_r2 = best["Sw-SSFCM_r2"]["acc"]

    failures = []
    if not deterministic:
        failures.append("repeated fit with the same seed produced different labels")
    if acc_sefcm <= acc_no_guidance:
        failures.append(
            f"softmax guidance did not help: SeFCM {acc_sefcm:.4f} "
            f"<= alpha=0 {acc_no_guidance:.4f}"
        )
    if max(acc_r1, acc_r2) < acc_sefcm:
        failures.append(
            f"spatial weighting did not help: best Sw-SSFCM "
            f"{max(acc_r1, acc_r2):.4f} < SeFCM {acc_sefcm:.4f}"
        )
    for name in ("tau_sweep.csv", "tau_sweep.json", "benchmark_table.md",
                 "benchmark_table.csv", "tau_star_table.md"):
        if not (out_dir / name).is_file():
            failures.append(f"missing output file {name}")

    if not args.keep:
        shutil.rmtree(out_dir, ignore_errors=True)

    elapsed = time.perf_counter() - started
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
    if failures:
        print("\nSMOKE TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
