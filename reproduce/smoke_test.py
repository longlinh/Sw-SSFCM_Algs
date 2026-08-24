#!/usr/bin/env python3
"""End-to-end check on a synthetic scene — no download, under a minute on a laptop CPU.

Runs the exact code path of the paper experiment (scene → stratified labels → shared
Softmax → Softmax / Sr-SSFCM / Sw-SSFCM r=1,2 → tables) and asserts what must
hold on any spatially structured scene:
  1. a repeated fit with the same seed is bit-identical;
  2. Sr-SSFCM reproduces the Softmax labels (π = p, τ = 0.99 ⇒ argmax u = argmax p);
  3. Sw-SSFCM r=2 improves on Softmax.

    python reproduce/smoke_test.py [--keep]
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce import benchmark_budget, make_tables                     # noqa: E402
from reproduce.datasets import make_synthetic_scene, stratified_labels  # noqa: E402
from swssfcm import sw_ssfcm                                            # noqa: E402

SEED, BUDGET, EPOCHS = 42, 10, 300


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", action="store_true", help="write results to reproduce/results")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    scene = make_synthetic_scene(seed=SEED)
    print(f"Synthetic scene {scene.height}x{scene.width}x{scene.n_bands}, C={scene.n_clusters}")
    y_lab = stratified_labels(scene.y_true, scene.valid_mask, BUDGET, seed=SEED)
    kw = dict(n_clusters=scene.n_clusters, seed=SEED, softmax_kw=dict(epochs=EPOCHS))
    a = sw_ssfcm(scene.X, y_lab, scene.height, scene.width, r=2, **kw)
    b = sw_ssfcm(scene.X, y_lab, scene.height, scene.width, r=2, **kw)
    deterministic = np.array_equal(a["labels"], b["labels"])
    print(f"deterministic: {deterministic}   (Sw-SSFCM r=2 converged in {a['n_iter']} iterations)")

    rows = benchmark_budget.run_cell(scene, BUDGET, SEED, epochs=EPOCHS)
    acc = {r["algo"]: r["acc_unl"] * 100 for r in rows}
    for r in rows:
        print(f"  {r['algo']:<12} ACC_unl={r['acc_unl'] * 100:6.2f}  NMI={r['nmi_unl']:.3f}  "
              f"XB={r['xb']:.2f}  iters={r['iters']}  {r['time_s']}s")

    out = Path(__file__).resolve().parent / "results" if args.keep else Path(tempfile.mkdtemp(prefix="swssfcm-"))
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "benchmark_budget.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=benchmark_budget.FIELDS)
        w.writeheader()
        w.writerows(rows)
    make_tables.main(["--results", str(out)])

    failures = []
    if not deterministic:
        failures.append("repeated fit with the same seed differs")
    if abs(acc["Sr-SSFCM"] - acc["Softmax"]) > 0.5:
        failures.append(f"Sr-SSFCM {acc['Sr-SSFCM']:.2f} != Softmax {acc['Softmax']:.2f}")
    if acc["Sw-SSFCM_r2"] <= acc["Softmax"]:
        failures.append(f"Sw-SSFCM r=2 {acc['Sw-SSFCM_r2']:.2f} <= Softmax {acc['Softmax']:.2f}")
    for name in ("benchmark_budget.csv", "tables.md", "summary.csv"):
        if not (out / name).is_file():
            failures.append(f"missing {name}")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)

    print(f"\nwall-clock {time.perf_counter() - t0:.1f}s")
    if failures:
        print("SMOKE TEST FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
