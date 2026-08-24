#!/usr/bin/env python3
"""Main experiment of the paper: 6 scenes × label budgets {5,10,20,40,60} per class × 10 seeds.

Per cell one Softmax is trained on the labelled pixels and shared by the four columns
    Softmax        argmax_k p_ik
    Sr-SSFCM          Sw-SSFCM with r = 0  (π = p)
    Sw-SSFCM r=1, Sw-SSFCM r=2                            (τ = 0.99, ω = 0.5, m = 2, ε = 1e-4)
Metrics: ACC / NMI / macro-F1 on the ground-truth pixels that carry no label ("unl", the
primary numbers of the paper) and on all ground-truth pixels ("all"); Xie–Beni; time.
The CSV layout is identical to the published run (reproduce/published/benchmark_budget.csv),
so make_tables.py works on either.

Usage
    python reproduce/benchmark_budget.py --data-root ~/data/HSI
    python reproduce/benchmark_budget.py --data-root ~/data/HSI --datasets botswana --budgets 60 --seeds 42
Resumable: existing (dataset, budget, seed, algo) rows are skipped.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import evaluate, xie_beni, centroids_from_u                  # noqa: E402
from reproduce.datasets import DATASET_KEYS, load_scene, stratified_labels  # noqa: E402
from swssfcm import THETA_DEFAULT, posterior, sw_ssfcm, theta_scales, train_softmax  # noqa: E402

BUDGETS = [5, 10, 20, 40, 60]
SEEDS = list(range(42, 52))
THETA, OMEGA, M, EPS, MAX_ITER = THETA_DEFAULT, 0.5, 2.0, 1e-4, 10000
SOFTMAX = dict(lr=0.01, l2=1e-4, epochs=10000, batch_size=64)
ALGOS = ["Softmax", "Sr-SSFCM", "Sw-SSFCM_r1", "Sw-SSFCM_r2"]
FIELDS = ["dataset", "budget", "seed", "algo", "theta", "alpha", "ratio", "share_g", "acc_unl",
          "nmi_unl", "f1_unl", "acc_all", "nmi_all", "xb", "iters", "time_s", "status", "note"]


def run_cell(scene, budget, seed, theta=THETA, epochs=SOFTMAX["epochs"]):
    """All four columns for one (scene, budget, seed).  Returns a list of row dicts."""
    X, y_true, H, W = scene.X, scene.y_true, scene.height, scene.width
    y_lab = stratified_labels(y_true, scene.valid_mask, budget, seed=seed)
    masks = {"all": y_true >= 0, "unl": (y_true >= 0) & (y_lab < 0)}
    lab = y_lab >= 0

    t0 = time.perf_counter()
    Wt, b = train_softmax(X[lab], y_lab[lab], seed=seed, **{**SOFTMAX, "epochs": epochs})
    P = posterior(X, Wt, b)
    t_soft = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc = theta_scales(X, y_lab, seed=seed)                    # θ-rule scales, once per cell
    t_theta = time.perf_counter() - t0

    def row(algo, labels, U, V, iters, t, theta_val="", alpha="", share=""):
        out = dict(dataset=scene.key, budget=budget, seed=seed, algo=algo, theta=theta_val,
                   alpha=alpha, ratio=sc["ratio"] if theta_val != "" else "", share_g=share,
                   iters=iters, time_s=round(t, 2), status="ok", note="")
        for name, mk in masks.items():
            ev = evaluate(y_true[mk], labels[mk])
            out[f"acc_{name}"], out[f"nmi_{name}"] = ev["acc"], ev["nmi"]
            if name == "unl":
                out["f1_unl"] = ev["f1"]
        out["xb"] = xie_beni(X, U, V if V is not None else centroids_from_u(X, U, M), M)
        return out

    rows = [row("Softmax", P.argmax(axis=1), P, None, "", t_soft)]
    for algo, r in (("Sr-SSFCM", 0), ("Sw-SSFCM_r1", 1), ("Sw-SSFCM_r2", 2)):
        t1 = time.perf_counter()
        res = sw_ssfcm(X, y_lab, H, W, n_clusters=scene.n_clusters, theta=theta, r=r,
                       omega=OMEGA, m=M, eps=EPS, max_iter=MAX_ITER, P=P, ratio=sc["ratio"], seed=seed)
        rows.append(row(algo, res["labels"], res["U"], res["V"], res["n_iter"],
                        t_soft + t_theta + time.perf_counter() - t1, theta, res["alpha"], res["share_g"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASET_KEYS, choices=DATASET_KEYS)
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--epochs", type=int, default=SOFTMAX["epochs"],
                    help="Softmax epochs (10000 in the paper; lower for a quick look)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "benchmark_budget.csv"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["dataset"], r["budget"], r["seed"], r["algo"]) for r in csv.DictReader(fh)}
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for key in args.datasets:
            scene = load_scene(key, args.data_root)
            print(f"[{scene.name}] {scene.height}x{scene.width}x{scene.n_bands}, C={scene.n_clusters}", flush=True)
            for budget in args.budgets:
                for seed in args.seeds:
                    if all((key, str(budget), str(seed), a) in done for a in ALGOS):
                        continue
                    t0 = time.perf_counter()
                    rows = run_cell(scene, budget, seed, args.theta, args.epochs)
                    for r in rows:
                        if (key, str(budget), str(seed), r["algo"]) not in done:
                            w.writerow(r)
                    fh.flush()
                    acc = {r["algo"]: r["acc_unl"] * 100 for r in rows}
                    print(f"  budget={budget:2d} seed={seed}  " +
                          "  ".join(f"{a}={acc[a]:.2f}" for a in ALGOS) +
                          f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"done → {out}")


if __name__ == "__main__":
    main()
