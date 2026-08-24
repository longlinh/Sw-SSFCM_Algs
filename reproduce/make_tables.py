#!/usr/bin/env python3
"""Benchmark tables from a benchmark_budget.csv (yours or the published one).

For every budget: mean ± std of unl-only ACC (%) over seeds, per scene and column, the
average over scenes, and the paired difference Sw-SSFCM r=2 − Softmax with
the number of (scene, seed) pairs won.  Writes <results>/tables.md and <results>/summary.csv.

Usage
    python reproduce/make_tables.py --results reproduce/results
    python reproduce/make_tables.py --csv reproduce/published/benchmark_budget.csv --results /tmp/tables
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGOS = ["Softmax", "Sr-SSFCM", "Sw-SSFCM_r1", "Sw-SSFCM_r2"]
METRIC = "acc_unl"


def load(csv_path):
    cells = defaultdict(dict)                     # (dataset, budget, seed) -> {algo: acc}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("status", "ok") != "ok" or r["algo"] not in ALGOS:
                continue
            cells[(r["dataset"], int(r["budget"]), int(r["seed"]))][r["algo"]] = float(r[METRIC]) * 100
    return cells


def summarise(cells):
    """-> {budget: {dataset: {algo: (mean, std, n)}}}, and paired deltas per budget."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pairs = defaultdict(lambda: defaultdict(list))      # budget -> ref -> [delta]
    for (ds, budget, _seed), accs in cells.items():
        for algo, v in accs.items():
            by[budget][ds][algo].append(v)
        if "Sw-SSFCM_r2" in accs:
            for ref in ("Softmax",):
                if ref in accs:
                    pairs[budget][ref].append(accs["Sw-SSFCM_r2"] - accs[ref])
    table = {b: {ds: {a: (float(np.mean(v)), float(np.std(v)), len(v)) for a, v in algos.items()}
                 for ds, algos in dss.items()} for b, dss in by.items()}
    return table, pairs


def render(table, pairs):
    lines = ["# Benchmark — unl-only ACC (%), mean ± std over seeds", ""]
    for budget in sorted(table):
        dss = table[budget]
        lines += [f"## {budget} labels per class", "",
                  "| scene | " + " | ".join(ALGOS) + " |",
                  "|---|" + "---|" * len(ALGOS)]
        avg = defaultdict(list)
        for ds in sorted(dss):
            cells = []
            for a in ALGOS:
                if a in dss[ds]:
                    m, s, _ = dss[ds][a]
                    cells.append(f"{m:.2f} ± {s:.2f}")
                    avg[a].append(m)
                else:
                    cells.append("—")
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("| **average** | " + " | ".join(
            f"**{np.mean(avg[a]):.2f}**" if avg[a] else "—" for a in ALGOS) + " |")
        for ref, d in pairs.get(budget, {}).items():
            d = np.asarray(d)
            lines.append(f"- Sw-SSFCM r=2 − {ref}: mean {d.mean():+.2f} points, "
                         f"wins {(d > 0).sum()}/{len(d)} pairs, ties {(d == 0).sum()}, "
                         f"losses {(d < 0).sum()}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--csv", default=None, help="input CSV (default <results>/benchmark_budget.csv)")
    args = ap.parse_args(argv)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    src = Path(args.csv) if args.csv else results / "benchmark_budget.csv"
    table, pairs = summarise(load(src))
    md = render(table, pairs)
    (results / "tables.md").write_text(md, encoding="utf-8")
    with open(results / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["budget", "dataset", "algo", "acc_unl_mean", "acc_unl_std", "n_seeds"])
        for b in sorted(table):
            for ds in sorted(table[b]):
                for a in ALGOS:
                    if a in table[b][ds]:
                        m, s, n = table[b][ds][a]
                        w.writerow([b, ds, a, f"{m:.4f}", f"{s:.4f}", n])
    print(md)
    print(f"written: {results / 'tables.md'}, {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
