#!/usr/bin/env python3
"""Turn a tau-sweep result file into the benchmark tables reported in the paper.

Reads ``tau_sweep.json`` (produced by ``reproduce/tau_sweep.py``) and emits:

* ``benchmark_table.md``  -- ACC / NMI / ARI at the best tau, one row per scene,
  one block per variant, plus the per-variant average across scenes.
* ``benchmark_table.csv`` -- same content, machine readable.
* ``tau_star_table.md``   -- the selected (tau*, alpha*) per scene and variant.

Usage
-----
    python reproduce/make_tables.py
    python reproduce/make_tables.py --results reproduce/results --out-dir reproduce/results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANT_ORDER = ["SeFCM", "Sw-SSFCM_r1", "Sw-SSFCM_r2"]
VARIANT_LABEL = {
    "SeFCM": "SeFCM",
    "Sw-SSFCM_r1": "Sw-SSFCM (r=1)",
    "Sw-SSFCM_r2": "Sw-SSFCM (r=2)",
}


def load_summary(results_dir: Path) -> dict:
    path = results_dir / "tau_sweep.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'python reproduce/tau_sweep.py' first."
        )
    with open(path) as handle:
        return json.load(handle)["summary"]


def ordered_variants(summary: dict) -> list:
    """Variants present in the results, in the canonical paper order."""
    present = {v for scene in summary.values() for v in scene}
    return [v for v in VARIANT_ORDER if v in present]


def build_rows(summary: dict, variants: list) -> list:
    """One row per (scene, variant) with the metrics at the selected tau."""
    rows = []
    for scene, per_variant in summary.items():
        for variant in variants:
            if variant not in per_variant:
                continue
            best = per_variant[variant]
            rows.append(
                {
                    "dataset": scene,
                    "variant": variant,
                    "tau_star": best["tau_star"],
                    "alpha_star": best["alpha_star"],
                    "acc": best["acc"],
                    "nmi": best["nmi"],
                    "ari": best["ari"],
                    "seconds": best["seconds"],
                }
            )
    return rows


def averages(rows: list, variants: list) -> dict:
    """Mean ACC / NMI / ARI per variant across all scenes."""
    result = {}
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        result[variant] = {
            metric: sum(r[metric] for r in subset) / len(subset)
            for metric in ("acc", "nmi", "ari")
        }
    return result


def _markdown_table(header: list, body: list) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines) + "\n"


def write_benchmark_markdown(rows: list, variants: list, path: Path) -> None:
    scenes = list(dict.fromkeys(r["dataset"] for r in rows))
    lookup = {(r["dataset"], r["variant"]): r for r in rows}

    header = ["Scene"]
    for variant in variants:
        header += [f"{VARIANT_LABEL[variant]} ACC", "NMI", "ARI"]

    body = []
    for scene in scenes:
        cells = [scene]
        for variant in variants:
            row = lookup.get((scene, variant))
            if row is None:
                cells += ["--", "--", "--"]
            else:
                cells += [
                    f"{row['acc'] * 100:.2f}",
                    f"{row['nmi'] * 100:.2f}",
                    f"{row['ari'] * 100:.2f}",
                ]
        body.append(cells)

    means = averages(rows, variants)
    mean_cells = ["**Average**"]
    for variant in variants:
        stats = means.get(variant)
        if stats is None:
            mean_cells += ["--", "--", "--"]
        else:
            mean_cells += [
                f"**{stats['acc'] * 100:.2f}**",
                f"**{stats['nmi'] * 100:.2f}**",
                f"**{stats['ari'] * 100:.2f}**",
            ]
    body.append(mean_cells)

    text = (
        "# Benchmark at best tau (all values in %)\n\n"
        + _markdown_table(header, body)
    )
    path.write_text(text)


def write_tau_star_markdown(rows: list, path: Path) -> None:
    header = ["Scene", "Variant", "tau*", "alpha*", "ACC (%)", "Time (s)"]
    body = [
        [
            row["dataset"],
            VARIANT_LABEL[row["variant"]],
            f"{row['tau_star']:.2f}",
            f"{row['alpha_star']:.2f}",
            f"{row['acc'] * 100:.2f}",
            f"{row['seconds']:.1f}",
        ]
        for row in rows
    ]
    path.write_text("# Selected operating points\n\n" + _markdown_table(header, body))


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        default=str(here / "results"),
        help="directory containing tau_sweep.json",
    )
    parser.add_argument(
        "--out-dir", default=None, help="destination directory (default: --results)"
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(results_dir)
    variants = ordered_variants(summary)
    rows = build_rows(summary, variants)
    if not rows:
        raise SystemExit("tau_sweep.json contains no results")

    write_benchmark_markdown(rows, variants, out_dir / "benchmark_table.md")
    write_tau_star_markdown(rows, out_dir / "tau_star_table.md")
    write_csv(rows, out_dir / "benchmark_table.csv")

    print((out_dir / "benchmark_table.md").read_text())
    for name in ("benchmark_table.md", "benchmark_table.csv", "tau_star_table.md"):
        print(f"Wrote {out_dir / name}")


if __name__ == "__main__":
    main()
