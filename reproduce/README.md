# Reproducing the published results

This directory contains everything needed to regenerate the main experiment of
the paper: obtaining the data, sweeping the guidance strength, and producing the
benchmark tables.

## Contents

| File | Purpose |
|---|---|
| `download_data.py` | Fetch the benchmark scenes and verify them against SHA-256 checksums |
| `datasets.py` | Scene registry, loader, stratified label sampling, synthetic scene generator |
| `tau_sweep.py` | Sweep `τ` for SeFCM and Sw-SSFCM `r∈{1,2}`; the main experiment |
| `make_tables.py` | Turn sweep results into benchmark tables |
| `smoke_test.py` | End-to-end check of the whole pipeline on synthetic data, no download |

## Quick check first

Before committing to a long run, confirm the pipeline works. This needs no data
and takes under ten seconds:

```bash
python reproduce/smoke_test.py
```

It generates a synthetic scene, runs the full pipeline on it, writes all output
files, and asserts three things: a repeated fit with the same seed is
bit-identical, softmax guidance beats no guidance, and the spatial term helps.
It ends with `SMOKE TEST PASSED`.

---

## 1. Get the data

```bash
python reproduce/download_data.py --data-root ~/data/HSI
```

Six of the eight scenes download automatically. Two require a free registration
with their providers and cannot be redistributed here; the script prints the
exact steps and the filenames to save.

| Scene | Size | Pixels | Bands | Classes | Access |
|---|---|---|---|---|---|
| Indian Pines | 6 MB | 21 025 | 200 | 16 | open |
| Salinas | 27 MB | 111 104 | 204 | 16 | open |
| Pavia University | 35 MB | 207 400 | 103 | 9 | open |
| Pavia Centre | 130 MB | 783 640 | 102 | 9 | open |
| Kennedy Space Center | 57 MB | 314 368 | 176 | 13 | open |
| Botswana | 79 MB | 377 856 | 145 | 14 | open |
| Houston 2013 | 169 MB | 664 845 | 144 | 15 | registration |
| WHU-Hi-LongKou | 186 MB | 220 000 | 270 | 9 | registration |

Open scenes come from the Computational Intelligence Group at the University of
the Basque Country:
<https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>

Houston 2013 is distributed by the IEEE GRSS Data Fusion Contest
(<https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/>) and
WHU-Hi-LongKou by the RSIDEA group at Wuhan University
(<http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm>).

Verify at any time — this confirms the local files are bit-identical to the
copies used for the published numbers:

```bash
python reproduce/download_data.py --data-root ~/data/HSI --verify
```

If the download server blocks automated requests, fetch the files by hand from
the source pages and place them under the paths listed in `datasets.py`; the
`--verify` run will then confirm they are correct.

### No data at all?

Every script accepts `--synthetic`, which generates a scene in memory with the
statistical structure of a real HSI (contiguous class regions, smooth per-class
spectra, additive noise, unlabelled background). The pipeline is identical, so
the code paths are fully exercised:

```bash
python reproduce/tau_sweep.py --synthetic
```

---

## 2. Run the sweep

```bash
python reproduce/tau_sweep.py --data-root ~/data/HSI
```

This is the main experiment. For each scene it samples 60 labelled pixels per
class, then fits SeFCM, Sw-SSFCM `r=1` and Sw-SSFCM `r=2` at each `τ` in
`{0.1, …, 0.9, 0.95}`, recording ACC, NMI, ARI and wall-clock time. Output goes
to `reproduce/results/tau_sweep.{csv,json}`.

Sweeping each variant separately is required, not redundant: the spatial term
shifts the optimum, so `τ*` differs between variants.

Useful for shorter runs:

```bash
# one scene
python reproduce/tau_sweep.py --data-root ~/data/HSI --datasets indian_pines

# coarse grid, one variant
python reproduce/tau_sweep.py --data-root ~/data/HSI --datasets salinas \
    --tau 0.3 0.6 0.9 --variants Sw-SSFCM_r2
```

### Fixed protocol

Hard-coded at the top of `tau_sweep.py`; change these and the numbers will not
match the paper.

| Parameter | Value |
|---|---|
| seed | 42 |
| labelled pixels per class | 60, stratified, capped at class size |
| fuzzifier `m` | 2.0 |
| convergence threshold `ε` | 1e-4 |
| max FCM iterations | 10 000 |
| softmax learning rate `γ` | 0.01 |
| softmax L2 `λ` | 1e-4 |
| softmax epochs `T_s` | 10 000, no early stopping |
| feature scaling | per-band zero-mean, unit-variance |
| `τ` grid | 0.1 … 0.9, 0.95 |

> **On the softmax epoch count.** `tab-params.tex` in the manuscript lists
> `T_s = 1000`. The runs that produced the published numbers used 10 000 epochs
> with early stopping disabled, and that is what this directory reproduces. At
> 1 000 epochs accuracy drops by roughly 2–3 points across the board.

---

## 3. Generate the tables

```bash
python reproduce/make_tables.py
```

Reads `reproduce/results/tau_sweep.json` and writes, next to it:

- `benchmark_table.md` — ACC / NMI / ARI at the best `τ`, per scene and variant,
  with the per-variant average across scenes.
- `benchmark_table.csv` — the same, machine readable.
- `tau_star_table.md` — the selected `(τ*, α*)` per scene and variant.

---

## Computational requirements

Everything here is pure NumPy on the CPU. There is no GPU requirement and no
GPU code path.

### Hardware

| Resource | Requirement |
|---|---|
| CPU | Any x86-64 or ARM64 processor. NumPy's BLAS uses all cores for the distance computation. |
| GPU | Not used. |
| RAM | Peak usage is roughly `8·N·(D + 8C)` bytes plus the raw cube during loading. 4 GB is enough for the six smaller scenes; 16 GB is comfortable for Pavia Centre and Houston 2013, the two largest. |
| Disk | About 700 MB for all eight scenes; results are a few hundred kB. |

### Runtime

Measured on one CPU core of a desktop workstation, full protocol
(`T_s = 10 000`, `τ = 0.9`, 60 labels per class). Timings scale roughly linearly
in the pixel count and in `τ`-grid length.

| Scene | Pixels | SeFCM, one fit | Sw-SSFCM `r=2`, one fit |
|---|---|---|---|
| Indian Pines | 21 025 | ~17 s | ~104 s |
| Salinas | 111 104 | ~50 s | ~138 s |

A full `τ` sweep is 10 grid points × 3 variants = 30 fits per scene. Budget a
few hours per large scene, and roughly a day of CPU time for all eight; run
scenes separately with `--datasets` rather than in one session.

The softmax regressor dominates the cost on scenes with few labelled samples,
because it runs 10 000 epochs without early stopping. While exploring, either
pass `--max-iter 200` or construct estimators with `softmax_tol=1e-6`, which
cuts the fit time by about an order of magnitude at a cost of 2–3 accuracy
points.

### Software

Python 3.9+ with `numpy`, `scipy` and `scikit-learn` (see `../requirements.txt`).
No compilation step and no optional accelerators.

---

## Relation to the published numbers

The estimators in this repository are a standalone, dependency-light NumPy
reference implementation written to match the pseudocode in the paper
line-for-line. The published experiments were run with the authors' internal
GPU-accelerated research library, which implements the same equations with
float32 arithmetic and a different random-shuffle stream.

Under the protocol above, this implementation reproduces the published results
to within a few tenths of an accuracy point, and reproduces the selected `τ*`
and the ranking of the three variants exactly. Two verified examples:

| Scene / variant | Published | This repository |
|---|---|---|
| Indian Pines, SeFCM, `τ = 0.90` | 71.32 % | 71.22 % |
| Salinas, Sw-SSFCM `r=2`, `τ = 0.90` | 89.55 % | 89.70 % |

Residual differences come from float32 vs float64 accumulation and from the
order in which mini-batches are drawn during softmax training. They do not
affect any conclusion in the paper.

The baseline methods in the comparison table (FCM, KFCM, SSFCM, eSFCM, S3FCM,
SMUC, S2-PFCM, GS-SPFCM) are previously published algorithms by other authors
and are not reimplemented here; consult the citations in the manuscript for
their original descriptions. Setting `alpha=0` gives an unguided FCM reference
point using the same code path, which is what `smoke_test.py` compares against.
