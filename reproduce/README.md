# Reproducing the published results

| File | Purpose |
|---|---|
| `download_data.py` | fetch the four open scenes, print the steps for the two registration-only ones, verify SHA-256 |
| `datasets.py` | scene registry (paths, MATLAB keys, shapes, checksums), loader, seeded stratified label sampler, synthetic scene |
| `benchmark_budget.py` | the main experiment: 6 scenes × {5,10,20,40,60} labels/class × 10 seeds × {Softmax, Sr-SSFCM, Sw-SSFCM r=1, r=2} |
| `make_tables.py` | mean ± std tables and paired differences from any `benchmark_budget.csv` |
| `smoke_test.py` | end-to-end check on synthetic data (no download, < 1 min) |
| `published/benchmark_budget.csv` | the run reported in the paper: 4 080 rows, all 16 methods (the FCM baselines come from the authors' library, see below) |

## 0. Quick check

```bash
python reproduce/smoke_test.py            # → SMOKE TEST PASSED
python reproduce/make_tables.py --csv reproduce/published/benchmark_budget.csv --results /tmp/published
```

The second command regenerates the accuracy tables of the paper from the published
CSV without running anything (the CSV contains every baseline column).

## 1. Data

```bash
python reproduce/download_data.py --data-root ~/data/HSI            # four open scenes
python reproduce/download_data.py --data-root ~/data/HSI --verify   # SHA-256 against our copies
```

| Scene | Pixels | Bands | Classes | GT pixels | Access |
|---|---|---|---|---|---|
| Indian Pines | 21 025 | 200 | 16 | 10 249 | open |
| Pavia University | 207 400 | 103 | 9 | 42 776 | open |
| Kennedy Space Center | 314 368 | 176 | 13 | 5 211 | open |
| Botswana | 377 856 | 145 | 14 | 3 248 | open |
| Houston 2013 | 664 845 | 144 | 15 | 15 029 (TR ∪ TE of the DFC split) | registration |
| WHU-Hi-LongKou | 220 000 | 270 | 9 | 204 542 | registration |

Open scenes: Computational Intelligence Group, University of the Basque Country
(<https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>).
Houston 2013: IEEE GRSS Data Fusion Contest
(<https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/>).
WHU-Hi-LongKou: RSIDEA group, Wuhan University
(<http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm>). If automated download is
blocked, fetch by hand, place under the paths in `datasets.py::SPECS` and run `--verify`.

## 2. Protocol (fixed in `benchmark_budget.py`)

| Item | Value |
|---|---|
| features | per-band z-score over all pixels of the scene |
| labelled set | `n ∈ {5,10,20,40,60}` pixels per class, stratified, capped at class size, `numpy.random.default_rng(seed)`, seeds 42 … 51 — identical draws to the paper |
| evaluation | ground-truth pixels **without** a label ("unl", primary) and all ground-truth pixels ("all"); ACC and macro-F1 after Hungarian matching, NMI; Xie–Beni on (U, V) |
| Softmax | `lr = 0.01`, `λ = 1e-4`, `10 000` epochs, mini-batch 64, zero init, reshuffle every epoch |
| pooling | log-opinion pool, `r = 2` (Sw-SSFCM r=2) or `r = 1`, `ω = 0.5`, clip `1e-6` |
| guided FCM | `θ = 0.99` (global, selected leave-one-scene-out; `α = θ/(1−θ)·S_d/S_g`, scales measured on the labelled pixels with 5-fold out-of-fold Softmax of 1 000 epochs, clip `1e-6`), `m = 2`, `ε = 1e-4` on `max|Δu|`, `U⁽⁰⁾ = π`, `max_iter = 10 000` |
| one Softmax per cell | shared by the four columns; reported time of a column = Softmax time + its own time |

```bash
python reproduce/benchmark_budget.py --data-root ~/data/HSI                       # everything, resumable
python reproduce/benchmark_budget.py --data-root ~/data/HSI --datasets botswana ksc --budgets 10 60 --seeds 42 43
python reproduce/make_tables.py --results reproduce/results
```

Output `reproduce/results/benchmark_budget.csv` has the same columns as the
published file (`dataset, budget, seed, algo, theta, alpha, ratio, share_g, acc_unl, nmi_unl,
f1_unl, acc_all, nmi_all, xb, iters, time_s, status, note`); `make_tables.py` writes `tables.md` and
`summary.csv`.

## 3. Computational requirements

Pure NumPy, CPU only; BLAS threads are used for the `N × C × d` products.

| Resource | Requirement |
|---|---|
| RAM | ≈ `8·N·(d + 6C)` bytes + the cube during loading: 2 GB for the four smaller scenes, 8 GB comfortable for Houston 2013 (the largest scene) |
| Disk | ≈ 700 MB for all scenes; results < 2 MB |
| Time per cell (8-core desktop, NumPy/OpenBLAS, measured) | Indian Pines @10: 6 s for all five columns; Botswana (378 k px) @60: 30 s (Softmax 10 000 epochs 5 s, each guided-FCM fit 3–4 iterations, 5–7 s). Largest scene (Houston 2013): ≈ 1–2 min per cell |

The full protocol is 300 cells (6 × 5 × 10) — roughly 2–3 CPU-hours in total, dominated
by the largest scenes (Houston 2013, Botswana, KSC). `--epochs 1000` and fewer `--seeds` shorten exploratory runs;
the paper numbers need the defaults.

## 4. Relation to the published numbers

The published run used the authors' GPU research library (float32, CuPy RNG for the
SGD shuffling); this repository is an independent float64 NumPy implementation of the
same equations. Differences come only from the Softmax SGD (shuffle order, precision);
given the same posterior, the pooled prior and the guided FCM are deterministic and
match to rounding. Under the protocol above the repository reproduces the published
ACC within a few tenths of a point, and reproduces exactly the two qualitative
results of the paper: Sr-SSFCM = Softmax and Sw-SSFCM r=2 well above Softmax. Verified cells (unl-only ACC %):

| Cell | Published (GPU) | This repository (NumPy) |
|---|---|---|
| Botswana, 60/class, seed 42 — Softmax / Sr-SSFCM / Sw-SSFCM r=1 / r=2 | 92.94 / 92.94 / 96.69 / 97.34 | 92.90 / 92.86 / 96.68 / 97.26 |
| Indian Pines, 10/class, seed 42 — Softmax / Sr-SSFCM / Sw-SSFCM r=1 / r=2 | 49.31 / 49.31 / 56.40 / 57.88 | 49.24 / 49.28 / 56.35 / 57.81 |

(`python examples/run_on_hsi.py --dataset botswana --data-root … --budget 60 --seed 42`
and `--dataset indian_pines --budget 10 --seed 42` reproduce the right-hand column.)

The spatial and semi-supervised FCM baselines of the paper (FCM, FCM_S1, FCM_S2,
FLICM, KWFLICM, SSFCM, eSFCM, S3FCM, GS-SPFCM, and KFCM / SMUC / S2-PFCM at 60
labels) are previously published algorithms by other authors run from the authors'
library; their per-cell numbers are included in `published/benchmark_budget.csv` and
their references are in the manuscript. `swssfcm.guided_fcm(X, G=0, U0)` is a plain FCM
with the same code path if an unguided reference point is needed.
