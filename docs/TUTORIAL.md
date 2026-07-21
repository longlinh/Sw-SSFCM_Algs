# Tutorial

Five worked examples, in increasing order of commitment. Every command is meant
to be run from the repository root and every one of them has been executed as
written.

| # | Task | Data needed | Time |
|---|---|---|---|
| [1](#1-first-run-no-data-required) | Check the install works | none | ~3 s |
| [2](#2-cluster-your-own-table-of-features) | Cluster a plain feature table | none | ~1 s |
| [3](#3-cluster-an-image-with-the-spatial-term) | Use the spatial term on an image | none | ~10 s |
| [4](#4-run-on-a-real-hyperspectral-scene) | Run on a benchmark scene | one download | ~2–5 min |
| [5](#5-find-the-best-guidance-strength) | Tune `τ` on your own scene | your scene | minutes to hours |

Before starting:

```bash
pip install -r requirements.txt
```

---

## 1. First run, no data required

`demo.py` fits both algorithms on data it generates itself. If this prints two
lines of scores, the install is good.

```bash
python demo.py
```

```
=== Demo 1: SeFCM on Iris ===
  ACC=0.9467  NMI=0.8449  iters=8  time=0.46s

=== Demo 2: Sw-SSFCM on synthetic 30x30 image ===
  ACC=0.9967  NMI=0.9830  iters=7  time=0.63s
```

For a stronger check that also exercises the tau-sweep and table generation,
run the smoke test — it asserts determinism and that both contributions help:

```bash
python reproduce/smoke_test.py
```

It finishes in under ten seconds and ends with `SMOKE TEST PASSED`.

---

## 2. Cluster your own table of features

Use `SeFCM` when samples have no spatial relationship — a CSV of measurements,
spectra from field sampling, anything shaped `(N, D)`.

```python
import numpy as np
from sklearn.datasets import load_iris

from metrics import evaluate
from sefcm import SeFCM

X, y = load_iris(return_X_y=True)

# Keep 10% of the labels, stratified; the rest are unknown.
rng = np.random.default_rng(42)
y_partial = np.full_like(y, -1)
for k in np.unique(y):
    idx = np.where(y == k)[0]
    y_partial[rng.choice(idx, size=5, replace=False)] = k

model = SeFCM(n_clusters=3, alpha=2.0, random_state=42).fit(X, y_partial)

print(model.labels_[:10])
print(evaluate(y, model.labels_))
```

Three things to carry over to your own data:

- **Standardise `X` first** if the columns have different units. Guidance is
  compared against a squared distance, so feature scale matters.
- **`-1` means unlabelled.** Class indices must run `0 … C-1`.
- **`labels_` holds cluster ids, not class ids.** Use `metrics.evaluate`, which
  resolves the permutation with the Hungarian algorithm.

---

## 3. Cluster an image with the spatial term

`SwSSFCM` additionally needs `image_shape`, and `X` must be flattened
**row-major** so that row `i` is pixel `(i // W, i % W)`.

```python
import numpy as np

from metrics import evaluate
from reproduce.datasets import make_synthetic_scene, stratified_labels
from sefcm import SeFCM
from sw_ssfcm import SwSSFCM

scene = make_synthetic_scene(height=64, width=64, n_bands=30, n_clusters=6, seed=42)
y_partial = stratified_labels(scene.y_true, scene.valid_mask, n_per_class=20, seed=42)

mask = scene.valid_mask
alpha = 0.9 / (1 - 0.9) * scene.n_bands / np.log(scene.n_clusters)   # tau = 0.9

no_spatial = SeFCM(n_clusters=6, alpha=alpha, random_state=42).fit(scene.X, y_partial)
spatial = SwSSFCM(n_clusters=6, alpha=alpha, radius=1, random_state=42).fit(
    scene.X, y_partial, image_shape=(scene.height, scene.width)
)

print("SeFCM   ", evaluate(scene.y_true[mask], no_spatial.labels_[mask])["acc"])
print("Sw-SSFCM", evaluate(scene.y_true[mask], spatial.labels_[mask])["acc"])
```

```
SeFCM    0.7459274078308088
Sw-SSFCM 0.794226921977708
```

The spatial term is worth about five accuracy points here. To view the result as
an image, reshape it:

```python
label_image = spatial.labels_.reshape(scene.height, scene.width)
```

Two mistakes to avoid:

- **Do not drop background pixels before fitting.** Cluster the whole image so
  neighbourhoods stay intact, then restrict the *evaluation* to labelled pixels.
- **Do not flatten column-major.** `cube.reshape(H * W, B)` is correct;
  transposing first silently corrupts every neighbourhood.

---

## 4. Run on a real hyperspectral scene

Fetch one scene (Indian Pines is the smallest, about 6 MB):

```bash
python reproduce/download_data.py --data-root ~/data/HSI --dataset indian_pines
```

Then run the example script, which handles loading, label sampling, fitting and
evaluation:

```bash
python examples/run_on_hsi.py --dataset indian_pines --data-root ~/data/HSI
```

```
[load] Indian Pines: 145x145x200, C=16, 10,249 labelled pixels
[fit ] Sw-SSFCM(tau=0.9, alpha=649.21, radius=2) on 874 labelled samples
       converged in 18 iterations, 103.7s
[eval] ACC=75.16%  NMI=65.76%  ARI=50.90%
```

Useful flags: `--tau`, `--radius`, `--labels-per-class`. See
`python examples/run_on_hsi.py --help`.

### On a scene of your own

The same script takes explicit paths. Give it the cube file, the ground-truth
file, and the MATLAB variable name inside each:

```bash
python examples/run_on_hsi.py \
    --cube my_scene.mat --cube-key data \
    --gt my_scene_gt.mat --gt-key labels \
    --tau 0.9 --radius 2
```

Requirements on your files: cube shaped `(H, W, B)`, ground truth shaped
`(H, W)` with `0` for background and `1 … C` for classes. Pass the same path
twice if both variables live in one file. Use `--n-clusters` to override the
inferred class count.

If your data is not in MATLAB format, load it however you like and construct a
`Scene` yourself — see `load_custom_scene` in `examples/run_on_hsi.py` for the
15 lines involved.

---

## 5. Find the best guidance strength

`τ` is the one parameter that genuinely needs tuning. Sweep it with
`reproduce/tau_sweep.py`, which fits all three variants over a grid and reports
the best operating point per variant.

Start on synthetic data to see the shape of the output:

```bash
python reproduce/tau_sweep.py --synthetic --tau 0.3 0.6 0.9
```

Then on a real scene:

```bash
python reproduce/tau_sweep.py --data-root ~/data/HSI --datasets indian_pines
```

```
Scene                Variant          tau*     alpha*      ACC      NMI
------------------------------------------------------------------------------
indian_pines         SeFCM            0.90     649.21   71.22%   59.99%
indian_pines         Sw-SSFCM_r1      0.95    1370.56   74.71%   65.42%
indian_pines         Sw-SSFCM_r2      0.95    1370.56   75.24%   65.85%
```

Turn the results into tables:

```bash
python reproduce/make_tables.py --results reproduce/results
```

Notes:

- **Sweep each variant separately.** The spatial term shifts the optimum;
  reusing SeFCM's `τ*` for Sw-SSFCM understates its accuracy.
- **`τ` is scale-free, `α` is not.** A `τ` that works on a 100-band scene
  transfers to a 270-band scene; the corresponding `α` does not.
- **Start coarse.** `{0.3, 0.6, 0.9}` locates the region; the accuracy curve is
  flat near its peak, so refine only if you need the last few tenths.
- Restrict work with `--variants`, `--tau` and `--labels-per-class` while
  exploring; drop `--max-iter` to something like `200` for a quick look.

---

## Where to go next

- [USER_GUIDE.md](USER_GUIDE.md) — every parameter, output and failure mode.
- [../reproduce/README.md](../reproduce/README.md) — reproducing the published
  tables, including runtimes and hardware requirements.
