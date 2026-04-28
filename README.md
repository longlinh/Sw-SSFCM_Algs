# SeFCM & Sw-SSFCM — Reference Implementation

Companion code for our ISI 2026 submission:

> *Spatial-weighted Softmax-Embedded Semi-Supervised Fuzzy C-Means for
> Hyperspectral Image Classification.*
> Nguyen Xuan Hoang, Mai Dinh Sinh, Nguyen Long Giang. 2026.

This repository contains a self-contained, NumPy-only implementation of the
two algorithms proposed in the paper:

- **SeFCM** — FCM with a Softmax cross-entropy guidance term added to the
  squared distance: `d² = ‖x − v‖² − α · ln p`.
- **Sw-SSFCM** — SeFCM extended with a spatial neighborhood term and an
  adaptive confidence weight `ω`, intended for image-shaped data
  (radius `r ∈ {1, 2}`).

The code mirrors the pseudocode in the paper one-to-one. Any production /
GPU / multi-view variants live in a separate research library and are *not*
required to reproduce the paper's results.

## Repository layout

```
.
├── _common.py          # internal: FCM core + Softmax regression
├── sefcm.py            # class SeFCM
├── sw_ssfcm.py         # class SwSSFCM
├── demo.py             # synthetic end-to-end demo (Iris + 30x30 image)
├── examples/
│   └── run_on_hsi.py   # template for plugging in your own HSI dataset
├── README.md
├── LICENSE
└── requirements.txt
```

All Python files live at the repo root — clone, `cd` in, and run.

## Install

```bash
git clone <this-repo>
cd <repo>
pip install -r requirements.txt
```

Only `numpy` is mandatory. `scikit-learn` and `scipy` are used by the demo
for dataset loading and Hungarian-matching cluster accuracy.

## Quick start

### SeFCM

```python
import numpy as np
from sklearn.datasets import load_iris
from sefcm import SeFCM

X, y = load_iris(return_X_y=True)
y_partial = y.copy()
rng = np.random.default_rng(42)
y_partial[rng.random(len(y)) > 0.10] = -1   # 90% unlabeled

model = SeFCM(n_clusters=3, alpha=2.0, random_state=42).fit(X, y_partial)
print(model.labels_, model.n_iter_)
```

### Sw-SSFCM

```python
from sw_ssfcm import SwSSFCM

# X has shape (H*W, D) in row-major order; image_shape=(H, W).
model = SwSSFCM(n_clusters=6, alpha=30.0, radius=2, random_state=42)
model.fit(X, y_partial, image_shape=(H, W))
print(model.labels_.reshape(H, W))
```

End-to-end synthetic demo (no external data, < 1s):

```bash
python demo.py
```

Run on your own hyperspectral image (`.mat` file with cube + ground truth):

```bash
python examples/run_on_hsi.py /path/to/dataset.mat
```

See `examples/run_on_hsi.py` for the expected file format and how to adapt
the loader to other HSI distributions (ENVI, HDF5, .npy).

## API

| Class | Signature |
|---|---|
| `SeFCM` | `SeFCM(n_clusters, alpha=1.0, m=2.0, max_iter=10000, tol=1e-4, softmax_lr=0.01, softmax_l2=1e-4, softmax_max_epoch=1000, random_state=42, verbose=False)` |
| `SwSSFCM` | `SwSSFCM(..., radius=1)` (inherits from `SeFCM`) |

Both classes follow the scikit-learn convention:

- `fit(X, y_partial)` for `SeFCM`,
- `fit(X, y_partial, image_shape=(H, W))` for `SwSSFCM`.

`y_partial[i] = -1` marks an unlabeled point; class indices are in `[0, C-1]`.

After fitting:

- `model.U_` — fuzzy membership matrix, shape `(N, C)`
- `model.V_` — cluster centroids, shape `(C, D)`
- `model.P_` — Softmax guidance matrix, shape `(N, C)`
- `model.labels_` — hard labels (argmax over `U_`)
- `model.n_iter_` — number of FCM iterations
- `model.omega_` *(Sw-SSFCM only)* — adaptive weights, shape `(N, 1)`

## Reproducing the paper results

The paper benchmarks on eight HSI datasets (Botswana, Salinas, KSC, Indian
Pines, PaviaU, PaviaC, Houston2013, WHU-Hi-LongKou) with 60 labeled samples
per class. The dataset preparation, ground-truth alignment, and per-dataset
`α`-sweep scripts are not part of this minimal repository — they live in the
authors' research repository and are described in detail in the paper.

The code in this folder is a *reference implementation*: given the same input
`(X, y_partial, image_shape)` and seed, it produces results numerically
consistent with those reported in the paper.

## Citation

<!-- TODO: replace with the final BibTeX once the DOI is assigned. -->

```bibtex
@article{hoang2026swssfcm,
  title   = {Spatial-weighted Softmax-Embedded Semi-Supervised Fuzzy C-Means
             for Hyperspectral Image Classification},
  author  = {Nguyen, Xuan Hoang and Mai, Dinh Sinh and Nguyen, Long Giang},
  journal = {(under review)},
  year    = {2026},
  note    = {DOI: TBA on acceptance}
}
```

## License

MIT — see [LICENSE](LICENSE).
