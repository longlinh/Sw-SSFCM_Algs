# SeFCM & Sw-SSFCM — reference implementation

Companion code for the manuscript

> **A novel approach to spatial-weighted semi-supervised fuzzy c-means clustering
> for hyperspectral image analysis.**
> Nguyen Xuan Hoang, Mai Dinh Sinh, Nguyen Long Giang.
> Submitted to *Computers & Geosciences* (under review, 2026).

A self-contained, NumPy-only implementation of the two algorithms proposed in
the paper, together with the scripts that reproduce its main experiment on eight
hyperspectral benchmark scenes.

- **SeFCM** — Fuzzy C-Means with a softmax cross-entropy guidance term added to
  the squared distance: `d² = ‖x − v‖² − α·ln p`.
- **Sw-SSFCM** — SeFCM extended with a spatial neighbourhood term and an
  adaptive confidence weight `ω`, for image-shaped data (radius `r ∈ {1, 2}`).

The code follows the pseudocode of Algorithms 1 and 2 of the paper
line-for-line. Everything needed to regenerate the published tables is in
[`reproduce/`](reproduce/), including a synthetic scene generator so the full
pipeline can be exercised without downloading anything.

## Install

Python 3.9 or newer.

```bash
git clone https://github.com/longlinh/Sw-SSFCM_Algs.git
cd Sw-SSFCM_Algs
pip install -r requirements.txt
```

Three dependencies, all pure-Python wheels: `numpy` (the algorithms),
`scipy` (MATLAB file reading and Hungarian matching) and `scikit-learn`
(NMI/ARI and the demo dataset). There is no build step and no GPU requirement.

Verify the install — this runs the entire pipeline on generated data and takes
under ten seconds:

```bash
python reproduce/smoke_test.py
```

It ends with `SMOKE TEST PASSED`.

## Quick start

### SeFCM, on any feature table

```python
import numpy as np
from sklearn.datasets import load_iris

from metrics import evaluate
from sefcm import SeFCM

X, y = load_iris(return_X_y=True)

# Keep 5 labels per class; -1 marks an unlabelled sample.
rng = np.random.default_rng(42)
y_partial = np.full_like(y, -1)
for k in np.unique(y):
    y_partial[rng.choice(np.where(y == k)[0], size=5, replace=False)] = k

model = SeFCM(n_clusters=3, alpha=2.0, random_state=42).fit(X, y_partial)
print(evaluate(y, model.labels_))       # {'acc': 0.947, 'nmi': 0.845, ...}
```

### Sw-SSFCM, on image-shaped data

```python
from sw_ssfcm import SwSSFCM

# X has shape (H*W, D), flattened row-major; image_shape=(H, W).
model = SwSSFCM(n_clusters=6, alpha=30.0, radius=2, random_state=42)
model.fit(X, y_partial, image_shape=(H, W))
label_image = model.labels_.reshape(H, W)
```

### On a real hyperspectral scene

```bash
python reproduce/download_data.py --data-root ~/data/HSI --dataset indian_pines
python examples/run_on_hsi.py --dataset indian_pines --data-root ~/data/HSI
```

```
[load] Indian Pines: 145x145x200, C=16, 10,249 labelled pixels
[fit ] Sw-SSFCM(tau=0.9, alpha=649.21, radius=2) on 874 labelled samples
       converged in 18 iterations, 103.7s
[eval] ACC=75.16%  NMI=65.76%  ARI=50.90%
```

The same script also accepts explicit `--cube` / `--gt` paths for your own data.

## Documentation

| Document | Contents |
|---|---|
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | Five worked examples, from a first run to tuning on your own scene |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Every input, output, parameter, expected behaviour and failure mode |
| [reproduce/README.md](reproduce/README.md) | Reproducing the published tables; data sources; runtime and memory requirements |

## Choosing `alpha`

`α` is not comparable across scenes, since it is added to a squared distance
whose magnitude grows with the feature dimension `d`. Tune the normalised
quantity used in the paper instead:

```
τ = α·ln C / (d + α·ln C) ∈ [0, 1)        α = τ/(1 − τ) · d/ln C
```

Sweep `τ` over `{0.1, …, 0.9, 0.95}`; on the eight benchmark scenes the optimum
sits at `τ* ∈ [0.6, 0.95]`. `τ = 0.9` is a good default. Sweep each variant
separately — the spatial term shifts the optimum. See
[`reproduce/tau_sweep.py`](reproduce/tau_sweep.py).

## API

| Class | Signature |
|---|---|
| `SeFCM` | `SeFCM(n_clusters, alpha=1.0, m=2.0, max_iter=10000, tol=1e-4, softmax_lr=0.01, softmax_l2=1e-4, softmax_max_epoch=10000, softmax_tol=0.0, random_state=42, verbose=False)` |
| `SwSSFCM` | `SwSSFCM(..., radius=1)`, inherits from `SeFCM` |

Both follow the scikit-learn convention — `fit(X, y_partial)` for `SeFCM`,
`fit(X, y_partial, image_shape=(H, W))` for `SwSSFCM`. In `y_partial`, `-1`
marks an unlabelled sample and class indices run `0 … C-1`.

After fitting: `U_` `(N, C)` memberships, `V_` `(C, D)` centroids, `P_` `(N, C)`
softmax posterior, `labels_` `(N,)` hard labels, `n_iter_` iteration count, and
`omega_` `(N, 1)` adaptive weights for `SwSSFCM`. Cluster ids are arbitrary —
use `metrics.evaluate`, which resolves the permutation with the Hungarian
algorithm. Full details in the [user guide](docs/USER_GUIDE.md).

## Computational requirements

Pure NumPy on the CPU; no GPU is used. Peak memory is about
`8·N·(D + 8C)` bytes plus the raw cube while loading — 4 GB suffices for the six
smaller scenes, 16 GB is comfortable for the two largest. All eight scenes need
about 700 MB of disk.

One Sw-SSFCM `r=2` fit under the full protocol takes roughly 100 s on Indian
Pines (21 025 pixels) and 140 s on Salinas (111 104 pixels) on a single CPU
core. A complete `τ` sweep is 30 fits per scene. Detailed figures and ways to
shorten exploratory runs are in [reproduce/README.md](reproduce/README.md).

## Repository layout

```
.
├── sefcm.py              # class SeFCM               (Algorithm 1)
├── sw_ssfcm.py           # class SwSSFCM             (Algorithm 2)
├── _common.py            # FCM update rules + softmax regression
├── metrics.py            # ACC (Hungarian-matched), NMI, ARI
├── demo.py               # synthetic end-to-end demo, no data needed
├── examples/
│   └── run_on_hsi.py     # run on a benchmark scene or your own file
├── reproduce/
│   ├── README.md         # reproduction guide, data sources, requirements
│   ├── datasets.py       # scene registry, loader, synthetic generator
│   ├── download_data.py  # fetch + checksum-verify the scenes
│   ├── tau_sweep.py      # the main experiment
│   ├── make_tables.py    # benchmark tables from sweep results
│   └── smoke_test.py     # end-to-end pipeline check, no data needed
├── docs/
│   ├── TUTORIAL.md
│   └── USER_GUIDE.md
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Data availability

The eight benchmark scenes are third-party datasets and are not redistributed
here. Six download automatically via `reproduce/download_data.py`; Houston 2013
and WHU-Hi-LongKou require a free registration with their providers, and the
script prints the exact steps. Every file is verified against the SHA-256 of the
copy used to produce the published numbers, so a successful `--verify` run
guarantees the inputs are identical to ours. Sources and licences are listed in
[reproduce/README.md](reproduce/README.md). The canonical semi-supervised splits
(60 labels per class, seed 42) and per-scene download links are also published
at <https://github.com/longlinh/Sw-SSFCM_Dataset>.

## Citation

```bibtex
@article{nguyen2026swssfcm,
  title   = {A novel approach to spatial-weighted semi-supervised fuzzy c-means
             clustering for hyperspectral image analysis},
  author  = {Nguyen, Xuan Hoang and Mai, Dinh Sinh and Nguyen, Long Giang},
  journal = {Computers and Geosciences},
  year    = {2026},
  note    = {Under review. DOI to be assigned on acceptance.}
}
```

See also [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
