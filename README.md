# Sw-SSFCM — reference implementation

Companion source code for the manuscript

> **A novel approach to spatial-weighted semi-supervised fuzzy c-means clustering
> for hyperspectral image analysis.**
> Xuan Hoang Nguyen, Dinh Sinh Mai, Long Giang Nguyen.
> Submitted to *Computers & Geosciences* (under review, 2026).

A compact, NumPy-only implementation of the proposed algorithm, written to be read
next to the equations of the paper: one module, plain functions, no class hierarchy.
This repository publishes the **algorithm** and a self-contained synthetic demo; the
hyperspectral scenes of the paper are third-party datasets and are not included (see
*Data availability*).

**Sw-SSFCM** is a fuzzy c-means whose distance is augmented by a *frozen spatial
prior* built from the Softmax posterior of the labelled pixels:

```
p_ik    = softmax_k(Wᵀx_i + b)                                  Softmax posterior (labelled pixels only)
ln π̃_ik = ω ln p_ik + (1−ω)/|N_i| Σ_{j∈N_i} ln p_jk              log-opinion pool over the (2r+1)² window
J_m(U,V) = Σ_i Σ_k u_ik^m [ ‖x_i − v_k‖² − α ln π_ik ]          objective; closed-form U and V updates ∀ m>1
α = θ/(1−θ) · S_d/S_g                                           guidance share θ ∈ [0,1); S_d, S_g = measured scales
S_d = mean_{i∈L,k} ‖x_i − μ_k‖²,  S_g = mean_{i∈L,k} −ln p̂_ik      on the labelled pixels L (p̂ = out-of-fold posterior)
```

π is computed once from the posterior and kept fixed, so both the membership and the
centroid updates stay closed-form for every `m > 1` and the iteration reduces to
classical FCM. One special case is part of the method: **Sr-SSFCM** (`r = 0`, `π = p`
— the unpooled guided FCM), used to isolate the contribution of the spatial prior.

## Install

Python 3.9 or newer, three pure-Python-wheel dependencies (no GPU):

```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn
python demo.py                         # synthetic scene, < 5 s, no data needed
```

`demo.py` prints, deterministically:

```
Softmax      ACC=0.7424
Sw-SSFCM r=2 ACC=0.9247  (alpha=1539.9, 3 iterations)
label image: (64, 64) memberships U: (4096, 6) centroids V: (6, 30)
```

The two accuracies are the story of the paper in miniature: the spatially pooled prior
lifts the raw Softmax posterior, and the guided clustering adds a fuzzy partition and
spectral centroids.

## Quick start

```python
import numpy as np
from swssfcm import sw_ssfcm
from metrics import evaluate

# X: (H*W, d) pixels in row-major order (band-wise standardised),
# y: (H*W,) partial labels with -1 = unlabelled, 0..C-1 = class.
res = sw_ssfcm(X, y, H, W, n_clusters=C, theta=0.99, r=2, omega=0.5, seed=42)
res["labels"]        # (N,)  hard labels = argmax_k u_ik
res["U"], res["V"]   # (N,C) memberships, (C,d) centroids
res["pi"], res["P"]  # (N,C) pooled prior π and Softmax posterior p
res["alpha"], res["ratio"], res["share_g"], res["n_iter"]   # α, measured S_d/S_g, realised guidance share

gt = (y_true >= 0) & (y < 0)                       # ground-truth pixels without a label
print(evaluate(y_true[gt], res["labels"][gt]))     # {'acc', 'nmi', 'f1'} after Hungarian matching
```

Apply it to your own cube: reshape an `H × W × d` image to `(H*W, d)` in row-major
order, standardise the bands (zero mean, unit variance per band), and pass a partial
label vector. `sr_ssfcm(X, y, ...)` runs the `r = 0` control on the same inputs.

## Documentation

| Document | Contents |
|---|---|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | every function, its inputs, outputs, parameters, expected behaviour and failure modes |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | worked examples: the synthetic run, your own cube, sharing one Softmax, the θ/ω sweep, classes without labels |

## Choosing θ and ω

`α` is a conversion factor with the units of a squared spectral distance (the guidance
term `−ln π` is an information quantity in nats), so it is not comparable across scenes.
The method therefore parameterises it by the **guidance share θ**: `α = θ/(1−θ)·S_d/S_g`,
where `S_d` (mean squared distance of the labelled pixels to the labelled class means)
and `S_g` (mean out-of-fold `−ln p̂`) are measured on the labelled set itself
(`theta_scales`) — no ground truth, no per-scene grid. The share of the guidance term
actually realised at convergence (`share_g`) tracks θ. The paper uses a single global
**θ = 0.99** across all scenes (selected leave-one-scene-out; accuracy increases
monotonically with θ and the loss of the global value against the per-scene optimum is
≤ 0.07 points). **ω = 0.5** and **r = 2** are the defaults; `ω = 0.25` gave about
+1.6 points on average in the ablation and is worth trying on agricultural scenes.

## API

| Function | Purpose |
|---|---|
| `train_softmax(X, y, lr, l2, epochs, batch_size, seed)` → `(W, b)` | multinomial logistic regression (cross-entropy + L2, mini-batch SGD) |
| `posterior(X, W, b)` → `P` | Softmax posterior p |
| `pooled_prior(P, H, W, r, omega, pool)` → `π` | log-opinion pool (`pool="log"`) or linear pool (`"arith"`, ablation) |
| `guided_fcm(X, G, U0, m, eps, max_iter)` → `(U, V, n_iter)` | alternating minimisation of J_m with a fixed `G = −α ln π` (`G = 0` is plain FCM) |
| `sw_ssfcm(X, y, H, W, n_clusters, theta, r, omega, m, eps, max_iter, pool, P, prior, ratio, seed, softmax_kw)` → dict | the full method |
| `sr_ssfcm(X, y, ...)` → dict | `r = 0` special case (`π = p`) |
| `open_set_prior(P, seen, n_clusters, H, W, r, omega, mode, X, y, logits)` → `π` | prior for classes without labels (`noop` / `energy` / `maha`) |
| `theta_scales(X, y, folds, epochs, seed)` → `dict(S_d, S_g, ratio)` | measured scales of the two cost terms on the labelled pixels (out-of-fold Softmax) |
| `alpha_from_theta(theta, ratio)`, `guidance_share(X, U, V, G, m)`, `objective(X, U, V, G, m)` | helpers |
| `metrics.evaluate / accuracy / nmi / macro_f1 / xie_beni` | evaluation |

Pixels are row-major: `X[i]` is the pixel at `(i // W, i % W)` and `labels.reshape(H, W)`
is the label image. Labels are `0 … C-1`, `-1` = unlabelled.

## Computational requirements

Pure NumPy on the CPU, no GPU. Peak memory ≈ `8·N·(d + 6C)` bytes plus the cube while
it is in memory. One Sw-SSFCM fit converges in 5–10 iterations; per fit the dominant
costs are training the Softmax (mini-batch SGD on `C × budget` labelled samples) and the
`N × C × d` distance matrix per iteration. For quick exploration pass
`softmax_kw=dict(epochs=1000)` — the guided FCM itself is cheap and deterministic given π.

## Repository layout

```
.
├── swssfcm.py       # the algorithm: Softmax, log-opinion pool, θ-rule, guided FCM, open-set prior
├── metrics.py       # ACC (Hungarian), NMI, macro-F1, Xie–Beni
├── demo.py          # self-contained synthetic demo, no data needed
├── docs/            # USER_GUIDE.md, TUTORIAL.md
├── requirements.txt, CITATION.cff, LICENSE
```

## Data availability

The hyperspectral scenes used in the paper are third-party benchmark datasets
distributed by their original providers under their own terms and are **not**
redistributed here. The algorithm takes any standardised `(N, d)` cube with a partial
label vector, so it can be applied to those scenes — or any other hyperspectral image —
once the data is obtained from its source. The label-sampling protocol and the download
pointers for the specific scenes of the paper are provided with the paper's data
material.

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

See also [CITATION.cff](CITATION.cff). License: MIT.
