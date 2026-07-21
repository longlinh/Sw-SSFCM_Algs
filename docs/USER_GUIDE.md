# User guide

Complete reference for the two estimators in this repository: what they take as
input, what they produce, every parameter they accept, and what behaviour to
expect. For a task-oriented walkthrough see [TUTORIAL.md](TUTORIAL.md); for
reproducing the published numbers see [../reproduce/README.md](../reproduce/README.md).

## Contents

1. [Concepts](#1-concepts)
2. [Inputs](#2-inputs)
3. [Parameters](#3-parameters)
4. [Choosing alpha through tau](#4-choosing-alpha-through-tau)
5. [Outputs](#5-outputs)
6. [Expected behaviour](#6-expected-behaviour)
7. [Errors and diagnostics](#7-errors-and-diagnostics)
8. [Helper modules](#8-helper-modules)

---

## 1. Concepts

Both algorithms are semi-supervised extensions of Fuzzy C-Means (FCM). FCM
assigns every sample a degree of membership in each of `C` clusters by
minimising a weighted sum of squared distances to cluster centroids. Neither
algorithm changes that optimisation scheme; both change the *distance* it
minimises.

**SeFCM** trains a multinomial logistic (softmax) regressor once on the labelled
samples, obtains a posterior `p_ik` for every sample `i` and class `k`, and
subtracts its log from the squared distance:

```
d²_ik = ‖x_i − v_k‖² − α · ln p_ik
```

A sample the classifier is confident about gets its distance to that class's
centroid reduced, so the clustering is pulled toward the labelled structure. The
strength of that pull is `α`.

**Sw-SSFCM** additionally exploits the fact that neighbouring pixels of an image
usually share a class. At every iteration it averages the current memberships
over the spatial neighbourhood of each pixel and blends that average with the
softmax posterior:

```
ũ_ik  = mean of u_jk over the neighbours j of i        (centre excluded)
ω_i   = max_k p_ik / max_i max_k p_ik                  (adaptive confidence)
p̂_ik  = ω_i · p_ik + (1 − ω_i) · ũ_ik                   (row-normalised)
d̂²_ik = ‖x_i − v_k‖² − α · ln p̂_ik
```

`ω_i` is computed once from the fixed softmax output and is constant across
iterations. Where the classifier is confident, guidance dominates; where it is
not, the neighbourhood consensus takes over.

---

## 2. Inputs

### `X` — feature matrix

`ndarray` of shape `(N, D)`, dtype convertible to `float64`.

For a hyperspectral scene of `H` rows, `W` columns and `B` bands, flatten the
cube in **row-major order** so that `X[i]` is pixel `(i // W, i % W)`:

```python
X = cube.reshape(H * W, B)
```

`SwSSFCM` relies on this ordering to reconstruct the neighbourhood structure;
column-major (Fortran) flattening will silently produce wrong neighbourhoods.

**Standardise the features.** Both algorithms compare a squared Euclidean
distance against `α · ln p`, so the feature scale directly changes the effective
guidance strength. All published results use per-band zero-mean, unit-variance
scaling (`reproduce.datasets.standardize`, equivalent to scikit-learn's
`StandardScaler`).

### `y_partial` — partial labels

`ndarray` of shape `(N,)`, integer dtype.

| Value | Meaning |
|---|---|
| `-1` | unlabelled — the vast majority of samples |
| `0 … C-1` | this sample is known to belong to that class |

Only entries `>= 0` are used to train the softmax regressor and to initialise
the centroids. Labels are never used to evaluate the model, so the same vector
can be reused for training and evaluation protocols.

Note the offset: ground truth in the standard `.mat` distributions uses `0` for
background and `1 … C` for classes. Subtract one from the classes and map
background to `-1`. `reproduce.datasets.load_scene` does this for you.

### `image_shape` — required by `SwSSFCM.fit`

Tuple `(H, W)` with `H * W == N`. Not accepted by `SeFCM`, which has no spatial
term and treats samples as an unordered set.

---

## 3. Parameters

Constructor signatures:

```python
SeFCM(n_clusters, alpha=1.0, m=2.0, max_iter=10000, tol=1e-4,
      softmax_lr=0.01, softmax_l2=1e-4, softmax_max_epoch=10000,
      softmax_tol=0.0, random_state=42, verbose=False)

SwSSFCM(n_clusters, alpha=1.0, radius=1, m=2.0, max_iter=10000, tol=1e-4,
        softmax_lr=0.01, softmax_l2=1e-4, softmax_max_epoch=10000,
        softmax_tol=0.0, random_state=42, verbose=False)
```

| Parameter | Type | Default | Paper symbol | Description |
|---|---|---|---|---|
| `n_clusters` | int | required | `C` | Number of clusters. Set it to the number of ground-truth classes; the labels in `y_partial` must lie in `[0, n_clusters-1]`. |
| `alpha` | float | `1.0` | `α` | Guidance strength. `0.0` disables guidance and recovers FCM (with label-informed initialisation). Larger values trust the softmax posterior more. Not scale-free — see [section 4](#4-choosing-alpha-through-tau). |
| `radius` | int | `1` | `r` | *`SwSSFCM` only.* Spatial neighbourhood radius; must be `1` or `2`, giving a `3×3` or `5×5` window with the centre pixel excluded. `r=2` was best on all eight benchmark scenes but costs more per iteration. |
| `m` | float | `2.0` | `m` | Fuzzifier. `m → 1` approaches hard assignment, larger `m` produces fuzzier memberships. Must be `> 1`. Every published result uses `2.0`, the FCM standard. |
| `max_iter` | int | `10000` | `T` | Upper bound on FCM iterations. Effectively never reached: convergence takes 5–20 iterations because the softmax posterior initialises the partition well. |
| `tol` | float | `1e-4` | `ε` | Convergence threshold on the Frobenius norm `‖V_new − V_old‖_F`. Iteration stops as soon as the centroids move less than this. |
| `softmax_lr` | float | `0.01` | `γ` | Learning rate of the softmax regressor's mini-batch SGD. |
| `softmax_l2` | float | `1e-4` | `λ` | L2 regularisation coefficient of the softmax regressor. |
| `softmax_max_epoch` | int | `10000` | `T_s` | Number of passes over the labelled set when training the softmax regressor. |
| `softmax_tol` | float | `0.0` | — | Early-stopping threshold on the softmax loss decrease. `0.0` disables early stopping so the full `softmax_max_epoch` epochs always run; this is the setting behind the published numbers. Setting it to e.g. `1e-6` cuts fitting time by roughly an order of magnitude at a cost of 2–3 accuracy points. |
| `random_state` | int | `42` | — | Seed for the mini-batch shuffling. It is the only source of randomness: with a fixed seed and fixed inputs, results are bit-identical across runs. |
| `verbose` | bool | `False` | — | Print the centroid movement at each iteration. |

The batch size of the softmax regressor is fixed at 64
(`_common.SoftmaxRegression`), matching the published configuration.

---

## 4. Choosing alpha through tau

`α` is not comparable across scenes: it is added to a squared distance whose
magnitude grows with the feature dimension `d`. The paper therefore reports a
normalised quantity

```
τ = α · ln C / (d + α · ln C)   ∈ [0, 1)
```

which measures the share of the total cost budget spent on guidance and *is*
comparable across scenes. Invert it to obtain the `α` a given scene needs:

```
α = τ / (1 − τ) · d / ln C
```

```python
import numpy as np
alpha = tau / (1 - tau) * n_bands / np.log(n_clusters)
```

Available as `reproduce.tau_sweep.tau_to_alpha(tau, n_bands, n_clusters)`.

Practical guidance:

- **Sweep `τ` on `{0.1, 0.2, …, 0.9, 0.95}`, not `α`.** On the eight benchmark
  scenes the optimum sits at `τ* ∈ [0.6, 0.95]`, most often at `0.9` or `0.95`.
  `τ = 0.9` is a good single starting value.
- **Sweep each variant separately.** The spatial term shifts the optimum, so
  `τ*` for SeFCM, Sw-SSFCM `r=1` and Sw-SSFCM `r=2` generally differ. Reusing
  one variant's `τ*` for another understates its accuracy.
- Accuracy is a flat, single-peaked function of `τ` near the optimum, so a
  coarse grid is sufficient.

---

## 5. Outputs

`fit` returns the estimator itself; `fit_predict` returns `labels_` directly.
After a successful fit:

| Attribute | Shape | Description |
|---|---|---|
| `U_` | `(N, C)` | Fuzzy membership matrix. Rows sum to 1. |
| `V_` | `(C, D)` | Cluster centroids in feature space. |
| `P_` | `(N, C)` | Softmax posterior, clipped to `[1e-12, 1]`. Constant across iterations. |
| `labels_` | `(N,)` | Hard labels, `U_.argmax(axis=1)`. |
| `n_iter_` | int | FCM iterations actually performed. |
| `omega_` | `(N, 1)` | *`SwSSFCM` only.* Adaptive confidence weight per sample, in `(0, 1]`. |

`predict(X_new)` assigns unseen samples to the nearest centroid under the
guided distance, reusing the fitted softmax regressor. It is available on both
classes, but note that it ignores the spatial term — for `SwSSFCM` it is a
spectral-only approximation, so prefer refitting on the full scene.

### Cluster ids are arbitrary

`labels_` contains cluster indices, not class indices. Cluster `3` is not
necessarily class `3`. Before comparing against ground truth, resolve the
permutation with the Hungarian algorithm:

```python
from metrics import evaluate
scores = evaluate(y_true, model.labels_)   # {'acc': ..., 'nmi': ..., 'ari': ...}
```

`metrics.evaluate` does this for accuracy; NMI and ARI are permutation-invariant
and need no mapping.

### Evaluate on labelled pixels only

Benchmark scenes leave part of the image without ground truth. Restrict the
comparison to pixels that carry a label, while still clustering the whole image
so the spatial term sees intact neighbourhoods:

```python
mask = y_true >= 0
scores = evaluate(y_true[mask], model.labels_[mask])
```

---

## 6. Expected behaviour

- **Convergence.** 5–20 iterations on the benchmark scenes. `n_iter_` equal to
  `max_iter` means it did not converge — check that features are standardised.
- **Determinism.** Fixed `random_state` and fixed inputs give bit-identical
  output. The `reproduce/smoke_test.py` script asserts this.
- **Effect of guidance.** `alpha=0` is the unguided reference. Guidance should
  improve accuracy substantially; on the benchmark scenes the gain over
  unguided FCM is large, and SeFCM reaches 86.65 % mean accuracy at 60 labels
  per class.
- **Effect of the spatial term.** Sw-SSFCM beats SeFCM on every benchmark scene
  (mean +2.22 points for `r=2`), and `r=2` beats `r=1` by a further 0.29 points
  on average. On data with no spatial structure the spatial term cannot help,
  and passing a meaningless `image_shape` will hurt.
- **Sensitivity to `τ`.** The dominant hyperparameter by a wide margin. Nothing
  else in the table needs tuning to reproduce the published results.

---

## 7. Errors and diagnostics

| Message | Cause | Fix |
|---|---|---|
| `X and y_partial must have same length.` | `y_partial` does not have one entry per row of `X`. | Flatten the ground truth the same way as the cube. |
| `image_shape (H, W) incompatible with N=...` | `H * W` differs from the number of rows in `X`. | Pass the true image dimensions; do not drop background pixels before fitting. |
| `radius must be 1 or 2, got r` | Unsupported neighbourhood size. | Use `1` or `2`. |
| `TypeError: fit() missing 1 required positional argument: 'image_shape'` | `SwSSFCM.fit` called with the `SeFCM` signature. | Pass `image_shape=(H, W)`. |

Behavioural symptoms:

| Symptom | Likely cause |
|---|---|
| Accuracy near chance | Features not standardised, or `alpha` orders of magnitude off. Set `alpha` from `τ = 0.9`. |
| Sw-SSFCM no better than SeFCM | Wrong `image_shape`, or column-major flattening, so neighbourhoods are meaningless. |
| Some clusters empty | A class has too few labelled samples; centroid initialisation fell back to the global mean. Increase labels for that class. |
| Fitting far slower than expected | `softmax_max_epoch=10000` with no early stop dominates the runtime on small labelled sets. Set `softmax_tol=1e-6` while exploring. |

---

## 8. Helper modules

Beyond the two estimators, the repository ships utilities used by the
reproduction scripts. They are ordinary modules and can be imported directly.

| Module | Contents |
|---|---|
| `metrics` | `evaluate`, `cluster_accuracy`, `nmi`, `ari`, `hungarian_mapping`, `remap_labels`. |
| `reproduce.datasets` | `SPECS` (the eight scene descriptors), `load_scene`, `standardize`, `stratified_labels`, `make_synthetic_scene`, `sha256_of`. |
| `reproduce.tau_sweep` | `tau_to_alpha`, `fit_variant`, `sweep_scene`, `summarize`. |
| `_common` | `SoftmaxRegression` and the FCM update rules shared by both estimators. |

### Sampling labels

`stratified_labels` implements the protocol used throughout the paper: draw at
most `n_per_class` labelled pixels per class without replacement, take all of
them for classes smaller than that, and mark everything else `-1`.

```python
from reproduce.datasets import stratified_labels
y_partial = stratified_labels(y_true, valid_mask, n_per_class=60, seed=42)
```

### Synthetic data

`make_synthetic_scene` generates a scene with the statistical structure of a
real HSI — spatially contiguous Voronoi class regions, smooth per-class spectra,
additive Gaussian noise, and a fraction of unlabelled background — without any
download:

```python
from reproduce.datasets import make_synthetic_scene
scene = make_synthetic_scene(height=64, width=64, n_bands=30, n_clusters=6, seed=42)
```

Raise `noise` to make the problem harder; the default puts accuracy in the same
range as the real benchmarks.
