# User guide

All functionality is in `swssfcm.py` (algorithm) and `metrics.py` (evaluation);
`demo.py` is a self-contained synthetic example. Everything takes and returns NumPy
arrays.

## Conventions

| Symbol / array | Shape | Meaning |
|---|---|---|
| `X` | `(N, d)` float | pixels of an `H × W` image, **row-major** (`X[i]` ↔ `(i // W, i % W)`), one row per pixel, `d` bands. Standardise bands first (zero mean, unit variance per band). |
| `y` | `(N,)` int | partial labels: class index `0 … C-1` for labelled pixels, `-1` elsewhere |
| `P` | `(N, C)` | Softmax posterior `p_ik` |
| `pi` | `(N, C)` | pooled prior `π_ik` |
| `U` | `(N, C)` | memberships, rows sum to 1 |
| `V` | `(C, d)` | centroids |
| `labels` | `(N,)` int | `argmax_k u_ik`; cluster ids are arbitrary — use `metrics.evaluate` (Hungarian matching) |

## `swssfcm.sw_ssfcm(...)` — the method

```python
sw_ssfcm(X, y, H, W, n_clusters=None, theta=0.99, r=2, omega=0.5, m=2.0, eps=1e-4,
         max_iter=10000, pool="log", P=None, prior=None, ratio=None, seed=42, softmax_kw=None) -> dict
```

Steps: (1) `train_softmax` on the labelled pixels → `posterior` P; (2) `pooled_prior`
→ π; (3) `theta_scales` on the labelled pixels → `α = θ/(1−θ)·S_d/S_g`; (4) `guided_fcm` with
`G = −α ln π`, `U⁽⁰⁾ = π`. Returns `dict(U, V, labels, pi, P, alpha, ratio, share_g, n_iter)`.

| Parameter | Default | Notes |
|---|---|---|
| `H, W` | — | image size, `H*W == N`. Ignored when `r = 0`. |
| `n_clusters` | number of distinct labels in `y` | set explicitly when some classes have no labels (open set) |
| `theta` | 0.99 | guidance share θ ∈ [0,1): expected fraction of the guidance term in the cost matrix. `α = θ/(1−θ)·S_d/S_g`, with `S_d` = mean squared distance of labelled pixels to the labelled class means and `S_g` = mean out-of-fold `−ln p̂` (5 stratified folds), both averaged over all (pixel, class) pairs. Paper: one global θ (LOSO) on all scenes. |
| `ratio` | `None` | pass a measured `S_d/S_g` (from `theta_scales`) to share it across variants / θ values |
| `r` | 2 | pooling radius, window `(2r+1)²` without centre; `0` → Sr-SSFCM |
| `omega` | 0.5 | self weight in the pool; `1` → π = p |
| `m`, `eps`, `max_iter` | 2.0, 1e-4, 10000 | fuzzifier; stop when `max |u⁽ᵗ⁾ − u⁽ᵗ⁻¹⁾| < eps` |
| `pool` | `"log"` | `"arith"` = linear opinion pool (ablation only; ≈ 1.2 points worse) |
| `P` | `None` | pass a posterior to skip the Softmax step (share one Softmax across variants; must be `(N, C)` with columns ordered as `np.unique(y[y>=0])`) |
| `prior` | `None` | pass π directly (skips pooling) — used with `open_set_prior` |
| `seed` | 42 | Softmax SGD shuffling; the FCM part is deterministic given π |
| `softmax_kw` | `{}` | forwarded to `train_softmax`: `lr=0.01, l2=1e-4, epochs=10000, batch_size=64` |

Expected behaviour: converges in 5–10 iterations at θ = 0.99 (the guidance
dominates, `J` decreases monotonically). At θ = 0.99 the hard labels coincide with
the argmax of the pooled prior `res["pi"]` up to a handful of pixels; lowering θ moves the
solution towards plain FCM (θ → 0) and the iteration count grows. The fuzzy partition `U` is meaningful (Xie–Beni
lower than the posterior treated as a partition) and varies with `m`, while
`labels` do not.

## Building blocks

- `train_softmax(X, y, lr=0.01, l2=1e-4, epochs=10000, batch_size=64, seed=42) -> (W, b)` — multinomial
  logistic regression, loss `−(1/N) Σ log p_{i,y_i} + (λ/2)‖W‖²`, mini-batch SGD with
  per-epoch reshuffling. Column `k` of `W`, `b` corresponds to the k-th smallest label.
- `posterior(X, W, b) -> P`.
- `pooled_prior(P, H, W, r=2, omega=0.5, pool="log", eps=1e-6) -> pi` — clips P to `[eps, 1]`,
  pools in the log domain, renormalises rows, clips again; image borders average over
  the neighbours that exist.
- `guided_fcm(X, G, U0, m=2.0, eps=1e-4, max_iter=10000) -> (U, V, n_iter)` — any fixed
  `(N, C)` offset `G ≥ 0` works; `G = 0` is plain FCM started from `U0`.
- `theta_scales(X, y, folds=5, epochs=1000, seed=42) -> dict(S_d, S_g, ratio)`; `alpha_from_theta(theta, ratio)`;
  `guidance_share(X, U, V, G, m)` (realised share Σu^m G / Σu^m(d²+G)); `objective(X, U, V, G, m)` (value of `J_m`).
- `sr_ssfcm(X, y, n_clusters=None, **kw)` = `sw_ssfcm(..., r=0)`.
- `open_set_prior(P, seen, n_clusters, H, W, r=2, omega=0.5, mode="maha", X=None, y=None, logits=None) -> pi`
  — when the labelled set covers only the classes `seen` (P has `len(seen)` columns in
  that order): seen columns get the pooled prior, each unseen column a rank-normalised
  novelty score in `(0, 1]` (`"noop"` = 1, `"energy"` = rank of `−log Σ_k e^{z_ik}` —
  needs `logits = X @ W + b`; `"maha"` = rank of the smallest diagonal Mahalanobis
  distance to a seen class mean — needs `X, y`). Feed the result to
  `sw_ssfcm(..., n_clusters=C, P=P, prior=pi, theta=…)`; the paper uses θ = 0.99 with
  `"maha"` (the scales are measured on the seen classes; a smaller θ trades seen-class
  accuracy for unseen-class recall because the unseen clusters have only geometric evidence).

## `metrics`

`evaluate(y_true, y_pred) -> {'acc','nmi','f1'}` (ACC and macro-F1 after Hungarian
matching; inputs must be `0 … C-1`, no `-1` — mask first), `accuracy`, `nmi`,
`macro_f1`, `hungarian_remap`, `xie_beni(X, U, V, m=2.0)`, `centroids_from_u(X, U, m)`.

## Preparing your own cube

`sw_ssfcm` consumes plain NumPy arrays, so there is no data loader to depend on:

- reshape an `H × W × d` image to `X = cube.reshape(H * W, d)` (row-major, the default
  of `numpy.reshape`);
- standardise the bands: `X = (X - X.mean(0)) / X.std(0)` (guard zero-variance bands);
- build the partial-label vector `y` of length `N = H * W`: class index `0 … C-1` for the
  labelled pixels, `-1` everywhere else. `demo.py` contains a 10-line `stratified_labels`
  helper (at most `n` labels per class, seeded) you can copy.

## Failure modes

| Symptom | Cause / fix |
|---|---|
| `ValueError: prior has … columns, n_clusters=…` | `P`/`prior` width must equal the number of classes present in `y` (closed set) or `n_clusters` (open set) |
| `reshape` error inside `pooled_prior` | `H*W != N` or pixels not row-major |
| `evaluate` raises on negative labels | mask to ground-truth pixels first: `m = y_true >= 0` |
| all pixels in one cluster / NaN | unstandardised bands with huge scale (standardise), or `theta` so close to 1 that `α·ln(1e-6)` overflows the distance scale — keep `theta ≤ 0.999` |
| Sw-SSFCM ≈ Softmax, no spatial gain | `r = 0` or `omega = 1`; or the scene has no spatial structure (pixels shuffled) |
| slow | Softmax `epochs=10000` dominates on small budgets; use `softmax_kw=dict(epochs=1000)` for exploration — the guided FCM itself converges in < 10 iterations |
