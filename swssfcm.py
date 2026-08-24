"""Sw-SSFCM — spatially pooled, Softmax-guided semi-supervised fuzzy c-means.

NumPy reference implementation of the algorithm proposed in the paper.  Notation:
X ∈ R^{N×d} are the pixels of an H×W image in row-major order, y ∈ {-1,0,…,C-1}^N the
partial labels (-1 = unlabelled), U ∈ [0,1]^{N×C} the memberships, V ∈ R^{C×d} the
centroids, N_i the (2r+1)²−1 neighbours of pixel i.

    p_ik   = exp(z_ik) / Σ_l exp(z_il),   z_i = Wᵀ x_i + b           (Softmax posterior)
    ln π̃_ik = ω ln p_ik + (1−ω)/|N_i| Σ_{j∈N_i} ln p_jk,  π_i = π̃_i / Σ_k π̃_ik   (log-opinion pool)
    J_m(U,V) = Σ_i Σ_k u_ik^m [ ‖x_i − v_k‖² − α ln π_ik ]                        (objective)
    u_ik = d̂_ik^{−1/(m−1)} / Σ_l d̂_il^{−1/(m−1)},   d̂_ik = ‖x_i − v_k‖² − α ln π_ik
    v_k  = Σ_i u_ik^m x_i / Σ_i u_ik^m
    α(θ) = θ/(1−θ) · S_d / S_g,   θ ∈ [0,1)                          (measured-scale rule)
    S_d = mean_{i∈L,k} ‖x_i − μ_k‖²,  S_g = mean_{i∈L,k} −ln p̂_ik  (p̂ = out-of-fold posterior on L)

π is computed once from the Softmax posterior and kept fixed, so both updates are closed
form for every m > 1.  Sr-SSFCM is the special case r = 0 (π = p).
"""

import numpy as np

EPS_P = 1e-6     # clipping of posteriors / priors  (ε_P)
EPS_D = 1e-12    # guard against division by zero


# --------------------------------------------------------------------------- Softmax
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def train_softmax(X, y, lr=0.01, l2=1e-4, epochs=10000, batch_size=64, seed=42):
    """Multinomial logistic regression: cross-entropy + (λ/2)‖W‖², mini-batch SGD.

    Returns (W, b), W ∈ R^{d×C}, b ∈ R^{1×C}; column k ↔ k-th smallest label in y.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    Y = np.eye(len(classes))[np.searchsorted(classes, y)]
    N, d = X.shape
    W = np.zeros((d, len(classes)))
    b = np.zeros((1, len(classes)))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = order[s:s + batch_size]
            G = (softmax(X[idx] @ W + b) - Y[idx]) / len(idx)          # ∂L/∂z
            W -= lr * (X[idx].T @ G + l2 * W)
            b -= lr * G.sum(axis=0, keepdims=True)
    return W, b


def posterior(X, W, b):
    return softmax(np.asarray(X, dtype=float) @ W + b)


# --------------------------------------------------------------------------- pooling
def neighbour_mean(A, H, W, r):
    """(1/|N_i|) Σ_{j∈N_i} a_j over the (2r+1)² window without its centre; image borders
    average over the neighbours that exist.  A: (N, C) row-major → (N, C)."""
    C = A.shape[1]
    L = A.reshape(H, W, C)
    S = np.zeros_like(L)
    n = np.zeros((H, W, 1))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            S[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] += \
                L[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
            n[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] += 1
    return (S / np.maximum(n, 1)).reshape(-1, C)


def pooled_prior(P, H, W, r=2, omega=0.5, pool="log", eps=EPS_P):
    """π (N, C) from the posterior P (N, C).  pool="log": geometric (log-opinion) pool;
    pool="arith": linear opinion pool (ablation only).  r ≤ 0 or ω ≥ 1 gives π = p."""
    P = np.asarray(P, dtype=float)
    if r <= 0 or omega >= 1.0:
        pi = np.clip(P, eps, 1.0)
        return pi / pi.sum(axis=1, keepdims=True)
    if pool == "log":
        lnP = np.log(np.clip(P, eps, 1.0))
        ln_pi = omega * lnP + (1.0 - omega) * neighbour_mean(lnP, H, W, r)
        pi = np.exp(ln_pi - ln_pi.max(axis=1, keepdims=True))
    elif pool == "arith":
        pi = np.clip(omega * P + (1.0 - omega) * neighbour_mean(P, H, W, r), eps, None)
    else:
        raise ValueError("pool must be 'log' or 'arith'")
    pi /= pi.sum(axis=1, keepdims=True)
    return np.clip(pi, eps, 1.0)


# --------------------------------------------------------------------------- θ rule
def theta_scales(X, y, folds=5, epochs=1000, seed=42, eps=EPS_P, softmax_kw=None):
    """Measured scales of the two cost terms on the labelled pixels L (labels ≥ 0):
    S_d = mean over (i∈L, k) of ‖x_i − μ_k‖² (μ_k = labelled class means) and S_g = mean over
    (i∈L, k) of −ln p̂_ik with p̂ the OUT-OF-FOLD Softmax posterior (stratified `folds`,
    reduced when a class has fewer samples; in-sample if < 2).  Returns dict(S_d, S_g, ratio).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    lab = np.where(y >= 0)[0]
    Xl, yl = X[lab], y[lab]
    classes = np.unique(yl)
    yl = np.searchsorted(classes, yl)                       # → 0..C_seen−1
    Cs = len(classes)
    mu = np.stack([Xl[yl == k].mean(axis=0) for k in range(Cs)])
    S_d = float(((Xl[:, None, :] - mu[None, :, :]) ** 2).sum()) / (len(lab) * Cs)
    kw = {'epochs': epochs, 'seed': seed, **(softmax_kw or {})}   # softmax_kw may override epochs
    n_splits = int(min(folds, np.bincount(yl, minlength=Cs).min()))
    if n_splits < 2:
        Wt, b = train_softmax(Xl, yl, **kw)
        P_oof = posterior(Xl, Wt, b)
    else:                                                   # stratified folds, deterministic
        rng = np.random.default_rng(seed)
        fold = np.empty(len(yl), dtype=int)
        for k in range(Cs):
            idx = rng.permutation(np.where(yl == k)[0])
            fold[idx] = np.arange(len(idx)) % n_splits
        P_oof = np.zeros((len(yl), Cs))
        for f in range(n_splits):
            tr, va = fold != f, fold == f
            Wt, b = train_softmax(Xl[tr], yl[tr], **kw)
            P_oof[va] = posterior(Xl[va], Wt, b)
    S_g = float(-np.log(np.clip(P_oof, eps, 1.0)).sum()) / (len(lab) * Cs)
    return dict(S_d=S_d, S_g=S_g, ratio=S_d / S_g)


def alpha_from_theta(theta, ratio):
    """α = θ/(1−θ) · S_d/S_g  (ratio = S_d/S_g from `theta_scales`)."""
    return theta / (1.0 - theta) * ratio


def guidance_share(X, U, V, G, m=2.0):
    """Share of the guidance term in the cost actually realised: Σ u^m G / Σ u^m (‖x−v‖² + G)."""
    Um = np.asarray(U, dtype=float) ** m
    num = float((Um * G).sum())
    return num / float((Um * (sq_distances(X, V) + G)).sum())


# --------------------------------------------------------------------------- guided FCM


def objective(X, U, V, G, m=2.0):
    """J_m(U,V) = Σ_i Σ_k u_ik^m (‖x_i − v_k‖² + G_ik),  G = −α ln π."""
    X = np.asarray(X, dtype=float)
    D = sq_distances(X, V) + G
    return float(np.sum(U ** m * D))


def sq_distances(X, V):
    x2 = (X * X).sum(axis=1, keepdims=True)
    v2 = (V * V).sum(axis=1)[None, :]
    return np.maximum(x2 + v2 - 2.0 * X @ V.T, 0.0)


def guided_fcm(X, G, U0, m=2.0, eps=1e-4, max_iter=10000):
    """Alternating minimisation of J_m with the fixed guidance matrix G = −α ln π (N×C).

    U⁽⁰⁾ = U0;  for t = 1, 2, …:  V⁽ᵗ⁾ ← v_k(U⁽ᵗ⁻¹⁾),  U⁽ᵗ⁾ ← u_ik(V⁽ᵗ⁾);
    stop when max_{i,k} |u_ik⁽ᵗ⁾ − u_ik⁽ᵗ⁻¹⁾| < eps.  Returns (U, V, n_iter).
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U0, dtype=float)
    e = -1.0 / (m - 1.0)
    for t in range(1, max_iter + 1):
        Um = U ** m
        V = (Um.T @ X) / np.maximum(Um.sum(axis=0)[:, None], EPS_D)       # v_k
        A = np.maximum(sq_distances(X, V) + G, EPS_D) ** e                 # d̂_ik^{−1/(m−1)}
        U_new = A / A.sum(axis=1, keepdims=True)                            # u_ik
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < eps:
            break
    return U, V, t


THETA_DEFAULT = 0.99   # global guidance share selected by leave-one-scene-out in the paper


def sw_ssfcm(X, y, H, W, n_clusters=None, theta=THETA_DEFAULT, r=2, omega=0.5, m=2.0, eps=1e-4,
             max_iter=10000, pool="log", P=None, prior=None, ratio=None, seed=42, softmax_kw=None):
    """Sw-SSFCM on an H×W image given as X (N, d) with partial labels y (N,).

    theta : guidance share θ ∈ [0,1); α = θ/(1−θ)·S_d/S_g with the scales measured on the
            labelled pixels (`theta_scales`) — pass `ratio` to reuse a measured S_d/S_g.
    P     : optional precomputed posterior (N, C) — lets several variants share one Softmax.
    prior : optional precomputed π (N, n_clusters) — replaces the pooling step (open-set).
    Returns dict(U, V, labels, pi, P, alpha, ratio, share_g, n_iter).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    lab = y >= 0
    if P is None:
        Wt, b = train_softmax(X[lab], y[lab], seed=seed, **(softmax_kw or {}))
        P = posterior(X, Wt, b)
    if prior is None:
        pi = pooled_prior(P, H, W, r, omega, pool)
    else:
        pi = np.clip(np.asarray(prior, dtype=float), EPS_P, 1.0)
    C = pi.shape[1] if n_clusters is None else int(n_clusters)
    if pi.shape[1] != C:
        raise ValueError(f"prior has {pi.shape[1]} columns, n_clusters={C}")
    if ratio is None:
        ratio = theta_scales(X, y, seed=seed, softmax_kw=softmax_kw)["ratio"]
    alpha = alpha_from_theta(theta, ratio)
    G = -alpha * np.log(pi)
    U, V, t = guided_fcm(X, G, pi, m, eps, max_iter)
    return dict(U=U, V=V, labels=U.argmax(axis=1), pi=pi, P=P, alpha=alpha, ratio=ratio,
                share_g=guidance_share(X, U, V, G, m), n_iter=t)


def sr_ssfcm(X, y, n_clusters=None, **kw):
    """Sr-SSFCM = Sw-SSFCM with r = 0 (π = p): no spatial pooling."""
    return sw_ssfcm(X, y, H=None, W=None, n_clusters=n_clusters, r=0, **kw)


# --------------------------------------------------------------------------- open set
def _rank01(v):
    ranks = v.argsort().argsort().astype(float)
    return np.clip((ranks + 1.0) / len(v), EPS_P, 1.0)


def open_set_prior(P, seen, n_clusters, H, W, r=2, omega=0.5, mode="maha",
                   X=None, y=None, logits=None):
    """π for partial class coverage: the Softmax knows only the classes `seen` (columns of
    P in that order).  Seen columns ← pooled prior of P; each unseen column ← a rank
    normalised novelty score in (0,1]:
      mode="noop"   : 1                     (purely geometric competition, −ln π = 0)
      mode="energy" : rank(−log Σ_k e^{z_ik})                          (needs logits)
      mode="maha"   : rank(min_{k∈seen} Σ_j (x_ij − μ_kj)² / σ²_kj)   (needs X, y)
    """
    pi = np.ones((len(P), n_clusters))
    pi[:, seen] = pooled_prior(P, H, W, r, omega)
    unseen = np.setdiff1d(np.arange(n_clusters), seen)
    if mode == "energy":
        zmax = logits.max(axis=1, keepdims=True)
        score = -(zmax[:, 0] + np.log(np.exp(logits - zmax).sum(axis=1)))
    elif mode == "maha":
        score = np.full(len(X), np.inf)
        for k in seen:
            xs = X[y == k]
            var = xs.var(axis=0) + 1e-3 if len(xs) > 1 else np.ones(X.shape[1])
            score = np.minimum(score, (((X - xs.mean(axis=0)) ** 2) / var).sum(axis=1))
    elif mode != "noop":
        raise ValueError("mode must be 'noop', 'energy' or 'maha'")
    if mode != "noop":
        pi[:, unseen] = _rank01(score)[:, None]
    return pi
