"""Internal machinery for SeFCM / Sw-SSFCM reference implementation.

Reviewers reading the paper algorithms can safely ignore this file. It contains
the standard FCM update rules and a from-scratch Softmax Regression
(cross-entropy + L2, mini-batch SGD) used by both proposed algorithms.

See ``sefcm.py`` and ``sw_ssfcm.py`` for the contributions.
"""
from __future__ import annotations

import numpy as np

EPS_P = 1e-12
EPS_DIV = 1e-12


# ---------------------------------------------------------------------------
# Softmax Regression (cross-entropy + L2, mini-batch SGD)
# ---------------------------------------------------------------------------
class SoftmaxRegression:
    """Multinomial logistic regression trained with mini-batch SGD."""

    def __init__(
        self,
        lr: float = 0.01,
        l2: float = 1e-4,
        max_epoch: int = 1000,
        batch_size: int = 64,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.lr = lr
        self.l2 = l2
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self.W_ = None  # (D, C)
        self.b_ = None  # (1, C)
        self.n_classes_ = None

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), EPS_DIV)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        N, D = X.shape
        C = int(y.max()) + 1
        self.n_classes_ = C
        Y = np.eye(C)[y]  # one-hot

        rng = np.random.default_rng(self.random_state)
        self.W_ = np.zeros((D, C), dtype=np.float64)
        self.b_ = np.zeros((1, C), dtype=np.float64)

        prev_loss = np.inf
        bs = min(self.batch_size, N)
        for _ in range(self.max_epoch):
            idx = rng.permutation(N)
            for s in range(0, N, bs):
                sl = idx[s : s + bs]
                Xb, Yb = X[sl], Y[sl]
                P = self._softmax(Xb @ self.W_ + self.b_)
                err = (P - Yb) / Xb.shape[0]
                self.W_ -= self.lr * (Xb.T @ err + self.l2 * self.W_)
                self.b_ -= self.lr * err.sum(axis=0, keepdims=True)
            P_full = self._softmax(X @ self.W_ + self.b_)
            loss = -np.mean(np.log(np.maximum(P_full[np.arange(N), y], EPS_P)))
            loss += 0.5 * self.l2 * np.sum(self.W_ ** 2)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._softmax(X @ self.W_ + self.b_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# FCM core (functional, NumPy only)
# ---------------------------------------------------------------------------
def pairwise_sq_dist(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, shape (N, C)."""
    # ||x - v||^2 = ||x||^2 + ||v||^2 - 2 x·v
    x2 = np.sum(X * X, axis=1, keepdims=True)         # (N, 1)
    v2 = np.sum(V * V, axis=1, keepdims=True).T       # (1, C)
    d2 = x2 + v2 - 2.0 * X @ V.T
    return np.maximum(d2, 0.0)


def update_membership_from_D2(D2: np.ndarray, m: float) -> np.ndarray:
    """U_ik = 1 / Σ_j (D2_ik / D2_ij) ** (1/(m-1))."""
    D2 = np.maximum(D2, EPS_DIV)
    exponent = 1.0 / (m - 1.0)
    inv = D2 ** (-exponent)                           # (N, C)
    return inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS_DIV)


def update_centroids(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """V_k = Σ_i u_ik^m x_i / Σ_i u_ik^m."""
    Um = U ** m
    num = Um.T @ X                                    # (C, D)
    den = np.maximum(Um.sum(axis=0, keepdims=True).T, EPS_DIV)
    return num / den


def init_centroids_from_U(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    return update_centroids(X, U, m)


def init_centroids_from_labels(
    X: np.ndarray, y_partial: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Class-mean init using labeled subset; fallback to global mean for empty classes."""
    D = X.shape[1]
    V = np.zeros((n_clusters, D), dtype=np.float64)
    mask = y_partial >= 0
    Xl, yl = X[mask], y_partial[mask]
    global_mean = X.mean(axis=0)
    for k in range(n_clusters):
        sel = yl == k
        V[k] = Xl[sel].mean(axis=0) if np.any(sel) else global_mean
    return V


def frobenius_delta(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B))


def fcm_loop(
    X: np.ndarray,
    V0: np.ndarray,
    m: float,
    max_iter: int,
    tol: float,
    cost_fn,
) -> tuple:
    """Generic FCM loop with pluggable cost function.

    cost_fn(X, V) -> (N, C) squared-distance-like matrix.
    Returns (U, V, n_iter).
    """
    V = V0.astype(np.float64, copy=True)
    U = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        D2 = cost_fn(X, V)
        U = update_membership_from_D2(D2, m)
        V_new = update_centroids(X, U, m)
        if frobenius_delta(V_new, V) < tol:
            V = V_new
            break
        V = V_new
    return U, V, n_iter


# ---------------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 0.5, (50, 2)),
        rng.normal(5, 0.5, (50, 2)),
        rng.normal((0, 5), 0.5, (50, 2)),
    ])
    V0 = X[rng.choice(150, 3, replace=False)]
    U, V, it = fcm_loop(X, V0, m=2.0, max_iter=300, tol=1e-6, cost_fn=pairwise_sq_dist)
    print(f"FCM converged in {it} iters, V=\n{V}")
