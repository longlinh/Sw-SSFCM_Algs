"""SeFCM — Softmax-Embedded Semi-Supervised Fuzzy C-Means.

Distance with Softmax guidance:

    d^2_{ik} = || x_i - v_k ||^2 - alpha * ln p_{ik}

where p_{ik} is the softmax probability that point i belongs to class k,
estimated by a multinomial logistic regressor trained on the labeled subset.

Implements Algorithm 1 of the paper; step numbers in the comments below refer
to that pseudocode.
"""
from __future__ import annotations

import numpy as np

from _common import (
    EPS_P,
    SoftmaxRegression,
    frobenius_delta,
    init_centroids_from_labels,
    pairwise_sq_dist,
    update_centroids,
    update_membership_from_D2,
)


class SeFCM:
    """SeFCM clustering with sklearn-like API.

    Parameters
    ----------
    n_clusters : int
        Number of clusters C.
    alpha : float, default=1.0
        Softmax-guidance weight. ``alpha=0`` recovers plain FCM.
    m : float, default=2.0
        Fuzzifier.
    max_iter : int, default=10000
        Maximum FCM iterations.
    tol : float, default=1e-4
        Convergence threshold on ||V_new - V_old||_F.
    softmax_lr, softmax_l2, softmax_max_epoch : floats / int
        Softmax-regression hyperparameters.
    softmax_tol : float, default=0.0
        Early-stopping threshold on the Softmax loss decrease. The default of
        ``0.0`` disables early stopping, so the classifier always runs the full
        ``softmax_max_epoch`` epochs; this is the setting used for the results
        reported in the paper. Raise it (e.g. ``1e-6``) to trade a little
        accuracy for a shorter fit.
    random_state : int, default=42
    verbose : bool, default=False

    Attributes (after ``fit``)
    --------------------------
    U_ : ndarray (N, C)
    V_ : ndarray (C, D)
    labels_ : ndarray (N,)
    P_ : ndarray (N, C)
    n_iter_ : int
    """

    def __init__(
        self,
        n_clusters: int,
        alpha: float = 1.0,
        m: float = 2.0,
        max_iter: int = 10000,
        tol: float = 1e-4,
        softmax_lr: float = 0.01,
        softmax_l2: float = 1e-4,
        softmax_max_epoch: int = 10000,
        softmax_tol: float = 0.0,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.softmax_lr = softmax_lr
        self.softmax_l2 = softmax_l2
        self.softmax_max_epoch = softmax_max_epoch
        self.softmax_tol = softmax_tol
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _train_softmax(self, X: np.ndarray, y_partial: np.ndarray):
        mask = y_partial >= 0
        clf = SoftmaxRegression(
            lr=self.softmax_lr,
            l2=self.softmax_l2,
            max_epoch=self.softmax_max_epoch,
            tol=self.softmax_tol,
            random_state=self.random_state,
        )
        clf.fit(X[mask], y_partial[mask])
        return clf

    def _build_cost_fn(self, P: np.ndarray):
        log_P = np.log(np.maximum(P, EPS_P))
        alpha = self.alpha

        def cost_fn(X, V):
            d2 = pairwise_sq_dist(X, V)
            if alpha == 0.0:
                return d2
            return d2 - alpha * log_P
        return cost_fn

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y_partial: np.ndarray) -> "SeFCM":
        """Fit on data ``X`` (N, D) with partial labels ``y_partial`` (N,).

        ``y_partial[i] = -1`` marks unlabeled points; class indices in [0, C-1].
        """
        X = np.asarray(X, dtype=np.float64)
        y_partial = np.asarray(y_partial).astype(int)
        if X.shape[0] != y_partial.shape[0]:
            raise ValueError("X and y_partial must have same length.")

        # Step 13: train Softmax on labeled subset.
        clf = self._train_softmax(X, y_partial)
        # Step 14: predict probabilities on all points.
        P = clf.predict_proba(X)
        # Step 15: lower-bound to avoid log(0).
        P = np.clip(P, EPS_P, 1.0)
        self.P_ = P
        self._softmax = clf

        # Steps 16-18: init U from P, V from labeled means.
        U = P.copy()
        V = init_centroids_from_labels(X, y_partial, self.n_clusters)

        cost_fn = self._build_cost_fn(P)

        # Steps 21-26: guided FCM loop.
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            D2 = cost_fn(X, V)
            U = update_membership_from_D2(D2, self.m)
            V_new = update_centroids(X, U, self.m)
            delta = frobenius_delta(V_new, V)
            if self.verbose:
                print(f"[SeFCM] iter={n_iter} dV={delta:.2e}")
            if delta < self.tol:
                V = V_new
                break
            V = V_new

        self.U_ = U
        self.V_ = V
        self.labels_ = U.argmax(axis=1)
        self.n_iter_ = n_iter
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard cluster labels via nearest centroid in *guided* distance."""
        X = np.asarray(X, dtype=np.float64)
        if self.alpha == 0.0 or not hasattr(self, "_softmax"):
            D2 = pairwise_sq_dist(X, self.V_)
        else:
            P = np.clip(self._softmax.predict_proba(X), EPS_P, 1.0)
            D2 = pairwise_sq_dist(X, self.V_) - self.alpha * np.log(P)
        return D2.argmin(axis=1)

    def fit_predict(self, X: np.ndarray, y_partial: np.ndarray) -> np.ndarray:
        return self.fit(X, y_partial).labels_
