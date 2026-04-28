"""Sw-SSFCM — Spatial-weighted Softmax-Embedded Semi-Supervised FCM (ISI 2026).

Extends SeFCM with neighborhood-aware guidance for image data:

    p_hat_{ik} = omega_i * p_{ik} + (1 - omega_i) * u_tilde_{ik}
    omega_i    = max_k p_{ik} / max_i max_k p_{ik}
    u_tilde_{ik} = (1 / |N_r(i)|) * sum_{j in N_r(i)} u_{jk}
    d_hat^2_{ik} = || x_i - v_k ||^2 - alpha * ln p_hat_{ik}

with neighborhood radius ``r in {1, 2}`` (window (2r+1)^2, center excluded).

See ``paper_final/algorithms/alg-swssfcm.tex`` for pseudocode.
"""
from __future__ import annotations

import numpy as np

from _common import (
    EPS_DIV,
    EPS_P,
    frobenius_delta,
    init_centroids_from_labels,
    pairwise_sq_dist,
    update_centroids,
    update_membership_from_D2,
)
from sefcm import SeFCM


class SwSSFCM(SeFCM):
    """Sw-SSFCM clustering for image-shaped data.

    Parameters
    ----------
    n_clusters : int
    alpha : float, default=1.0
    radius : int, default=1
        Spatial neighborhood radius (1 or 2).
    m, max_iter, tol, softmax_*, random_state, verbose
        See :class:`SeFCM`.

    Attributes (after ``fit``)
    --------------------------
    U_, V_, labels_, n_iter_, P_ : as in SeFCM
    omega_ : ndarray (N, 1)
    """

    def __init__(
        self,
        n_clusters: int,
        alpha: float = 1.0,
        radius: int = 1,
        m: float = 2.0,
        max_iter: int = 10000,
        tol: float = 1e-4,
        softmax_lr: float = 0.01,
        softmax_l2: float = 1e-4,
        softmax_max_epoch: int = 1000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            n_clusters=n_clusters,
            alpha=alpha,
            m=m,
            max_iter=max_iter,
            tol=tol,
            softmax_lr=softmax_lr,
            softmax_l2=softmax_l2,
            softmax_max_epoch=softmax_max_epoch,
            random_state=random_state,
            verbose=verbose,
        )
        if radius not in (1, 2):
            raise ValueError(f"radius must be 1 or 2, got {radius}")
        self.radius = radius

    # ------------------------------------------------------------------
    def _spatial_average(self, U: np.ndarray, H: int, W: int) -> np.ndarray:
        """``u_tilde`` via vectorized neighborhood shifting (center excluded)."""
        N, C = U.shape
        r = self.radius
        U_img = U.reshape(H, W, C)
        spatial_sum = np.zeros((H, W, C), dtype=np.float64)
        neighbor_count = np.zeros((H, W, 1), dtype=np.float64)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                src_y0, src_y1 = max(0, -dy), H + min(0, -dy)
                src_x0, src_x1 = max(0, -dx), W + min(0, -dx)
                tgt_y0, tgt_y1 = max(0, dy), H + min(0, dy)
                tgt_x0, tgt_x1 = max(0, dx), W + min(0, dx)
                if src_y1 <= src_y0 or src_x1 <= src_x0:
                    continue
                spatial_sum[tgt_y0:tgt_y1, tgt_x0:tgt_x1, :] += (
                    U_img[src_y0:src_y1, src_x0:src_x1, :]
                )
                neighbor_count[tgt_y0:tgt_y1, tgt_x0:tgt_x1, :] += 1

        neighbor_count = np.maximum(neighbor_count, 1.0)
        return (spatial_sum / neighbor_count).reshape(N, C)

    @staticmethod
    def _adaptive_omega(P: np.ndarray) -> np.ndarray:
        conf = P.max(axis=1)
        max_conf = max(conf.max(), EPS_DIV)
        return (conf / max_conf)[:, None]            # (N, 1)

    @staticmethod
    def _combine(P: np.ndarray, U_tilde: np.ndarray, omega: np.ndarray) -> np.ndarray:
        P_hat = omega * P + (1.0 - omega) * U_tilde
        row_sum = np.maximum(P_hat.sum(axis=1, keepdims=True), EPS_DIV)
        return P_hat / row_sum

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y_partial: np.ndarray,
        image_shape: tuple,
    ) -> "SwSSFCM":
        """Fit on flattened image data ``X`` (N, D) with ``image_shape=(H, W)``.

        Pixels are assumed to be in row-major order so that
        ``X.reshape(H, W, D)`` reconstructs the image.
        """
        X = np.asarray(X, dtype=np.float64)
        y_partial = np.asarray(y_partial).astype(int)
        H, W = image_shape
        if H * W != X.shape[0]:
            raise ValueError(
                f"image_shape {image_shape} incompatible with N={X.shape[0]}"
            )

        clf = self._train_softmax(X, y_partial)
        P = np.clip(clf.predict_proba(X), EPS_P, 1.0)
        self.P_ = P
        self._softmax = clf

        omega = self._adaptive_omega(P)               # Step 22 (alg-swssfcm.tex)
        self.omega_ = omega

        U = P.copy()
        V = init_centroids_from_labels(X, y_partial, self.n_clusters)
        alpha = self.alpha

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            U_tilde = self._spatial_average(U, H, W)  # Step 21
            P_hat = self._combine(P, U_tilde, omega)  # Step 23
            P_hat = np.clip(P_hat, EPS_P, 1.0)

            D2 = pairwise_sq_dist(X, V)               # Step 24
            if alpha != 0.0:
                D2 = D2 - alpha * np.log(P_hat)

            U = update_membership_from_D2(D2, self.m)         # Step 25
            V_new = update_centroids(X, U, self.m)
            delta = frobenius_delta(V_new, V)
            if self.verbose:
                print(f"[Sw-SSFCM] iter={n_iter} dV={delta:.2e}")
            if delta < self.tol:                              # Step 26
                V = V_new
                break
            V = V_new

        self.U_ = U
        self.V_ = V
        self.labels_ = U.argmax(axis=1)
        self.n_iter_ = n_iter
        return self

    # ------------------------------------------------------------------
    def fit_predict(self, X, y_partial, image_shape):
        return self.fit(X, y_partial, image_shape).labels_
