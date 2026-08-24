"""Evaluation metrics.

Cluster ids are defined up to a permutation, so ACC and macro-F1 are computed after the
optimal cluster→class assignment (Hungarian algorithm).  NMI is permutation invariant.
Xie–Beni uses the fuzzy partition:  XB = Σ_i Σ_k u_ik^m ‖x_i − v_k‖² / (N · min_{k≠l} ‖v_k − v_l‖²).
"""

import numpy as np


def hungarian_remap(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = int(max(y_true.max(), y_pred.max())) + 1
    M = np.zeros((n, n), dtype=np.int64)
    np.add.at(M, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(-M)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    return np.array([mapping[p] for p in y_pred], dtype=int)


def accuracy(y_true, y_pred):
    return float(np.mean(hungarian_remap(y_true, y_pred) == np.asarray(y_true, dtype=int)))


def nmi(y_true, y_pred):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, hungarian_remap(y_true, y_pred), average="macro"))


def evaluate(y_true, y_pred):
    """ACC, NMI and macro-F1 on the given label vectors (both 0-indexed, no −1)."""
    return {"acc": accuracy(y_true, y_pred), "nmi": nmi(y_true, y_pred),
            "f1": macro_f1(y_true, y_pred)}


def xie_beni(X, U, V, m=2.0):
    from scipy.spatial.distance import cdist, pdist
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    comp = float(np.sum(U ** m * cdist(X, V, "sqeuclidean")))
    sep = float(np.min(pdist(V) ** 2))
    return comp / (len(X) * sep) if sep > 0 else float("inf")


def centroids_from_u(X, U, m=2.0):
    """v_k = Σ_i u_ik^m x_i / Σ_i u_ik^m — used to evaluate XB for a classifier whose
    posterior plays the role of U."""
    Um = np.asarray(U, dtype=float) ** m
    return (Um.T @ np.asarray(X, dtype=float)) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)
