"""Clustering evaluation metrics used throughout this repository.

Clustering produces cluster ids that are only defined up to a permutation, so
accuracy is computed after an optimal cluster-to-class assignment obtained with
the Hungarian algorithm. NMI and ARI are permutation-invariant by construction
and are therefore computed directly.
"""
from __future__ import annotations

import numpy as np


def hungarian_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the cluster-id -> class-id map maximizing the number of matches.

    Both label vectors must be 0-indexed and contain no negative entries.
    """
    from scipy.optimize import linear_sum_assignment

    n = int(max(y_true.max(), y_pred.max())) + 1
    contingency = np.zeros((n, n), dtype=np.int64)
    np.add.at(contingency, (y_pred.astype(int), y_true.astype(int)), 1)
    rows, cols = linear_sum_assignment(-contingency)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def remap_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Relabel predicted clusters with their best-matching ground-truth class."""
    mapping = hungarian_mapping(y_true, y_pred)
    return np.array([mapping[int(p)] for p in y_pred], dtype=int)


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy after optimal cluster-to-class assignment."""
    return float(np.mean(remap_labels(y_true, y_pred) == y_true.astype(int)))


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mutual information."""
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(y_true, y_pred))


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand index."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(y_true, y_pred))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute ACC, NMI and ARI in one pass. Returns a dict of floats."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "acc": cluster_accuracy(y_true, y_pred),
        "nmi": nmi(y_true, y_pred),
        "ari": ari(y_true, y_pred),
    }
