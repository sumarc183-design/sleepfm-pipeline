"""Sélection pondérée de seuils et métriques de calibration par nuit."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

POINTS = ("initial_f1", "sensitivity_70", "sensitivity_80", "sensitivity_90")
TARGETS = dict(zip(POINTS[1:], (0.7, 0.8, 0.9), strict=True))


def weighted_thresholds(y, probability, weights):
    """Reproduit la duplication de nuits via des poids par fenêtre, sans ajuster de modèle."""
    y, probability, weights = map(np.asarray, (y, probability, weights))
    if (
        y.ndim != 1
        or not len(y)
        or probability.shape != y.shape
        or weights.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or any(weights[y == label].sum() <= 0 for label in (0, 1))
    ):
        raise ValueError("Deux classes pondérées et vecteurs valides requis")
    precision, recall, thresholds = precision_recall_curve(y, probability, sample_weight=weights)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    result = {"initial_f1": float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])}
    positive = (y == 1) & (weights > 0)
    p, w = probability[positive], weights[positive]
    order = np.argsort(-p, kind="stable")
    p, cumulative = p[order], np.cumsum(w[order])
    for name, target in TARGETS.items():
        result[name] = float(p[np.searchsorted(cumulative, target * cumulative[-1], side="left")])
    return result


def binary_counts(y, probability, threshold, weights=None):
    """Matrice [négatif/positif réel, négatif/positif prédit]."""
    y, probability = np.asarray(y), np.asarray(probability)
    if weights is None:
        weights = np.ones(len(y), dtype=np.int64)
    weights = np.asarray(weights)
    if (
        y.ndim != 1
        or probability.shape != y.shape
        or weights.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.equal(weights, np.floor(weights)).all()
    ):
        raise ValueError("Labels, probabilités ou multiplicités invalides")
    code = 2 * y.astype(int) + (probability >= threshold)
    return np.bincount(code, weights=weights, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_counts(matrix):
    tn, fp, fn, tp = np.asarray(matrix).ravel()
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": np.asarray(matrix).tolist(),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "percentile_2_5": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "percentile_97_5": float(np.percentile(finite, 97.5)) if len(finite) else None,
    }
