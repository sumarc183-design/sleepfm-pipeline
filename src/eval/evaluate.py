"""Métriques d'évaluation pour la classification des stades du sommeil."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_report": classification_report(y_true, y_pred, output_dict=True),
    }


def compare_to_published(our_results: dict, published_kappa: float, tolerance: float = 0.05) -> bool:
    """Vérifie que notre reproduction est dans la tolérance du chiffre publié (κ)."""
    delta = abs(our_results["cohen_kappa"] - published_kappa)
    reproduced = delta <= tolerance
    status = "REPRODUIT" if reproduced else "ÉCART SIGNIFICATIF"
    print(f"κ obtenu = {our_results['cohen_kappa']:.3f} | κ publié = {published_kappa:.3f} | {status}")
    return reproduced
