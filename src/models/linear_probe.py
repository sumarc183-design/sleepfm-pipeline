"""Sonde linéaire = LogisticRegression sur les embeddings figés, comme dans
4_classification_eval_pretraining.py du dépôt officiel SleepFM (pas une
couche PyTorch entraînée par SGD : c'est bien une régression logistique
sklearn qui fait office de "linear probe" dans la littérature)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 100,
) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_probe(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def compare_to_published(results: dict, published_kappa: float, tolerance: float = 0.05) -> bool:
    delta = abs(results["cohen_kappa"] - published_kappa)
    reproduced = delta <= tolerance
    status = "REPRODUIT" if reproduced else "ÉCART SIGNIFICATIF"
    print(f"κ obtenu = {results['cohen_kappa']:.3f} | κ publié = {published_kappa:.3f} | {status}")
    return reproduced
