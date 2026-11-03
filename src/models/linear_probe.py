"""Sondes sur les embeddings figés.

La baseline est la sonde linéaire = LogisticRegression, comme dans
4_classification_eval_pretraining.py du dépôt officiel SleepFM (pas une
couche PyTorch entraînée par SGD : c'est bien une régression logistique
sklearn qui fait office de "linear probe" dans la littérature) -- c'est
elle qui se compare au chiffre publié.

Random Forest et XGBoost (le dépôt officiel propose aussi XGBoost) sont des
sondes non linéaires évaluées à côté, sur exactement les mêmes embeddings et
le même split, pour voir si une frontière non linéaire apporte quelque chose.
Hyperparamètres fixes, raisonnables, jamais réglés sur le jeu de test.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score

PROBE_KINDS = ("logistic", "random_forest", "xgboost")


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 100,
) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


class _EncodedXGBClassifier:
    """XGBoost exige des labels 0..K-1 contigus : si un stade manque dans le
    train (petits runs), on ré-encode et on redonne les labels d'origine."""

    def __init__(self, **params):
        from xgboost import XGBClassifier  # import paresseux : dépendance optionnelle

        self.model = XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_EncodedXGBClassifier":
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def train_probe(kind: str, X_train: np.ndarray, y_train: np.ndarray, max_iter: int = 100,
                seed: int = 42, device: str = "cpu"):
    if kind == "logistic":
        return train_linear_probe(X_train, y_train, max_iter=max_iter)
    if kind == "random_forest":
        # max_samples / min_samples_leaf : limite le temps (CPU Colab à 2 cœurs)
        # et le surapprentissage d'arbres profonds sur 1536 dimensions.
        model = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=5, max_samples=0.5,
            n_jobs=-1, random_state=seed,
        )
    elif kind == "xgboost":
        model = _EncodedXGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
            tree_method="hist", device="cuda" if device.startswith("cuda") else "cpu",
            random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(f"Sonde inconnue {kind!r} (disponibles : {PROBE_KINDS})")
    model.fit(X_train, y_train)
    return model


def macro_roc_auc(model, X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC un-contre-tous, moyenne sur les stades présents dans y (un
    stade absent n'a pas d'AUC définie)."""
    proba = model.predict_proba(X)
    aucs = [roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(model.classes_)
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_probe(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_macro": macro_roc_auc(model, X_test, y_test),
        "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def compare_probes(kinds, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   max_iter: int = 100, seed: int = 42, device: str = "cpu") -> dict:
    """Entraîne chaque sonde sur le même train, l'évalue sur le train ET le
    test : l'écart (gap) train - test mesure le surapprentissage."""
    results = {}
    for kind in kinds:
        start = time.time()
        model = train_probe(kind, X_train, y_train, max_iter=max_iter, seed=seed, device=device)
        results[kind] = {
            "train": evaluate_probe(model, X_train, y_train),
            "test": evaluate_probe(model, X_test, y_test),
            "fit_seconds": time.time() - start,
        }
    return results


def compare_to_published(results: dict, published_kappa: float, tolerance: float = 0.05) -> bool:
    delta = abs(results["cohen_kappa"] - published_kappa)
    reproduced = delta <= tolerance
    status = "REPRODUIT" if reproduced else "ÉCART SIGNIFICATIF"
    print(f"κ obtenu = {results['cohen_kappa']:.3f} | κ publié = {published_kappa:.3f} | {status}")
    return reproduced
