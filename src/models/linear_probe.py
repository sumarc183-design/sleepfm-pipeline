"""Sondes sur les embeddings figés.

La baseline est la sonde linéaire = LogisticRegression, comme dans
4_classification_eval_pretraining.py du dépôt officiel SleepFM (pas une
couche PyTorch entraînée par SGD : c'est bien une régression logistique
sklearn qui fait office de "linear probe" dans la littérature) -- c'est
elle qui se compare au chiffre publié.

Les autres sondes sont évaluées à côté, sur exactement les mêmes embeddings
et le même split, chacune pour tester un levier :
  - logistic_balanced : même modèle, classes pondérées (N1 est rare) ;
  - xgboost (le dépôt officiel le propose aussi) : frontière non linéaire
    par arbres ;
  - mlp : frontière non linéaire par réseau, plus adaptée aux embeddings
    denses que les arbres et plus rapide ;
  - sequence : contexte temporel (époques voisines), voir sequence_probe.py.
Hyperparamètres fixes, raisonnables, jamais réglés sur le jeu de test.
(Random Forest, testé, a été retiré : plus lent que XGBoost et moins adapté
aux embeddings denses.)

Entre XGBoost et MLP, une seule sonde non linéaire est gardée, choisie par
validation croisée par sujet sur le train (voir choose_mlp_or_xgboost) : le
test n'intervient qu'à la fin, sur le modèle retenu.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

PROBE_KINDS = ("logistic", "logistic_balanced", "xgboost", "mlp", "sequence")
MIN_XGBOOST_GAIN = 0.02  # F1 macro : en dessous, l'écart relève du bruit du tirage des sujets de test
MAX_F1_GAP = 0.15  # gap F1 apprentissage - validation au-delà duquel une sonde est jugée surapprise
CV_FOLDS = 5


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
                seed: int = 42, device: str = "cpu", groups: tuple[np.ndarray, np.ndarray] | None = None):
    """`groups` = (sujet, numéro d'époque) de chaque ligne, requis par "sequence"."""
    if kind == "logistic":
        return train_linear_probe(X_train, y_train, max_iter=max_iter)
    if kind == "logistic_balanced":
        model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    elif kind == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True,
                              max_iter=100, random_state=seed)
    elif kind == "sequence":
        if groups is None:
            raise ValueError("La sonde 'sequence' a besoin de groups=(sujets, numéros d'époque)")
        from .sequence_probe import SequenceProbe  # import paresseux : torch

        return SequenceProbe(device=device, seed=seed).fit(X_train, y_train, groups)
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


def macro_roc_auc(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC un-contre-tous, moyenne sur les stades présents dans y (un
    stade absent n'a pas d'AUC définie)."""
    aucs = [roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(classes)
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_probe(model, X_test: np.ndarray, y_test: np.ndarray,
                   groups: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    if getattr(model, "needs_groups", False):
        y_pred, proba = model.predict(X_test, groups), model.predict_proba(X_test, groups)
    else:
        y_pred, proba = model.predict(X_test), model.predict_proba(X_test)
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_macro": macro_roc_auc(proba, model.classes_, y_test),
        "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis de validation croisée groupés par sujet : un sujet (une nuit)
    n'est jamais à la fois dans l'apprentissage et la validation d'un pli.
    Renvoie [(indices d'apprentissage, indices de validation), ...]."""
    unique = np.array(sorted(set(subjects)))
    np.random.default_rng(seed).shuffle(unique)
    n_folds = min(n_folds, len(unique))
    fold_of_subject = {subject: k % n_folds for k, subject in enumerate(unique)}
    fold = np.array([fold_of_subject[s] for s in subjects])
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]


def cross_validate_probes(kinds, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, n_folds: int = 5,
                          max_iter: int = 100, seed: int = 42, device: str = "cpu") -> dict:
    """Validation croisée par sujet sur le TRAIN uniquement, mêmes plis pour
    chaque sonde (comparaison appariée). Le jeu de test n'intervient pas :
    il reste réservé à l'évaluation finale du modèle choisi.
    Renvoie {sonde: {"val_f1": [par pli], "train_f1": [par pli], "fit_seconds": total}}."""
    folds = subject_folds(subjects, n_folds, seed)
    results = {kind: {"val_f1": [], "train_f1": [], "fit_seconds": 0.0} for kind in kinds}
    for fit_idx, val_idx in folds:
        for kind in kinds:
            start = time.time()
            model = train_probe(kind, X[fit_idx], y[fit_idx], max_iter=max_iter, seed=seed, device=device)
            results[kind]["fit_seconds"] += time.time() - start
            results[kind]["train_f1"].append(evaluate_probe(model, X[fit_idx], y[fit_idx])["macro_f1"])
            results[kind]["val_f1"].append(evaluate_probe(model, X[val_idx], y[val_idx])["macro_f1"])
    return results


def choose_mlp_or_xgboost(cv: dict, min_gain: float = MIN_XGBOOST_GAIN,
                          max_gap: float = MAX_F1_GAP) -> tuple[str | None, str]:
    """Règle du projet pour garder une seule sonde non linéaire, appliquée à
    la validation croisée par sujet sur le train (cross_validate_probes),
    jamais au test :
    1. une sonde dont le gap F1 moyen (apprentissage - validation) dépasse
       `max_gap` est éliminée, quel que soit son score (elle apprend par cœur) ;
    2. si les deux restent, XGBoost n'est retenu que s'il bat le MLP de plus
       de `min_gain` en F1 macro de validation en moyenne ET sur tous les plis
       sauf un au plus (un gain qui ne tient que sur quelques plis dépend du
       tirage) ; sinon le MLP, plus rapide à entraîner ;
    3. si une seule reste, c'est elle ; si aucune, aucune n'est retenue (on
       s'en tient à la régression logistique).
    Renvoie (sonde retenue ou None, explication)."""
    names = {"mlp": "MLP", "xgboost": "XGBoost"}
    gaps = {kind: float(np.mean(np.array(cv[kind]["train_f1"]) - np.array(cv[kind]["val_f1"])))
            for kind in ("mlp", "xgboost")}
    eliminated = [kind for kind, gap in gaps.items() if gap > max_gap]
    note = "".join(f" ; {names[kind]} éliminé (gap F1 moyen {gaps[kind]:+.4f} > {max_gap})" for kind in eliminated)
    remaining = [kind for kind in ("mlp", "xgboost") if kind not in eliminated]
    if not remaining:
        return None, f"aucune sonde retenue, les deux surapprennent{note}"
    if len(remaining) == 1:
        return remaining[0], f"{names[remaining[0]]} retenu{note}"

    diffs = np.array(cv["xgboost"]["val_f1"]) - np.array(cv["mlp"]["val_f1"])
    mean_gain, wins, n_folds = float(diffs.mean()), int((diffs > 0).sum()), len(diffs)
    detail = f"gain moyen XGBoost - MLP = {mean_gain:+.4f}, XGBoost meilleur sur {wins}/{n_folds} plis"
    if mean_gain > min_gain and wins >= n_folds - 1:
        return "xgboost", f"{detail} : écart net et régulier, XGBoost retenu"
    return "mlp", (f"{detail} : écart trop faible ou irrégulier, MLP retenu, plus rapide "
                   f"({cv['mlp']['fit_seconds']:.0f}s contre {cv['xgboost']['fit_seconds']:.0f}s sur les plis)")


def compare_probes(kinds, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   max_iter: int = 100, seed: int = 42, device: str = "cpu",
                   train_groups: tuple[np.ndarray, np.ndarray] | None = None,
                   test_groups: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Entraîne chaque sonde sur le même train, l'évalue sur le train ET le
    test : l'écart (gap) train - test mesure le surapprentissage.
    `*_groups` = (sujet, numéro d'époque) par ligne, requis par "sequence"."""
    results = {}
    for kind in kinds:
        start = time.time()
        model = train_probe(kind, X_train, y_train, max_iter=max_iter, seed=seed, device=device, groups=train_groups)
        results[kind] = {
            "train": evaluate_probe(model, X_train, y_train, train_groups),
            "test": evaluate_probe(model, X_test, y_test, test_groups),
            "fit_seconds": time.time() - start,
        }
    return results


def compare_to_published(results: dict, published_kappa: float, tolerance: float = 0.05) -> bool:
    delta = abs(results["cohen_kappa"] - published_kappa)
    reproduced = delta <= tolerance
    status = "REPRODUIT" if reproduced else "ÉCART SIGNIFICATIF"
    print(f"κ obtenu = {results['cohen_kappa']:.3f} | κ publié = {published_kappa:.3f} | {status}")
    return reproduced
