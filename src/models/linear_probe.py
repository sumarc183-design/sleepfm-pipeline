"""Sondes sur les embeddings figés.

La baseline ("logistic") reproduit la sonde officielle : train_model() de
sleepfm/utils.py (appelé par 4_classification_eval_pretraining.py) entraîne
`LogisticRegression(class_weight="balanced")` -- classes pondérées, solveur
lbfgs. Différence assumée : le script officiel expose --max_iter mais ne le
transmet pas au modèle, qui garde le défaut de scikit-learn (100 itérations,
sans garantie de convergence sur ~70 000 époques x 1 536 dimensions). Ici,
max_iter vaut 1 000 et la convergence est vérifiée et affichée (n_iter_),
plutôt que de comparer un modèle arrêté avant d'avoir convergé.

Les autres sondes sont évaluées à côté, sur exactement les mêmes embeddings
et le même split, chacune pour tester un levier :
  - logistic_unweighted : même modèle sans pondération des classes (la
    baseline utilisée pour les premiers résultats, avant correction) ;
  - xgboost (le dépôt officiel le propose aussi) : frontière non linéaire
    par arbres ;
  - mlp : frontière non linéaire par réseau, plus adaptée aux embeddings
    denses que les arbres et plus rapide ; arrêt précoce sur des nuits mises
    à part (pas sur des époques tirées au hasard, qui viendraient des mêmes
    nuits que l'entraînement) ;
  - sequence : contexte temporel (époques voisines), voir sequence_probe.py.
Hyperparamètres fixes, raisonnables, jamais réglés sur le jeu de test.
(Random Forest, testé, a été retiré : plus lent que XGBoost et moins adapté
aux embeddings denses.)

Entre XGBoost et MLP, une seule sonde non linéaire est gardée, choisie par
validation croisée par sujet sur le train (voir choose_mlp_or_xgboost) : le
test n'intervient qu'à la fin, sur le modèle retenu.
"""

from __future__ import annotations

import copy
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

PROBE_KINDS = ("logistic", "logistic_unweighted", "xgboost", "mlp", "sequence")
LOGISTIC_MAX_ITER = 1000
MIN_XGBOOST_GAIN = 0.02  # F1 macro : en dessous, l'écart relève du bruit du tirage des sujets de test
MAX_F1_GAP = 0.15  # gap F1 apprentissage - validation au-delà duquel une sonde est jugée surapprise
CV_FOLDS = 5


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = LOGISTIC_MAX_ITER,
    class_weight: str | None = "balanced",
) -> LogisticRegression:
    """Régression logistique ; `class_weight="balanced"` = protocole officiel.
    Enregistre sur le modèle `converged_` et `n_iter_used_` : un modèle non
    convergé n'est pas une baseline fiable, l'appelant doit pouvoir le dire."""
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    model.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    model.n_iter_used_ = int(np.max(model.n_iter_))
    return model


class _NightEarlyStoppingMLP:
    """MLP dont l'arrêt précoce se fait sur ~10 % des NUITS du train mises à
    part : `MLPClassifier(early_stopping=True)` tirerait 10 % d'époques au
    hasard, issues des mêmes nuits que l'entraînement (fuite), et s'arrêterait
    trop tard. Garde l'état du meilleur passage (perte de validation)."""

    def __init__(self, seed: int = 42, max_epochs: int = 100, patience: int = 10) -> None:
        self.seed, self.max_epochs, self.patience = seed, max_epochs, patience

    def fit(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> _NightEarlyStoppingMLP:
        self.classes_ = np.unique(y)
        nights = np.array(sorted(set(subjects)))
        np.random.default_rng(self.seed).shuffle(nights)
        val_nights = set(nights[: max(1, len(nights) // 10)]) if len(nights) >= 5 else set()
        is_val = np.isin(subjects, list(val_nights))
        X_fit, y_fit = X[~is_val], y[~is_val]

        model = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, random_state=self.seed)
        best_loss, best_model, stale = float("inf"), None, 0
        for _ in range(self.max_epochs):
            model.partial_fit(X_fit, y_fit, classes=self.classes_)  # une passe sur le train
            if not is_val.any():
                continue
            val_loss = log_loss(y[is_val], model.predict_proba(X[is_val]), labels=self.classes_)
            if val_loss < best_loss - 1e-4:
                best_loss, best_model, stale = val_loss, copy.deepcopy(model), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_ = best_model or model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)


class _EncodedXGBClassifier:
    """XGBoost exige des labels 0..K-1 contigus : si un stade manque dans le
    train (petits runs), on ré-encode et on redonne les labels d'origine."""

    def __init__(self, **params) -> None:
        from xgboost import XGBClassifier  # import paresseux : dépendance optionnelle

        self.model = XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _EncodedXGBClassifier:
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def train_probe(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = LOGISTIC_MAX_ITER,
    seed: int = 42,
    device: str = "cpu",
    groups: tuple[np.ndarray, np.ndarray] | None = None,
):
    """`groups` = (sujet, numéro d'époque) de chaque ligne, requis par "mlp"
    (nuits de validation) et "sequence" (ordre des époques)."""
    if kind == "logistic":
        return train_linear_probe(X_train, y_train, max_iter=max_iter, class_weight="balanced")
    if kind == "logistic_unweighted":
        return train_linear_probe(X_train, y_train, max_iter=max_iter, class_weight=None)
    if kind == "mlp":
        if groups is None:
            raise ValueError("La sonde 'mlp' a besoin de groups=(sujets, ...) pour son arrêt précoce par nuit")
        return _NightEarlyStoppingMLP(seed=seed).fit(X_train, y_train, groups[0])
    if kind == "sequence":
        if groups is None:
            raise ValueError("La sonde 'sequence' a besoin de groups=(sujets, numéros d'époque)")
        from .sequence_probe import SequenceProbe  # import paresseux : torch

        return SequenceProbe(device=device, seed=seed).fit(X_train, y_train, groups)
    if kind == "xgboost":
        model = _EncodedXGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            device="cuda" if device.startswith("cuda") else "cpu",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Sonde inconnue {kind!r} (disponibles : {PROBE_KINDS})")
    model.fit(X_train, y_train)
    return model


def macro_roc_auc(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC un-contre-tous, moyenne sur les stades présents dans y (un
    stade absent n'a pas d'AUC définie)."""
    aucs = [roc_auc_score(y == c, proba[:, i]) for i, c in enumerate(classes) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_auc(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> dict[int, dict[str, float]]:
    """ROC-AUC et AUPRC un-contre-tous de chaque stade présent dans y.
    L'AUPRC (aire sous la courbe précision-rappel) dépend de la fréquence du
    stade : elle est plus sévère que la ROC-AUC pour les stades rares (N1)."""
    out = {}
    for i, c in enumerate(classes):
        positives = y == c
        if 0 < positives.sum() < len(y):
            out[int(c)] = {
                "roc_auc": float(roc_auc_score(positives, proba[:, i])),
                "auprc": float(average_precision_score(positives, proba[:, i])),
            }
    return out


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray, n_boot: int = 500, seed: int = 42
) -> dict:
    """Intervalles de confiance à 95 % du kappa et du F1 macro par bootstrap
    sur les SUJETS (on retire des nuits entières avec remise) : c'est la
    bonne unité d'échantillonnage, les époques d'une même nuit n'étant pas
    indépendantes. Avec ~20 sujets de test, l'intervalle est large."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(subjects)))
    rows_of = {s: np.flatnonzero(subjects == s) for s in unique}
    kappas, f1s = [], []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[s] for s in rng.choice(unique, len(unique), replace=True)])
        kappas.append(cohen_kappa_score(y_true[rows], y_pred[rows]))
        f1s.append(f1_score(y_true[rows], y_pred[rows], average="macro"))
    return {
        "cohen_kappa": [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))],
        "macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "n_boot": n_boot,
        "n_subjects": len(unique),
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    subjects: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Écart b - a du kappa et du F1 macro, avec son IC à 95 %, par bootstrap
    APPARIÉ sur les sujets : chaque tirage prend les mêmes nuits pour les deux
    sondes. Bien plus précis que de comparer deux IC séparés, qui se
    chevauchent presque toujours : la variabilité commune (nuits faciles ou
    difficiles) s'annule dans la différence. Un écart dont l'IC exclut 0 est
    significatif."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(subjects)))
    rows_of = {s: np.flatnonzero(subjects == s) for s in unique}
    metrics = {"cohen_kappa": cohen_kappa_score, "macro_f1": lambda t, p: f1_score(t, p, average="macro")}
    draws = {name: [] for name in metrics}
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[s] for s in rng.choice(unique, len(unique), replace=True)])
        for name, metric in metrics.items():
            draws[name].append(metric(y_true[rows], pred_b[rows]) - metric(y_true[rows], pred_a[rows]))
    result = {
        name: {
            "diff": float(metric(y_true, pred_b) - metric(y_true, pred_a)),
            "ci95": [float(np.percentile(draws[name], 2.5)), float(np.percentile(draws[name], 97.5))],
        }
        for name, metric in metrics.items()
    }
    return result | {"n_boot": n_boot, "n_subjects": len(unique)}


def predict_labels(model, X: np.ndarray, groups: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    return model.predict(X, groups) if getattr(model, "needs_groups", False) else model.predict(X)


def predict_probabilities(model, X: np.ndarray, groups: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    return model.predict_proba(X, groups) if getattr(model, "needs_groups", False) else model.predict_proba(X)


def evaluate_probe(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups: tuple[np.ndarray, np.ndarray] | None = None,
    with_ci: bool = False,
) -> dict:
    """`with_ci=True` (et `groups` fourni) ajoute les intervalles de confiance
    par bootstrap sur les sujets : à réserver à l'évaluation finale."""
    y_pred = predict_labels(model, X_test, groups)
    proba = predict_probabilities(model, X_test, groups)
    results = {
        "accuracy": float((y_pred == y_test).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_macro": macro_roc_auc(proba, model.classes_, y_test),
        "per_class_auc": per_class_auc(proba, model.classes_, y_test),
        "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        # lignes = stade réel, colonnes = stade prédit, dans l'ordre de model.classes_
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_).tolist(),
        "classes": [int(c) for c in model.classes_],
    }
    aucs = results["per_class_auc"].values()
    results["auprc_macro"] = float(np.mean([a["auprc"] for a in aucs])) if aucs else float("nan")
    if with_ci and groups is not None:
        results["ci95"] = bootstrap_ci(y_test, y_pred, groups[0])
    return results


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


def cross_validate_probes(
    kinds,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int = 5,
    max_iter: int = LOGISTIC_MAX_ITER,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Validation croisée par sujet sur le TRAIN uniquement, mêmes plis pour
    chaque sonde (comparaison appariée). Le jeu de test n'intervient pas :
    il reste réservé à l'évaluation finale du modèle choisi.
    Renvoie {sonde: {"val_f1": [par pli], "train_f1": [par pli], "fit_seconds": total}}."""
    folds = subject_folds(subjects, n_folds, seed)
    results = {kind: {"val_f1": [], "train_f1": [], "fit_seconds": 0.0} for kind in kinds}
    for fit_idx, val_idx in folds:
        for kind in kinds:
            start = time.time()
            fit_groups = (subjects[fit_idx], np.zeros(len(fit_idx), dtype=int))  # nuits, pour l'arrêt précoce du MLP
            model = train_probe(
                kind, X[fit_idx], y[fit_idx], max_iter=max_iter, seed=seed, device=device, groups=fit_groups
            )
            results[kind]["fit_seconds"] += time.time() - start
            results[kind]["train_f1"].append(evaluate_probe(model, X[fit_idx], y[fit_idx])["macro_f1"])
            results[kind]["val_f1"].append(evaluate_probe(model, X[val_idx], y[val_idx])["macro_f1"])
    return results


def choose_mlp_or_xgboost(
    cv: dict, min_gain: float = MIN_XGBOOST_GAIN, max_gap: float = MAX_F1_GAP
) -> tuple[str | None, str]:
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
    gaps = {
        kind: float(np.mean(np.array(cv[kind]["train_f1"]) - np.array(cv[kind]["val_f1"])))
        for kind in ("mlp", "xgboost")
    }
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
    return "mlp", (
        f"{detail} : écart trop faible ou irrégulier, MLP retenu, plus rapide "
        f"({cv['mlp']['fit_seconds']:.0f}s contre {cv['xgboost']['fit_seconds']:.0f}s sur les plis)"
    )


def compare_probes(
    kinds,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = LOGISTIC_MAX_ITER,
    seed: int = 42,
    device: str = "cpu",
    train_groups: tuple[np.ndarray, np.ndarray] | None = None,
    test_groups: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Entraîne chaque sonde sur le même train, l'évalue sur le train ET le
    test : l'écart (gap) train - test mesure le surapprentissage.
    `*_groups` = (sujet, numéro d'époque) par ligne, requis par "sequence".
    Si la baseline "logistic" est évaluée, chaque autre sonde reçoit
    "vs_baseline" : son écart à la baseline sur les mêmes nuits de test
    (paired_bootstrap_diff)."""
    results, test_preds = {}, {}
    for kind in kinds:
        start = time.time()
        model = train_probe(kind, X_train, y_train, max_iter=max_iter, seed=seed, device=device, groups=train_groups)
        results[kind] = {
            "train": evaluate_probe(model, X_train, y_train, train_groups),
            "test": evaluate_probe(model, X_test, y_test, test_groups, with_ci=True),
            "fit_seconds": time.time() - start,
            # None pour les sondes sans notion de convergence (XGBoost, MLP, BiGRU)
            "converged": getattr(model, "converged_", None),
            "n_iter": getattr(model, "n_iter_used_", None),
        }
        test_preds[kind] = predict_labels(model, X_test, test_groups)
        proba = predict_probabilities(model, X_test, test_groups)
        # Sorties brutes du test, pour recalculer IC et comparaisons appariées sans réentraîner.
        results[kind]["test_outputs"] = {"pred": test_preds[kind], "proba": proba, "classes": model.classes_}
    if "logistic" in results and test_groups is not None:
        for kind in results:
            if kind != "logistic":
                results[kind]["vs_baseline"] = paired_bootstrap_diff(
                    y_test, test_preds["logistic"], test_preds[kind], test_groups[0], seed=seed
                )
    return results


def compare_to_published(results: dict, published_kappa: float, tolerance: float = 0.05) -> bool:
    delta = abs(results["cohen_kappa"] - published_kappa)
    reproduced = delta <= tolerance
    status = "REPRODUIT" if reproduced else "ÉCART SIGNIFICATIF"
    print(f"κ obtenu = {results['cohen_kappa']:.3f} | κ publié = {published_kappa:.3f} | {status}")
    return reproduced
