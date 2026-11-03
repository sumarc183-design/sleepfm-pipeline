"""Vérifie la sonde logistique (entraînement, évaluation, comparaison)."""
import numpy as np

import pytest

from src.models.linear_probe import (
    PROBE_KINDS, compare_probes, compare_to_published, evaluate_probe, train_linear_probe, train_probe,
)


def test_train_and_evaluate_probe():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50, 32))
    y_train = rng.integers(0, 5, 50)
    X_test = rng.standard_normal((20, 32))
    y_test = rng.integers(0, 5, 20)

    probe = train_linear_probe(X_train, y_train, max_iter=50)
    results = evaluate_probe(probe, X_test, y_test)

    assert set(results.keys()) == {"accuracy", "cohen_kappa", "macro_f1", "roc_auc_macro", "per_class_report"}
    assert 0.0 <= results["roc_auc_macro"] <= 1.0
    assert 0.0 <= results["accuracy"] <= 1.0
    assert -1.0 <= results["cohen_kappa"] <= 1.0


def test_compare_to_published_reproduced_flag():
    results = {"cohen_kappa": 0.62}
    assert compare_to_published(results, published_kappa=0.60, tolerance=0.05) is True
    assert compare_to_published(results, published_kappa=0.20, tolerance=0.05) is False


def _separable(n, rng):
    y = rng.integers(0, 5, n)
    X = rng.standard_normal((n, 16)) * 0.5
    X[np.arange(n), y] += 3.0  # un stade = une direction : apprenable par les 3 sondes
    return X, y


def test_compare_probes_reports_train_test_and_gap():
    rng = np.random.default_rng(0)
    X_train, y_train = _separable(400, rng)
    X_test, y_test = _separable(150, rng)

    results = compare_probes(PROBE_KINDS, X_train, y_train, X_test, y_test, max_iter=200)

    assert set(results) == set(PROBE_KINDS)
    for kind, r in results.items():
        assert r["test"]["macro_f1"] > 0.8, kind
        assert r["test"]["roc_auc_macro"] > 0.9, kind
        assert r["train"]["macro_f1"] >= r["test"]["macro_f1"] - 0.1, kind
        assert r["fit_seconds"] >= 0


@pytest.mark.parametrize("kind", ["random_forest", "xgboost"])
def test_tree_probes_handle_a_stage_missing_from_train(kind):
    """Petit run : un stade (ici 2) peut manquer au train ; XGBoost exige des
    labels contigus, la sonde doit quand même fonctionner."""
    rng = np.random.default_rng(1)
    X_train, y_train = _separable(300, rng)
    keep = y_train != 2
    X_test, y_test = _separable(100, rng)

    model = train_probe(kind, X_train[keep], y_train[keep])
    assert set(np.unique(model.predict(X_test))) <= {0, 1, 3, 4}  # labels d'origine, jamais 2
    assert model.predict_proba(X_test).shape == (len(X_test), 4)
    assert 0.0 <= evaluate_probe(model, X_test, y_test)["roc_auc_macro"] <= 1.0
