"""Vérifie la sonde logistique (entraînement, évaluation, comparaison)."""
import numpy as np

from src.models.linear_probe import compare_to_published, evaluate_probe, train_linear_probe


def test_train_and_evaluate_probe():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50, 32))
    y_train = rng.integers(0, 5, 50)
    X_test = rng.standard_normal((20, 32))
    y_test = rng.integers(0, 5, 20)

    probe = train_linear_probe(X_train, y_train, max_iter=50)
    results = evaluate_probe(probe, X_test, y_test)

    assert set(results.keys()) == {"accuracy", "cohen_kappa", "macro_f1", "per_class_report"}
    assert 0.0 <= results["accuracy"] <= 1.0
    assert -1.0 <= results["cohen_kappa"] <= 1.0


def test_compare_to_published_reproduced_flag():
    results = {"cohen_kappa": 0.62}
    assert compare_to_published(results, published_kappa=0.60, tolerance=0.05) is True
    assert compare_to_published(results, published_kappa=0.20, tolerance=0.05) is False
