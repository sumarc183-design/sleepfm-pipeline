"""Vérifie la sonde logistique (entraînement, évaluation, comparaison)."""
import numpy as np

import pytest

from src.models.linear_probe import (
    PROBE_KINDS, choose_mlp_or_xgboost, compare_probes, compare_to_published, evaluate_probe, train_linear_probe, train_probe,
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

    kinds = [k for k in PROBE_KINDS if k != "sequence"]  # "sequence" : tests dédiés (besoin des nuits)
    results = compare_probes(kinds, X_train, y_train, X_test, y_test, max_iter=200)

    assert set(results) == set(kinds)
    for kind, r in results.items():
        assert r["test"]["macro_f1"] > 0.8, kind
        assert r["test"]["roc_auc_macro"] > 0.9, kind
        assert r["train"]["macro_f1"] >= r["test"]["macro_f1"] - 0.1, kind
        assert r["fit_seconds"] >= 0


@pytest.mark.parametrize("kind", ["xgboost", "mlp"])
def test_probes_handle_a_stage_missing_from_train(kind):
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


def test_sequence_probe_uses_temporal_context():
    """Stade impossible à deviner sur l'époque seule : l'indice du stade de
    l'époque t est porté par l'embedding de l'époque t+1. Seule la sonde
    séquentielle, qui voit les voisines, peut le retrouver."""
    rng = np.random.default_rng(0)

    def nights(n_nights, length=60):
        X, y, subjects, idx = [], [], [], []
        for night in range(n_nights):
            stages = rng.integers(0, 3, length)
            emb = rng.standard_normal((length, 8)) * 0.3
            emb[1:, :3] += np.eye(3)[stages[:-1]] * 3  # stade t encodé dans l'époque t+1
            order = rng.permutation(length)  # lignes mélangées : l'ordre vient de epoch_idx
            X.append(emb[order]); y.append(stages[order])
            subjects.append(np.full(length, f"s{night}")); idx.append(np.arange(length)[order])
        return np.concatenate(X), np.concatenate(y), (np.concatenate(subjects), np.concatenate(idx))

    X_train, y_train, g_train = nights(20)
    X_test, y_test, g_test = nights(5)
    results = compare_probes(["logistic", "sequence"], X_train, y_train, X_test, y_test,
                             train_groups=g_train, test_groups=g_test)
    assert results["sequence"]["test"]["macro_f1"] > results["logistic"]["test"]["macro_f1"] + 0.3


def test_sequence_probe_requires_groups():
    X, y = _separable(50, np.random.default_rng(0))
    with pytest.raises(ValueError, match="groups"):
        train_probe("sequence", X, y)


@pytest.mark.parametrize("mlp, xgb, expected", [
    # (F1 train, F1 test) de chaque sonde
    ((0.78, 0.75), (0.75, 0.70), "mlp"),      # XGBoost moins bon
    ((0.65, 0.60), (0.66, 0.615), "mlp"),     # XGBoost devant de 0,015 : quasi-égalité, le MLP (plus rapide) gagne
    ((0.65, 0.60), (0.70, 0.63), "xgboost"),  # XGBoost devant de 0,03, gap raisonnable : écart net
    ((0.65, 0.60), (1.00, 0.70), "mlp"),      # XGBoost bien meilleur au test mais gap 0,30 : éliminé
    ((0.99, 0.70), (0.66, 0.62), "xgboost"),  # MLP meilleur au test mais gap 0,29 : éliminé
    ((1.00, 0.70), (1.00, 0.65), None),       # les deux surapprennent : aucune retenue
])
def test_choose_mlp_or_xgboost(mlp, xgb, expected):
    def run(train_f1, test_f1, seconds):
        return {"train": {"macro_f1": train_f1}, "test": {"macro_f1": test_f1}, "fit_seconds": seconds}

    results = {"mlp": run(*mlp, 60), "xgboost": run(*xgb, 120)}
    assert choose_mlp_or_xgboost(results, min_gain=0.02, max_gap=0.15)[0] == expected
