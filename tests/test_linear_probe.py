"""Vérifie la sonde logistique (entraînement, évaluation, comparaison)."""

import numpy as np
import pytest

from src.models.linear_probe import (
    PROBE_KINDS,
    choose_mlp_or_xgboost,
    compare_probes,
    compare_to_published,
    cross_validate_probes,
    evaluate_probe,
    per_class_auc,
    subject_folds,
    train_linear_probe,
    train_probe,
)


def test_train_and_evaluate_probe():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50, 32))
    y_train = rng.integers(0, 5, 50)
    X_test = rng.standard_normal((20, 32))
    y_test = rng.integers(0, 5, 20)

    probe = train_linear_probe(X_train, y_train, max_iter=50)
    results = evaluate_probe(probe, X_test, y_test)

    assert set(results.keys()) == {
        "accuracy",
        "cohen_kappa",
        "macro_f1",
        "roc_auc_macro",
        "auprc_macro",
        "per_class_auc",
        "per_class_report",
        "confusion_matrix",
        "classes",
    }
    assert np.array(results["confusion_matrix"]).sum() == len(y_test)
    assert 0.0 <= results["roc_auc_macro"] <= 1.0
    assert 0.0 <= results["auprc_macro"] <= 1.0
    assert 0.0 <= results["accuracy"] <= 1.0
    assert -1.0 <= results["cohen_kappa"] <= 1.0


def test_compare_to_published_reproduced_flag():
    results = {"cohen_kappa": 0.62}
    assert compare_to_published(results, published_kappa=0.60, tolerance=0.05) is True
    assert compare_to_published(results, published_kappa=0.20, tolerance=0.05) is False


def _groups(n, n_subjects=10):
    """(sujet, numéro d'époque) pour n lignes réparties en n_subjects nuits."""
    subjects = np.array([f"s{i % n_subjects}" for i in range(n)])
    return subjects, np.arange(n)


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
    results = compare_probes(
        kinds, X_train, y_train, X_test, y_test, train_groups=_groups(400), test_groups=_groups(150)
    )

    assert set(results) == set(kinds)
    assert results["logistic"]["converged"] is True and results["logistic"]["n_iter"] > 0
    ci = results["logistic"]["test"]["ci95"]
    assert ci["cohen_kappa"][0] <= results["logistic"]["test"]["cohen_kappa"] <= ci["cohen_kappa"][1]
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

    model = train_probe(kind, X_train[keep], y_train[keep], groups=_groups(int(keep.sum())))
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
            X.append(emb[order])
            y.append(stages[order])
            subjects.append(np.full(length, f"s{night}"))
            idx.append(np.arange(length)[order])
        return np.concatenate(X), np.concatenate(y), (np.concatenate(subjects), np.concatenate(idx))

    X_train, y_train, g_train = nights(20)
    X_test, y_test, g_test = nights(5)
    results = compare_probes(
        ["logistic", "sequence"], X_train, y_train, X_test, y_test, train_groups=g_train, test_groups=g_test
    )
    assert results["sequence"]["test"]["macro_f1"] > results["logistic"]["test"]["macro_f1"] + 0.3


def test_sequence_probe_requires_groups():
    X, y = _separable(50, np.random.default_rng(0))
    with pytest.raises(ValueError, match="groups"):
        train_probe("sequence", X, y)


@pytest.mark.parametrize(
    "mlp, xgb, expected",
    [
        # (F1 apprentissage, F1 validation par pli) de chaque sonde, 5 plis
        ((0.78, [0.75] * 5), (0.75, [0.70] * 5), "mlp"),  # XGBoost moins bon
        ((0.65, [0.60] * 5), (0.66, [0.615] * 5), "mlp"),  # +0,015 : quasi-égalité, MLP (plus rapide)
        ((0.65, [0.60] * 5), (0.70, [0.63] * 5), "xgboost"),  # +0,03 sur les 5 plis : net et régulier
        ((0.65, [0.60] * 5), (0.70, [0.75, 0.75, 0.58, 0.58, 0.58]), "mlp"),  # +0,036 en moyenne mais 2/5 plis : tirage
        ((0.65, [0.60] * 5), (1.00, [0.70] * 5), "mlp"),  # XGBoost meilleur mais gap 0,30 : éliminé
        ((0.99, [0.70] * 5), (0.66, [0.62] * 5), "xgboost"),  # MLP meilleur mais gap 0,29 : éliminé
        ((1.00, [0.70] * 5), (1.00, [0.65] * 5), None),  # les deux surapprennent
    ],
)
def test_choose_mlp_or_xgboost(mlp, xgb, expected):
    def cv(train_f1, val_f1, seconds):
        return {"train_f1": [train_f1] * len(val_f1), "val_f1": val_f1, "fit_seconds": seconds}

    results = {"mlp": cv(*mlp, 60), "xgboost": cv(*xgb, 120)}
    assert choose_mlp_or_xgboost(results, min_gain=0.02, max_gap=0.15)[0] == expected


def test_subject_folds_never_split_a_subject():
    subjects = np.repeat([f"s{i}" for i in range(12)], 7)
    folds = subject_folds(subjects, n_folds=5)
    assert len(folds) == 5
    covered = np.concatenate([val for _, val in folds])
    assert sorted(covered) == list(range(len(subjects)))  # chaque ligne validée une fois
    for fit_idx, val_idx in folds:
        assert set(subjects[fit_idx]).isdisjoint(subjects[val_idx])


def test_cross_validate_probes_uses_only_given_data():
    rng = np.random.default_rng(0)
    X, y = _separable(300, rng)
    subjects = np.repeat([f"s{i}" for i in range(10)], 30)
    cv = cross_validate_probes(["mlp", "xgboost"], X, y, subjects, n_folds=5)
    for kind in ("mlp", "xgboost"):
        assert len(cv[kind]["val_f1"]) == len(cv[kind]["train_f1"]) == 5
        assert np.mean(cv[kind]["val_f1"]) > 0.8


def test_baseline_is_the_official_balanced_logistic():
    """Protocole officiel (sleepfm/utils.py::train_model) :
    LogisticRegression(class_weight="balanced")."""
    X, y = _separable(200, np.random.default_rng(0))
    assert train_probe("logistic", X, y).class_weight == "balanced"
    assert train_probe("logistic_unweighted", X, y).class_weight is None


def test_non_convergence_is_reported_not_hidden():
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((300, 50)) * 50, rng.integers(0, 5, 300)  # mal conditionné, 1 itération
    model = train_linear_probe(X, y, max_iter=1)
    assert model.converged_ is False and model.n_iter_used_ == 1


def test_mlp_early_stopping_holds_out_whole_nights(monkeypatch):
    """Les lignes de validation de l'arrêt précoce doivent être des nuits
    entières absentes de l'apprentissage."""
    from sklearn.neural_network import MLPClassifier

    seen_fit_rows = []
    original = MLPClassifier.partial_fit

    def spy(self, X, y, classes=None):
        seen_fit_rows.append(len(X))
        return original(self, X, y, classes=classes)

    monkeypatch.setattr(MLPClassifier, "partial_fit", spy)
    X, y = _separable(300, np.random.default_rng(0))
    subjects, _ = _groups(300, n_subjects=10)  # 10 nuits de 30 époques -> 1 nuit de validation
    train_probe("mlp", X, y, groups=(subjects, np.arange(300)))
    assert set(seen_fit_rows) == {270}


def test_bootstrap_ci_is_ordered_and_exact_on_perfect_predictions():
    from src.models.linear_probe import bootstrap_ci

    y = np.random.default_rng(0).integers(0, 5, 200)
    subjects = np.array([f"s{i % 8}" for i in range(200)])
    ci = bootstrap_ci(y, y, subjects, n_boot=50)
    assert ci["cohen_kappa"] == [1.0, 1.0] and ci["n_subjects"] == 8
    noisy = np.where(np.random.default_rng(1).random(200) < 0.3, (y + 1) % 5, y)
    low, high = bootstrap_ci(y, noisy, subjects, n_boot=200)["macro_f1"]
    assert low <= high


def test_paired_bootstrap_detects_a_real_gain_and_not_a_null_one():
    from src.models.linear_probe import paired_bootstrap_diff

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    subjects = np.repeat([f"s{i}" for i in range(20)], 100)
    worse = np.where(rng.random(2000) < 0.4, (y + 1) % 5, y)
    better = np.where(rng.random(2000) < 0.2, (y + 1) % 5, y)

    gain = paired_bootstrap_diff(y, worse, better, subjects, n_boot=200)
    assert gain["cohen_kappa"]["diff"] > 0 and gain["cohen_kappa"]["ci95"][0] > 0
    assert gain["n_subjects"] == 20

    same = paired_bootstrap_diff(y, worse, worse, subjects, n_boot=50)
    assert same["cohen_kappa"]["diff"] == 0 and same["cohen_kappa"]["ci95"] == [0.0, 0.0]


def test_compare_probes_reports_paired_gap_to_baseline():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    X = rng.standard_normal((400, 8))
    X[np.arange(400), y] += 2.0
    groups = (np.repeat([f"s{i}" for i in range(8)], 50), np.tile(np.arange(50), 8))
    train, test = slice(0, 300), slice(300, 400)
    results = compare_probes(
        ["logistic", "logistic_unweighted"],
        X[train],
        y[train],
        X[test],
        y[test],
        train_groups=(groups[0][train], groups[1][train]),
        test_groups=(groups[0][test], groups[1][test]),
    )
    assert "vs_baseline" not in results["logistic"]
    paired = results["logistic_unweighted"]["vs_baseline"]
    low, high = paired["cohen_kappa"]["ci95"]
    assert low <= paired["cohen_kappa"]["diff"] <= high


def test_per_class_auc_perfect_random_and_rare_stage():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    perfect = np.stack([1.0 - (y == 1), (y == 1).astype(float)], axis=1)
    scores = per_class_auc(perfect, np.array([0, 1]), y)
    assert scores[1] == {"roc_auc": 1.0, "auprc": 1.0}
    # Probabilités constantes : ROC-AUC 0,5, AUPRC = fréquence du stade (plus sévère pour un stade rare).
    flat = per_class_auc(np.full((10, 2), 0.5), np.array([0, 1]), y)
    assert flat[1]["roc_auc"] == 0.5
    assert flat[1]["auprc"] == pytest.approx(0.3)
    # Un stade absent du test n'a pas de score.
    assert 2 not in per_class_auc(np.full((10, 3), 1 / 3), np.array([0, 1, 2]), y)


def test_bigru_padding_does_not_leak_into_real_epochs():
    import torch

    from src.models.sequence_probe import _BiGRUTagger

    torch.manual_seed(0)
    model = _BiGRUTagger(embedding_dim=8, hidden_dim=6, num_classes=5, dropout=0.0).eval()
    short, long_ = torch.randn(3, 8), torch.randn(7, 8)
    batch = torch.zeros(2, 7, 8)
    batch[0, :3], batch[1] = short, long_
    with torch.no_grad():
        alone = model(short[None])[0]
        packed = model(batch, torch.tensor([3, 7]))[0, :3]
        unpacked = model(batch)[0, :3]
    # Avec les longueurs, la séquence courte donne la même sortie que seule ;
    # sans elles, le sens retour lisait le rembourrage.
    assert torch.allclose(alone, packed, atol=1e-6)
    assert not torch.allclose(alone, unpacked, atol=1e-4)
