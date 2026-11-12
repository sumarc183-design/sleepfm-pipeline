"""Seuils sur validation, ex æquo et incertitude par nuit."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_respiratory_thresholds as evaluation
from src.data.cache_contract import file_sha256


@pytest.mark.parametrize("target", [0.7, 0.8, 0.9, 1.0])
def test_threshold_is_largest_value_reaching_validation_sensitivity_with_ties(target):
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.8, 0.2, 0.99, 0.1])
    t = evaluation.sensitivity_threshold(y, p, target)
    assert (p[y == 1] >= t).mean() >= target
    assert all((p[y == 1] >= v).mean() < target for v in np.unique(p[p > t]))
    assert t == (0.8 if target == 0.7 else 0.2)


@pytest.mark.parametrize("target", [0.0, -1.0, 1.1, np.nan])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError, match="Cible"):
        evaluation.sensitivity_threshold(np.array([0, 1]), np.array([0.1, 0.9]), target)


@pytest.mark.parametrize("y,p", [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1, 1.1]), ([0, 1], [0.1])])
def test_invalid_labels_or_probabilities_are_rejected(y, p):
    with pytest.raises(ValueError, match="classes"):
        evaluation.sensitivity_threshold(y, p, 0.8)


def test_metrics_distinguish_windows_from_events_and_handle_empty_hypopnea():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.4, 0.9])
    hypo = np.array([False, False, True, False])
    r = evaluation.operating_metrics(y, p, 0.5, hypo)
    assert r["confusion_matrix"] == [[1, 1], [1, 1]]
    assert r["false_positive_windows"] == 1 and r["predicted_positive"] == 2
    assert r["sensitivity"] == r["specificity"] == r["precision"] == r["f1"] == 0.5
    assert r["hypopnea"] == {"windows": 1, "true_positive": 0, "recall": 0.0}
    empty = evaluation.operating_metrics(y, p, 1.0, np.zeros(4, dtype=bool))
    assert empty["precision"] is None and empty["hypopnea"]["recall"] is None
    with pytest.raises(ValueError, match="hypopnée"):
        evaluation.operating_metrics(y, p, 0.5, np.array([True, False, False, False]))


def test_subject_bootstrap_matches_explicit_whole_night_resampling():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.7, 0.4, 0.4, 0.8])
    s = np.repeat(["a", "b", "c"], 2)
    hypo = np.array([False, True, False, True, False, False])
    actual = evaluation.operating_intervals(y, p, 0.5, s, hypo, n_boot=30)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(30):
        rows = np.concatenate([np.flatnonzero(s == sid) for sid in rng.choice(np.unique(s), 3)])
        m = evaluation.operating_metrics(y[rows], p[rows], 0.5, hypo[rows])
        expected.append([m["sensitivity"], m["specificity"], m["precision"], m["hypopnea"]["recall"]])
    values = np.array([[np.nan if v is None else v for v in row] for row in expected])
    for i, name in enumerate(["sensitivity", "specificity", "precision", "hypopnea_recall"]):
        finite = np.isfinite(values[:, i])
        assert actual["metrics"][name]["valid_draws"] == finite.sum()
        np.testing.assert_allclose(actual["metrics"][name]["ci95"], np.percentile(values[finite, i], [2.5, 97.5]))
    empty = evaluation.operating_intervals(y, p, 0.5, s, np.zeros(6, dtype=bool), n_boot=10)
    assert empty["metrics"]["hypopnea_recall"] == {"ci95": None, "valid_draws": 0}


def test_artifact_hash_and_location_are_checked(tmp_path):
    artifact = tmp_path / "probabilities.npz"
    artifact.write_bytes(b"original")
    report = {"artifacts": {"test": {"file": artifact.name, "sha256": file_sha256(artifact)}}}
    source = tmp_path / "source.json"
    assert evaluation.checked_artifact(source, report, "test") == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Artefact modifié"):
        evaluation.checked_artifact(source, report, "test")
    report["artifacts"]["test"]["file"] = "../other.npz"
    with pytest.raises(ValueError, match="hors dossier"):
        evaluation.checked_artifact(source, report, "test")


def test_outputs_cannot_overwrite_sources_or_enter_caches(tmp_path):
    out = tmp_path / "result"
    out.with_suffix(".json").write_text("keep")
    with pytest.raises(ValueError, match="Sortie existante"):
        evaluation.run(SimpleNamespace(out=out))
    assert out.with_suffix(".json").read_text() == "keep"
    args = SimpleNamespace(
        out=tmp_path / "cache/result",
        sleepfm_dir=tmp_path / "cache",
        osf_dir=tmp_path / "osf",
        labels_dir=tmp_path / "labels",
    )
    with pytest.raises(ValueError, match="entrée"):
        evaluation.run(args)
    assert not (tmp_path / "cache").exists()


def test_pipeline_freezes_all_six_thresholds_before_new_test_metrics(tmp_path, monkeypatch):
    val_y = np.array([0, 1, 0, 1, 0, 1])
    test_y = 1 - val_y

    def rows(y):
        return {
            "y_true": y,
            "subject": np.repeat(["a", "b"], 3),
            "epoch_idx": np.tile(np.arange(3), 2),
            "onset_s": np.tile(np.arange(3) * 30.0, 2),
            "probability_sleepfm": np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.7]),
            "probability_osf": np.array([0.2, 0.8, 0.3, 0.9, 0.4, 0.75]),
        }

    validation, test = rows(val_y), rows(test_y)
    comparison = {"thresholds": {"sleepfm": 0.5, "osf": 0.6}, "artifacts": {}}
    baseline = {"partitions": {"fit": ["fit"], "validation": ["a", "b"], "test": ["c", "d"]}}
    groups = {"event_group": np.array(["resp_hypopnea" if x else "aucun" for x in test_y])}
    monkeypatch.setattr(evaluation, "load_sources", lambda args: (baseline, {}, comparison, test, groups, {}))
    monkeypatch.setattr(evaluation, "validation_probabilities", lambda *args: (validation, {}, {}))
    calls = []
    original_threshold, original_metrics = evaluation.sensitivity_threshold, evaluation.operating_metrics

    def threshold(y, p, target):
        assert np.array_equal(y, val_y)
        calls.append(target)
        return original_threshold(y, p, target)

    def metrics(*args, **kwargs):
        assert len(calls) == 6
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(evaluation, "sensitivity_threshold", threshold)
    monkeypatch.setattr(evaluation, "operating_metrics", metrics)
    args = SimpleNamespace(
        out=tmp_path / "result",
        sleepfm_dir=tmp_path / "sleep",
        osf_dir=tmp_path / "osf",
        labels_dir=tmp_path / "labels",
    )
    for name in ["source", "baseline", "comparison", "errors", "plan"]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"original": name}))
        setattr(args, name, path)
    snapshots = {name: getattr(args, name).read_bytes() for name in ["baseline", "comparison", "errors"]}
    result = evaluation.run(args)
    assert calls == [0.7, 0.8, 0.9] * 2
    assert result["results"]["sleepfm"]["initial_f1"]["threshold"] == 0.5
    assert result["results"]["osf"]["initial_f1"]["threshold"] == 0.6
    for kind in evaluation.MODELS:
        for target in evaluation.TARGETS:
            entry = result["results"][kind][f"sensitivity_{round(target * 100)}"]
            assert entry["validation"]["sensitivity"] >= target
            assert entry["test"]["sensitivity"] < target
    assert all(getattr(args, name).read_bytes() == data for name, data in snapshots.items())
