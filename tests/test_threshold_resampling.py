"""Pondération par nuit, seuil choisi sans la nuit évaluée et référence immuable."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from scripts import analyse_threshold_stability as study
from scripts.evaluate_respiratory_thresholds import sensitivity_threshold
from scripts.respiratory_baseline import select_threshold
from src.data.cache_contract import file_sha256
from src.models.threshold_resampling import POINTS, TARGETS, binary_counts, distribution, weighted_thresholds


def explicit_thresholds(y, p):
    return {"initial_f1": select_threshold(y, p), **{k: sensitivity_threshold(y, p, t) for k, t in TARGETS.items()}}


def test_weighted_selection_matches_explicit_duplication_with_ties_and_zero_weights():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.8, 0.7, 0.4, 0.4])
    w = np.array([0, 2, 1, 3, 2, 1])
    rows = np.repeat(np.arange(len(y)), w)
    assert weighted_thresholds(y, p, w) == explicit_thresholds(y[rows], p[rows])
    for t in weighted_thresholds(y, p, w).values():
        np.testing.assert_array_equal(binary_counts(y, p, t, w), confusion_matrix(y[rows], p[rows] >= t, labels=[0, 1]))


@pytest.mark.parametrize("bad", ["no_positive", "negative_weight", "nan_score", "bad_shape"])
def test_invalid_calibration_is_refused(bad):
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    w = np.array([1, 1])
    if bad == "no_positive":
        w[1] = 0
    elif bad == "negative_weight":
        w[0] = -1
    elif bad == "nan_score":
        p[0] = np.nan
    else:
        w = np.ones(3)
    with pytest.raises(ValueError):
        weighted_thresholds(y, p, w)


def fixture():
    y = np.tile([0, 1, 0, 1], 3)
    p = np.array([0.1, 0.8, 0.4, 0.5, 0.2, 0.9, 0.7, 0.8, 0.6, 0.7, 0.1, 0.4])
    s = np.repeat(["tr03-0001", "tr03-0002", "tr03-0003"], 4)
    return y, {"sleepfm": p, "osf": p * 0.8 + 0.1}, s, sorted(set(s))


def test_bootstrap_uses_whole_nights_and_scores_only_absent_nights():
    y, p, s, subjects = fixture()
    summary, saved, details = study.bootstrap_diagnostic(y, p, s, subjects, n_boot=30)
    assert details["valid_draws"] > 0 and "no_oob" in details["skipped"]
    for b, counts in enumerate(saved["subject_multiplicities"]):
        assert counts.sum() == len(subjects)
        if saved["skip_reason"][b]:
            assert np.isnan(saved["threshold_osf"][b]).all()
            continue
        calibr = np.concatenate(
            [np.tile(np.flatnonzero(s == sid), int(n)) for sid, n in zip(subjects, counts, strict=True)]
        )
        heldout = np.isin(s, np.array(subjects)[counts == 0])
        assert not (set(s[calibr]) & set(s[heldout]))
        for model in study.MODELS:
            expected = explicit_thresholds(y[calibr], p[model][calibr])
            for j, point in enumerate(POINTS):
                threshold = saved[f"threshold_{model}"][b, j]
                assert threshold == expected[point]
                np.testing.assert_array_equal(
                    saved[f"oob_cm_{model}"][b, j],
                    confusion_matrix(y[heldout], p[model][heldout] >= threshold, labels=[0, 1]),
                )
    assert summary["osf"]["sensitivity_80"]["threshold"]["count"] == details["valid_draws"]


def test_leave_one_night_out_threshold_cannot_use_its_heldout_labels():
    y, p, s, subjects = fixture()
    results, saved = study.leave_one_night_out(y, p, s, subjects)
    for i, sid in enumerate(subjects):
        calibration = s != sid
        heldout = ~calibration
        for model in study.MODELS:
            expected = explicit_thresholds(y[calibration], p[model][calibration])
            for j, point in enumerate(POINTS):
                assert saved[f"threshold_{model}"][i, j] == expected[point]
                np.testing.assert_array_equal(
                    saved[f"prediction_{model}"][heldout, j], p[model][heldout] >= expected[point]
                )
    changed = y.copy()
    changed[s == subjects[0]] = 1 - changed[s == subjects[0]]
    _, again = study.leave_one_night_out(changed, p, s, subjects)
    for model in study.MODELS:
        np.testing.assert_array_equal(saved[f"threshold_{model}"][0], again[f"threshold_{model}"][0])
    assert (
        results["osf"]["initial_f1"]["metrics"]["confusion_matrix"]
        == confusion_matrix(y, saved["prediction_osf"][:, 0], labels=[0, 1]).tolist()
    )


def test_distribution_handles_missing_values_and_empty_predictions():
    assert distribution([1, np.nan, 3]) == {"count": 2, "median": 2.0, "percentile_2_5": 1.05, "percentile_97_5": 2.95}
    assert distribution([])["median"] is None
    with pytest.raises(ValueError):
        binary_counts(np.array([0, 1]), np.array([0.1, 0.9]), 0.5, np.array([0.5, 1.0]))


def source_fixture(tmp_path):
    y, p, s, subjects = fixture()
    arrays = {
        "y_true": y,
        "subject": s,
        "epoch_idx": np.tile(np.arange(4), 3),
        "onset_s": np.tile(np.arange(4) * 30.0, 3),
        **{f"probability_{m}": v for m, v in p.items()},
    }
    archive = tmp_path / "validation.npz"
    np.savez(archive, **arrays)
    source = tmp_path / "source.json"
    r = {
        "status": "terminé",
        "partitions": {"fit": ["fit"], "validation": subjects, "test": ["never_opened_test"]},
        "artifacts": {"validation.npz": {"file": archive.name, "sha256": file_sha256(archive)}},
        "results": {},
    }
    for model in study.MODELS:
        r["results"][model] = {}
        for point, t in explicit_thresholds(y, p[model]).items():
            r["results"][model][point] = {
                "threshold": t,
                "validation": {"confusion_matrix": confusion_matrix(y, p[model] >= t, labels=[0, 1]).tolist()},
            }
    source.write_text(json.dumps(r))
    return source, archive, r


def test_validation_source_checks_original_thresholds_and_artifact_integrity(tmp_path):
    source, archive, r = source_fixture(tmp_path)
    _, _, subjects, _ = study.validation_inputs(source)
    assert len(subjects) == 3
    r["results"]["osf"]["initial_f1"]["threshold"] += 0.01
    source.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="Seuil complet"):
        study.validation_inputs(source)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive de validation modifiée"):
        study.validation_inputs(source)


def test_driver_preserves_sources_and_archives_only_validation_predictions(tmp_path, monkeypatch):
    source, archive, _ = source_fixture(tmp_path)
    before = source.read_bytes(), archive.read_bytes()
    original = study.bootstrap_diagnostic
    monkeypatch.setattr(study, "bootstrap_diagnostic", lambda *a: original(*a, n_boot=10))
    plan = tmp_path / "plan.md"
    plan.write_text("fixed plan")
    args = SimpleNamespace(source=source, plan=plan, out=tmp_path / "result")
    result = study.run(args)
    assert result["windows"] == 12 and result["bootstrap"]["n_boot"] == 10
    with np.load(tmp_path / "result.leave_one_out.npz") as f:
        assert set(f["subject"]) == set(result["subjects"]) and f["prediction_osf"].shape == (12, 4)
    assert before == (source.read_bytes(), archive.read_bytes())
    with pytest.raises(ValueError, match="Sortie existante"):
        study.run(args)
    assert not (tmp_path / "never_opened_test.npz").exists()


def test_bootstrap_counts_one_class_exclusions_without_replacing_draws():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.repeat(["a", "b", "c"], 2)
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.8])
    models = {"sleepfm": p, "osf": p}
    _, saved, details = study.bootstrap_diagnostic(y, models, s, ["a", "b", "c"], n_boot=100)
    assert details["skipped"]["calibration_one_class"] > 0
    assert details["skipped"]["oob_one_class"] > 0
    assert details["valid_draws"] + sum(details["skipped"].values()) == 100
    assert saved["subject_multiplicities"].shape == (100, 3)
