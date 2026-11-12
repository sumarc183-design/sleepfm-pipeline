"""Sous-groupes exclusifs, IC par sujet et analyse des seuls résultats sauvegardés."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyse_respiratory_errors as analysis
from src.data.cache_contract import file_sha256


def test_positive_groups_are_exclusive_even_with_a_minor_second_type():
    y = np.array([0, 1, 1, 1, -1])
    coverage = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 12, 0, 0], [12, 0, 0, 1], [5, 0, 0, 0]])
    groups = analysis.subtype_groups(y, coverage)
    assert groups.tolist() == ["aucun", "resp_obstructiveapnea", "resp_centralapnea", "plusieurs_types", "exclu"]
    assert sum(int((groups == kind).sum()) for kind in analysis.GROUPS) == int((y == 1).sum())


@pytest.mark.parametrize(
    "y,coverage",
    [
        ([1], [[0, 0, 0, 0]]),
        ([0], [[1, 0, 0, 0]]),
        ([1], [[31, 0, 0, 0]]),
        ([1], [[float("nan"), 0, 0, 0]]),
    ],
)
def test_inconsistent_groups_are_refused(y, coverage):
    with pytest.raises(ValueError):
        analysis.subtype_groups(np.array(y), np.array(coverage))


def test_summary_does_not_invent_metrics_for_absent_classes():
    result = analysis.summary(np.array([1, 1]), np.array([0.9, 0.1]), 0.5)
    assert result["recall"] == 0.5 and result["false_negative"] == 1
    assert result["specificity"] is None and result["average_precision"] is None
    empty = analysis.summary(np.array([], dtype=int), np.array([]), 0.5)
    assert empty["recall"] is None and empty["average_precision"] is None


def test_recall_ci_matches_explicit_subject_resampling():
    y = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.repeat(["a", "b", "c"], 2)
    pred = {"sleepfm": np.array([0, 1, 0, 0, 0, 0], dtype=bool), "osf": np.array([0, 1, 0, 1, 0, 0], dtype=bool)}
    group = y == 1
    actual = analysis.recall_intervals(y, pred, subjects, group, n_boot=40)
    rng, expected = np.random.default_rng(42), []
    for _ in range(40):
        draw = rng.integers(3, size=3)
        rows = np.concatenate([np.flatnonzero(subjects == ["a", "b", "c"][i]) for i in draw])
        positives = rows[y[rows] == 1]
        expected.append(float(pred["osf"][positives].mean() - pred["sleepfm"][positives].mean()))
    np.testing.assert_allclose(actual["difference_osf_minus_sleepfm"], np.percentile(expected, [2.5, 97.5]))
    absent = analysis.recall_intervals(y, pred, subjects, np.zeros(6, dtype=bool), n_boot=10)
    assert absent["valid_draws"] == 0 and absent["difference_osf_minus_sleepfm"] is None


def test_analysis_on_synthetic_archives_never_trains_and_preserves_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sklearn.linear_model.LogisticRegression.fit", lambda *a, **k: pytest.fail("No fitting allowed")
    )
    subjects = ["s1", "s2"]
    labels_dir, cache_dir, annotations = [tmp_path / n for n in ("labels", "cache", "annotations")]
    for path in (labels_dir, cache_dir, annotations):
        path.mkdir()
    y = np.array([0, 1, 1, 0])
    indices, onsets, stages = np.arange(4), np.arange(4) * 30.0, np.arange(4)
    sources = {}
    for subject in subjects:
        folder = annotations / subject
        folder.mkdir()
        for ext in ("hea", "arousal"):
            (folder / f"{subject}.{ext}").write_text("synthetic")
        np.savez(labels_dir / f"{subject}.npz", y=y, epoch_idx=indices, onset_s=onsets)
        np.savez(cache_dir / f"{subject}.npz", y=stages, epoch_idx=indices, onset_s=onsets)
        sources[subject] = {
            "header_sha256": file_sha256(folder / f"{subject}.hea"),
            "annotation_sha256": file_sha256(folder / f"{subject}.arousal"),
            "labels_sha256": file_sha256(labels_dir / f"{subject}.npz"),
            "embedding_sha256": file_sha256(cache_dir / f"{subject}.npz"),
        }
    (labels_dir / "manifest.json").write_text(json.dumps({"sources": sources}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"status": "terminé"}))
    comparison = tmp_path / "comparison.json"
    predictions = tmp_path / "comparison.predictions.npz"
    proba = np.tile([0.1, 0.8, 0.2, 0.6], 2)
    np.savez(
        predictions,
        subject=np.repeat(subjects, 4),
        y_true=np.tile(y, 2),
        epoch_idx=np.tile(indices, 2),
        onset_s=np.tile(onsets, 2),
        probability_sleepfm=proba,
        probability_osf=proba,
    )
    comparison.write_text(
        json.dumps(
            {
                "status": "terminé",
                "baseline_sha256": file_sha256(baseline),
                "label_manifest_sha256": file_sha256(labels_dir / "manifest.json"),
                "artifacts": {"predictions.npz": {"sha256": file_sha256(predictions)}},
                "partitions": {"test": subjects},
                "thresholds": {"sleepfm": 0.5, "osf": 0.5},
                "metrics": {"sleepfm": {"sensitivity": 0.5}, "osf": {"sensitivity": 0.5}},
            }
        )
    )

    def fake_labels(record, times):
        return y.copy(), np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 10], [0, 0, 0, 0]])

    monkeypatch.setattr(analysis, "annotations_by_type", fake_labels)
    args = SimpleNamespace(
        comparison=comparison,
        baseline=baseline,
        labels_dir=labels_dir,
        sleepfm_dir=cache_dir,
        annotations_dir=annotations,
        plan=Path(__file__),
        out=tmp_path / "output/errors",
    )
    before = {p: p.read_bytes() for p in (baseline, comparison, predictions)}
    report = analysis.run(args)
    assert report["windows"] == 8 and report["positive_windows"] == 4
    assert report["by_type"]["resp_obstructiveapnea"]["models"]["osf"]["recall"] == 1.0
    assert report["by_type"]["resp_hypopnea"]["models"]["osf"]["recall"] == 0.0
    assert "average_precision" not in report["by_type"]["resp_hypopnea"]["models"]["osf"]
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(ValueError, match="existante"):
        analysis.run(args)
    args.out = tmp_path / "new/errors"
    (annotations / "s1/s1.arousal").write_text("changed")
    with pytest.raises(ValueError, match="modifiée"):
        analysis.run(args)
    assert not args.out.parent.exists()
