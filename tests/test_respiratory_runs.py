"""Regroupement temporel, lacunes, séquences mixtes et comptes conservés."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyse_respiratory_runs as analysis
from src.data.cache_contract import file_sha256


def test_runs_break_at_negative_windows_subjects_indices_and_time_gaps():
    subject = np.array(["a"] * 7 + ["b"] * 2)
    idx = np.array([0, 1, 2, 3, 4, 6, 7, 0, 1])
    onset = np.array([0.0, 30.0, 60.0, 90.0, 150.0, 180.0, 210.0, 0.0, 30.0])
    mask = np.array([True, True, False, True, True, True, True, True, True])
    analysis.check_rows(subject, idx, onset)
    start, end = analysis.consecutive_runs(mask, subject, idx, onset)
    assert start.tolist() == [0, 3, 4, 5, 7]
    assert end.tolist() == [2, 4, 5, 7, 9]
    assert int((end - start).sum()) == mask.sum()


def test_small_time_jitter_is_allowed_but_longer_gaps_are_not_joined():
    subject = np.array(["a"] * 3)
    idx = np.arange(3)
    onset = np.array([0.0, 29.995, 60.015])
    analysis.check_rows(subject, idx, onset)
    a, b = analysis.consecutive_runs(np.ones(3, dtype=bool), subject, idx, onset)
    assert a.tolist() == [0, 2] and b.tolist() == [2, 3]


@pytest.mark.parametrize("problem", ["duplicate", "reversed", "overlap", "non_grouped", "nan"])
def test_invalid_chronology_is_refused(problem):
    s = np.array(["a", "a", "b", "b"])
    i = np.array([0, 1, 0, 1])
    t = np.array([0.0, 30.0, 0.0, 30.0])
    if problem == "duplicate":
        i[1] = 0
    elif problem == "reversed":
        i[:2] = [1, 0]
    elif problem == "overlap":
        t[1] = 29.0
    elif problem == "non_grouped":
        s[:] = ["a", "b", "a", "b"]
        i[:] = [0, 0, 1, 1]
        t[:] = [0, 0, 30, 30]
    else:
        t[1] = np.nan
    with pytest.raises(ValueError):
        analysis.check_rows(s, i, t)


def test_mixed_positive_sequence_contains_two_separate_false_positive_sequences():
    s = np.array(["a"] * 3)
    i = np.arange(3)
    t = i * 30.0
    r, intervals = analysis.analyse_configuration(np.array([0, 1, 0]), np.array([0.9, 0.9, 0.9]), 0.5, s, i, t)
    assert r["predicted_positive_runs"] == {"count": 1, "negative_only": 0, "positive_only": 0, "mixed": 1}
    assert r["false_positive_windows"] == 2 and r["false_positive_runs"]["count"] == 2
    assert r["false_positive_runs"]["nominal_minutes"] == 1.0
    pred = [x for x in intervals if x["kind"] == "predicted_positive"][0]
    assert pred["true_positive_windows"] == 1 and pred["false_positive_windows"] == 2
    assert pred["start_s"] == 0.0 and pred["end_s"] == 90.0


def test_no_prediction_gives_zero_burden_and_undefined_duration_quantiles():
    r, intervals = analysis.analyse_configuration(
        np.array([0, 1]), np.array([0.1, 0.1]), 0.5, np.array(["a", "a"]), np.arange(2), np.array([0.0, 30.0])
    )
    assert not intervals and r["false_positive_runs"]["count"] == 0
    assert r["false_positive_runs"]["median_s"] is None
    assert r["false_positive_runs_per_evaluated_hour"]["ci95"] == [0.0, 0.0]


def test_rate_bootstrap_resamples_whole_nights_with_exposure_weights():
    counts = np.array([2, 1, 0])
    hours = np.array([1.0, 2.0, 3.0])
    r = analysis.rate_interval(counts, hours, n_boot=30)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(30):
        selected = rng.choice(3, 3)
        expected.append(counts[selected].sum() / hours[selected].sum())
    assert r["value"] == 0.5
    np.testing.assert_allclose(r["ci95"], np.percentile(expected, [2.5, 97.5]))
    with pytest.raises(ValueError):
        analysis.rate_interval(counts, np.array([1.0, 0.0, 3.0]))


def test_existing_output_is_preserved_before_reading_inputs(tmp_path):
    out = tmp_path / "result"
    out.with_suffix(".json").write_text("keep")
    with pytest.raises(ValueError, match="Sortie existante"):
        analysis.run(SimpleNamespace(out=out))
    assert out.with_suffix(".json").read_text() == "keep"


@pytest.mark.parametrize("reorder", [False, True])
def test_pipeline_preserves_reference_counts_and_checks_artifact_integrity(tmp_path, reorder):
    s = np.repeat(["tr03-0001", "tr03-0002"], 4)
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
    rows = {
        "subject": s,
        "epoch_idx": np.tile(np.arange(4), 2),
        "onset_s": np.tile(np.arange(4) * 30.0, 2),
        "y_true": y,
        "probability_sleepfm": p,
        "probability_osf": p,
    }
    if reorder:
        permutation = np.array([2, 0, 3, 1, 6, 4, 7, 5])
        rows = {key: value[permutation] for key, value in rows.items()}
    archive = tmp_path / "predictions.npz"
    np.savez(archive, **rows)
    parts = {"test": ["tr03-0001", "tr03-0002"]}
    comparison = {
        "status": "terminé",
        "partitions": parts,
        "artifacts": {"predictions.npz": {"file": archive.name, "sha256": file_sha256(archive)}},
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison))
    point = {
        "threshold": 0.5,
        "test": {"confusion_matrix": [[2, 3], [1, 2]], "false_positive_windows": 3, "predicted_positive": 5},
    }
    reference = {
        "status": "terminé",
        "source_sha256": {"comparison": file_sha256(comparison_path)},
        "partitions": parts,
        "results": {m: {p: point for p in analysis.POINTS} for m in analysis.MODELS},
    }
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(reference))
    plan = tmp_path / "plan.md"
    plan.write_text("fixed plan")
    args = SimpleNamespace(out=tmp_path / "result", thresholds=thresholds, comparison=comparison_path, plan=plan)
    result = analysis.run(args)
    row = result["results"]["osf"]["initial_f1"]
    assert row["false_positive_runs"]["count"] == 3
    assert row["predicted_positive_runs"]["mixed"] == 2
    assert result["ordering"]["nights_reordered"] == (2 if reorder else 0)
    with np.load(tmp_path / "result.ordering.npz") as order:
        np.testing.assert_array_equal(rows["epoch_idx"][order["original_row"]], np.tile(np.arange(4), 2))
    before = thresholds.read_bytes()
    # A source with mismatching window counts is refused before output.
    reference["results"]["osf"]["initial_f1"]["test"]["false_positive_windows"] = 4
    thresholds.write_text(json.dumps(reference))
    args.out = tmp_path / "wrong_count"
    with pytest.raises(ValueError, match="comptes"):
        analysis.run(args)
    assert not (tmp_path / "wrong_count.json").exists()
    thresholds.write_bytes(before)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive de prédictions modifiée"):
        analysis.run(args)
