"""Couverture conditionnelle et appariement événementiel sans lecture du test."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_respiratory_events as evaluation
from src.data.cache_contract import file_sha256
from src.data.respiratory_event_eval import covered_duration, individual_events, match_one_to_one
from src.data.respiratory_events import TARGET_EVENTS


def test_individual_events_keep_touching_events_and_overlapping_types_separate():
    samples = np.array([0, 5, 10, 10, 15, 20])
    notes = [
        "(resp_hypopnea",
        "(resp_centralapnea",
        "resp_hypopnea)",
        "(resp_hypopnea",
        "resp_centralapnea)",
        "resp_hypopnea)",
    ]
    result = individual_events(samples, notes, 30)
    assert result.tolist() == [[0, 10, 3], [5, 15, 1], [10, 20, 3]]
    with pytest.raises(ValueError, match="sans fin"):
        individual_events(np.array([0]), ["(resp_hypopnea"], 30)


def test_coverage_uses_union_and_preserves_gaps_without_double_counting():
    refs = np.array([[0.0, 30.0], [10.0, 20.0], [40.0, 50.0]])
    windows = np.array([[0.0, 10.0], [5.0, 15.0], [20.0, 30.0]])
    np.testing.assert_allclose(covered_duration(refs, windows), [25.0, 5.0, 0.0])
    np.testing.assert_array_equal(covered_duration(refs, []), [0.0, 0.0, 0.0])
    assert covered_duration(np.empty((0, 2)), windows).shape == (0,)
    with pytest.raises(ValueError):
        covered_duration(refs, [[3.0, 2.0]])


def test_matching_maximizes_cardinality_before_iou_not_a_greedy_pair_choice():
    overlap = np.array([[20.0, 10.0], [10.0, 0.0]])
    match = match_one_to_one(overlap, np.array([30.0, 30.0]), np.array([30.0, 30.0]))
    assert match.tolist() == [1, 0]
    # One pair only: prefer the larger IoU when cardinality is tied.
    match = match_one_to_one(np.array([[10.0, 20.0]]), np.array([30.0]), np.array([30.0, 30.0]))
    assert match.tolist() == [1]


def test_matching_tolerance_and_unmatched_shapes_are_explicit():
    assert match_one_to_one(np.array([[9.995]]), np.array([20.0]), np.array([20.0])).tolist() == [0]
    assert match_one_to_one(np.array([[9.98]]), np.array([20.0]), np.array([20.0])).tolist() == [-1]
    assert match_one_to_one(np.empty((2, 0)), np.array([10.0, 10.0]), np.array([])).tolist() == [-1, -1]
    with pytest.raises(ValueError):
        match_one_to_one(np.array([[30.0]]), np.array([20.0]), np.array([20.0]))
    with pytest.raises(ValueError):
        match_one_to_one(np.array([[10.0]]), np.array(20.0), np.array([20.0]))


def test_one_long_sequence_can_cover_two_events_but_match_only_one():
    refs = np.array([[0.0, 10.0], [20.0, 30.0]])
    windows = np.array([[0.0, 30.0]])
    r = evaluation.event_scores(refs, windows, np.array([True]), np.array([0]), np.array([1]))
    assert r["eligible"].tolist() == [True, True] and r["covered"].sum() == 2
    assert (r["matched_run"] >= 0).sum() == 1


def test_partial_coverage_and_short_events_are_excluded_from_recall():
    refs = np.array([[0.0, 30.0], [40.0, 48.0]])
    windows = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 70.0]])
    r = evaluation.event_scores(refs, windows, np.ones(3, dtype=bool), np.arange(3), np.arange(1, 4))
    np.testing.assert_allclose(r["available_s"], [20.0, 8.0])
    assert r["full"].tolist() == [False, True]
    assert not r["eligible"].any() and not r["covered"].any()
    assert r["matched_run"].tolist() == [-1, -1]


def test_bootstrap_recall_matches_explicit_night_resampling_and_handles_no_eligible():
    den = np.array([2, 3, 0])
    covered = np.array([2, 1, 0])
    matched = np.array([1, 1, 0])
    r = evaluation.recall_summary(den, covered, matched)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(500):
        rows = rng.choice(3, 3)
        if den[rows].sum():
            values.append([covered[rows].sum() / den[rows].sum(), matched[rows].sum() / den[rows].sum()])
    ci = np.percentile(values, [2.5, 97.5], axis=0).T
    for j, key in enumerate(["positive_window_coverage", "one_to_one_matching"]):
        np.testing.assert_allclose(r[key]["ci95"], ci[j])
    empty = evaluation.recall_summary([0, 0], [0, 0], [0, 0])
    assert empty["one_to_one_matching"]["recall"] is None and empty["one_to_one_matching"]["ci95"] is None


def test_existing_output_and_annotation_directory_are_protected(tmp_path):
    out = tmp_path / "result"
    out.with_suffix(".json").write_text("keep")
    with pytest.raises(ValueError, match="Sortie existante"):
        evaluation.run(SimpleNamespace(out=out))
    assert out.with_suffix(".json").read_text() == "keep"
    with pytest.raises(ValueError, match="entrée"):
        evaluation.run(SimpleNamespace(out=tmp_path / "annotations/out", annotations_dir=tmp_path / "annotations"))


def test_validation_pipeline_verifies_labels_and_does_not_open_test_predictions(tmp_path, monkeypatch):
    from scripts.generate_synthetic_physionet2018 import write_hea

    subjects = ["tr03-0001", "tr03-0002"]
    parts = {"fit": ["fit"], "validation": subjects, "test": ["unopened_test"]}
    folder = tmp_path / "annotations"
    sources = {}
    for subject in subjects:
        directory = folder / subject
        directory.mkdir(parents=True)
        record = directory / subject
        write_hea(str(record.with_suffix(".hea")), subject, 1, 120)
        record.with_suffix(".arousal").write_bytes(b"fixture")
        sources[subject] = {
            "header_sha256": file_sha256(record.with_suffix(".hea")),
            "annotation_sha256": file_sha256(record.with_suffix(".arousal")),
            "event_counts": {kind: int(kind == "resp_hypopnea") for kind in TARGET_EVENTS},
        }
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"sources": sources}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"status": "terminé", "partitions": parts, "label_manifest_sha256": file_sha256(manifest)})
    )
    y = np.tile([1, 1, 0, 0], 2)
    p = np.tile([0.9, 0.9, 0.1, 0.1], 2)
    permutation = np.array([2, 0, 3, 1, 6, 4, 7, 5])
    arrays = {
        "subject": np.repeat(subjects, 4),
        "epoch_idx": np.tile(np.arange(4), 2),
        "onset_s": np.tile(np.arange(4) * 30.0, 2),
        "y_true": y,
        "probability_sleepfm": p,
        "probability_osf": p,
    }
    archive = tmp_path / "validation.npz"
    np.savez(archive, **{k: v[permutation] for k, v in arrays.items()})
    reference = {
        "status": "terminé",
        "source_sha256": {"baseline": file_sha256(baseline)},
        "partitions": parts,
        "artifacts": {"validation.npz": {"file": archive.name, "sha256": file_sha256(archive)}},
        "results": {
            model: {
                point: {"threshold": 0.5, "validation": {"confusion_matrix": [[4, 0], [0, 4]]}}
                for point in evaluation.POINTS
            }
            for model in evaluation.MODELS
        },
    }
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(reference))
    plan = tmp_path / "plan.md"
    plan.write_text("fixed plan")

    def read(record, extension):
        assert record.split("/")[-1] in subjects
        return SimpleNamespace(fs=1, sample=np.array([10, 50]), aux_note=["(resp_hypopnea", "resp_hypopnea)"])

    monkeypatch.setitem(sys.modules, "wfdb", SimpleNamespace(rdann=read, __version__="fixture"))
    args = SimpleNamespace(
        out=tmp_path / "output",
        annotations_dir=folder,
        thresholds=thresholds,
        baseline=baseline,
        labels_manifest=manifest,
        plan=plan,
    )
    r = evaluation.run(args)
    assert r["coverage"] == {"total": 2, "complete": 2, "partial": 0, "none": 0, "too_short": 0, "eligible": 2}
    assert r["subjects"] == subjects
    assert r["results"]["osf"]["initial_f1"]["one_to_one_matching"]["recall"] == 1.0
    # Readers must stop on an annotation hash mismatch, before new output.
    (folder / subjects[0] / f"{subjects[0]}.arousal").write_bytes(b"changed")
    args.out = tmp_path / "bad"
    with pytest.raises(ValueError, match="Annotation modifiée"):
        evaluation.run(args)
    assert not (tmp_path / "bad.json").exists()
    # No test archive was created in this fixture: successful run used validation only.
    assert not (tmp_path / "test.predictions.npz").exists()
