"""Appariement, bootstrap et comparaison exploratoire sans sélection sur le test."""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from scripts import compare_respiratory_osf as compare
from scripts import respiratory_baseline as baseline
from src.data.cache_contract import file_sha256
from src.data.respiratory_events import MIN_OVERLAP_S, TARGET_EVENTS


def arrays():
    target = {"y": np.array([0, 1, -1]), "epoch_idx": np.array([0, 1, 2]), "onset_s": np.array([0.0, 30.0, 60.0])}
    sleepfm = {"y": np.array([1, 2, 3]), "epoch_idx": target["epoch_idx"].copy(), "onset_s": target["onset_s"].copy()}
    osf = {
        "X": np.zeros((3, 768), np.float32),
        "y": np.array([3, 1, 2]),
        "epoch_idx": np.array([2, 0, 1]),
        "onset_s": np.array([60.0, 0.0, 30.0]),
    }
    return target, sleepfm, osf


def test_osf_rows_are_matched_by_index_not_storage_order():
    target, sleepfm, osf = arrays()
    np.testing.assert_array_equal(compare.align_subject(target, sleepfm, osf), [1, 2, 0])


@pytest.mark.parametrize("problem", ["duplicate", "different_index", "time", "nan_time", "stage", "shape", "nan_X"])
def test_wrong_input_is_refused_before_comparison(problem):
    target, sleepfm, osf = arrays()
    if problem == "duplicate":
        osf["epoch_idx"][0] = 0
    elif problem == "different_index":
        osf["epoch_idx"][0] = 5
    elif problem == "time":
        osf["onset_s"][0] += 1
    elif problem == "nan_time":
        osf["onset_s"][0] = np.nan
    elif problem == "stage":
        osf["y"][0] = 4
    elif problem == "shape":
        osf["X"] = np.zeros((3, 767))
    else:
        osf["X"][0, 0] = np.nan
    with pytest.raises(ValueError):
        compare.align_subject(target, sleepfm, osf)


def test_paired_ap_matches_explicit_resampling_with_tied_scores():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.2, 0.7, 0.7, 0.4, 0.4, 0.4])
    b = np.array([0.1, 0.8, 0.3, 0.6, 0.4, 0.7])
    subjects = np.repeat(["s1", "s2", "s3"], 2)
    actual = compare.paired_ap(y, a, b, subjects, n_boot=30)
    rng, expected = np.random.default_rng(42), []
    for _ in range(30):
        draw = rng.integers(3, size=3)
        rows = np.concatenate([np.flatnonzero(subjects == ["s1", "s2", "s3"][i]) for i in draw])
        expected.append(average_precision_score(y[rows], b[rows]) - average_precision_score(y[rows], a[rows]))
    np.testing.assert_allclose(actual["ci95"], np.percentile(expected, [2.5, 97.5]), atol=1e-14)
    zero = compare.paired_ap(y, a, a, subjects, n_boot=20)
    assert zero["difference_osf_minus_sleepfm"] == 0 and zero["ci95"] == [0.0, 0.0]


def test_output_cannot_modify_input(tmp_path):
    args = SimpleNamespace(
        out=tmp_path / "osf/run",
        sleepfm_dir=tmp_path / "sleepfm",
        osf_dir=tmp_path / "osf",
        labels_dir=tmp_path / "labels",
    )
    with pytest.raises(ValueError, match="entrée"):
        compare.run(args)
    assert not list(tmp_path.iterdir())


def test_full_synthetic_comparison_preserves_reference_and_checks_hashes(tmp_path, monkeypatch):
    split = {"train": [f"s{i:02d}" for i in range(10)], "test": ["test1", "test2"]}
    parts = baseline.partition_subjects(split)
    sleepfm_dir, osf_dir, labels_dir = [tmp_path / name for name in ("sleepfm", "osf", "labels")]
    for folder in (sleepfm_dir, osf_dir, labels_dir):
        folder.mkdir()
    contract = {"checkpoint_sha256": "s", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
    osf_contract = {"model": "OSF", "extraction_preprocessing_version": 2, "osf_checkpoint_sha256": "o"}
    (sleepfm_dir / "manifest.json").write_text(json.dumps(contract))
    (osf_dir / "manifest.json").write_text(json.dumps(osf_contract))
    subjects = split["train"] + split["test"]
    (osf_dir / "run_status.json").write_text(json.dumps({"complete": True, "embedded": subjects, "rejected": {}}))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    y = np.tile([0, 1, -1, 0, 1], 4)
    indices, onsets, stages = np.arange(20), np.arange(20) * 30.0, np.arange(20) % 5
    sources = {}
    for subject in subjects:
        X = np.zeros((20, 768), dtype=np.float32)
        X[:, 0] = y
        np.savez(sleepfm_dir / f"{subject}.npz", y=stages, epoch_idx=indices, onset_s=onsets)
        np.savez(osf_dir / f"{subject}.npz", X=X[::-1], y=stages[::-1], epoch_idx=indices[::-1], onset_s=onsets[::-1])
        np.savez(labels_dir / f"{subject}.npz", y=y, epoch_idx=indices, onset_s=onsets)
        sources[subject] = {
            "embedding_sha256": file_sha256(sleepfm_dir / f"{subject}.npz"),
            "labels_sha256": file_sha256(labels_dir / f"{subject}.npz"),
        }
    (labels_dir / "manifest.json").write_text(json.dumps({"embeddings_manifest": contract, "sources": sources}))
    keep = y >= 0
    truth = np.tile(y[keep], 2)
    probability = np.where(truth == 1, 0.8, 0.2)
    reference_path = tmp_path / "reference.json"
    reference_predictions = tmp_path / "reference.predictions.npz"
    np.savez(
        reference_predictions,
        y_true=truth,
        probability=probability,
        subject=np.repeat(parts["test"], keep.sum()),
        epoch_idx=np.tile(indices[keep], 2),
        onset_s=np.tile(onsets[keep], 2),
    )
    reference_model = tmp_path / "reference.joblib"
    joblib.dump({"fixture": True}, reference_model)
    reference = {
        "status": "terminé",
        "partitions": parts,
        "split_sha256": file_sha256(split_path),
        "target_events": list(TARGET_EVENTS),
        "min_overlap_s": MIN_OVERLAP_S,
        "label_manifest_sha256": file_sha256(labels_dir / "manifest.json"),
        "predictions_sha256": file_sha256(reference_predictions),
        "model_sha256": file_sha256(reference_model),
        "embeddings_manifest": contract,
        "threshold": {"value": 0.5},
        "coverage": {},
        "metrics": baseline.binary_metrics(truth, probability, 0.5),
    }
    reference_path.write_text(json.dumps(reference))
    args = SimpleNamespace(
        baseline=reference_path,
        sleepfm_dir=sleepfm_dir,
        osf_dir=osf_dir,
        labels_dir=labels_dir,
        split_file=split_path,
        plan=Path(__file__),
        out=tmp_path / "new/comparison",
    )
    original = {path: path.read_bytes() for path in (reference_path, reference_predictions, reference_model)}
    bootstrap = compare.paired_ap
    monkeypatch.setattr(compare, "paired_ap", lambda *a: bootstrap(*a, n_boot=10))
    report = compare.run(args)
    assert report["status"] == "terminé" and report["primary"]["ci95"] == [0.0, 0.0]
    assert report["verdict"] == "pas de différence établie"
    assert all(path.read_bytes() == value for path, value in original.items())
    with pytest.raises(ValueError, match="existante"):
        compare.run(args)
    reference_predictions.write_bytes(b"modified")
    with pytest.raises(ValueError, match="modifiées"):
        compare.validated_inputs(args)
