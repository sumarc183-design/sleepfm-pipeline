"""Labels temporels, absence de fuite et mesure binaire de la nouvelle tâche."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from scripts import respiratory_baseline as baseline
from src.data.respiratory_events import event_intervals, window_targets


def test_intervals_merge_overlaps_and_keep_target_types():
    samples = [0, 5, 10, 20, 25, 35, 40]
    notes = [
        "N2",
        "(resp_obstructiveapnea",
        "(resp_hypopnea",
        "resp_obstructiveapnea)",
        "resp_hypopnea)",
        "(arousal_rera",
        "arousal_rera)",
    ]
    intervals, counts = event_intervals(samples, notes, 60)
    np.testing.assert_array_equal(intervals, [[5, 25]])
    assert counts["resp_obstructiveapnea"] == counts["resp_hypopnea"] == 1


@pytest.mark.parametrize(
    "samples,notes",
    [
        ([5], ["resp_centralapnea)"]),
        ([5], ["(resp_centralapnea"]),
        ([5, 10], ["(resp_centralapnea", "(resp_centralapnea"]),
        ([5, 5], ["(resp_centralapnea", "resp_centralapnea)"]),
        ([20, 10], ["N1", "N2"]),
        ([5.5], ["N1"]),
    ],
)
def test_malformed_intervals_fail(samples, notes):
    with pytest.raises(ValueError):
        event_intervals(samples, notes, 60)


def test_labels_use_original_times_and_half_open_boundaries():
    # 0..30 : aucune cible ; 30..60 : 10 s ; 60..90 : 9 s ; 90..120 : 0.
    intervals = np.array([[30, 40], [60, 69]], dtype=np.int64)
    y, overlap = window_targets([90.0, 30.0, 0.0, 60.0], intervals, 1, 120)
    np.testing.assert_array_equal(y, [0, 1, 0, -1])
    np.testing.assert_array_equal(overlap, [0.0, 10.0, 0.0, 9.0])
    y, overlap = window_targets([0.0], np.empty((0, 2), dtype=np.int64), 200, 6000)
    assert y.tolist() == [0] and overlap.tolist() == [0.0]


@pytest.mark.parametrize(
    "onsets,intervals",
    [
        ([-1.0], [[0, 10]]),
        ([float("nan")], [[0, 10]]),
        ([100.0], [[0, 10]]),
        ([0.0], [[0, 15], [10, 20]]),
    ],
)
def test_invalid_windows_are_rejected(onsets, intervals):
    with pytest.raises(ValueError):
        window_targets(onsets, np.array(intervals), 1, 60)


def test_validation_is_disjoint_deterministic_and_never_uses_test():
    split = {"train": [f"s{i}" for i in range(40)], "test": ["test1", "test2"]}
    parts = baseline.partition_subjects(split)
    assert parts == baseline.partition_subjects(split)
    assert len(parts["fit"]) == 36 and len(parts["validation"]) == 4
    assert set(parts["fit"]) | set(parts["validation"]) == set(split["train"])
    assert not set(parts["fit"]) & set(parts["validation"])
    assert not (set(parts["fit"]) | set(parts["validation"])) & set(parts["test"])


def test_threshold_and_bootstrap_agree_with_explicit_subject_resampling():
    y = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.6, 0.9, 0.3, 0.4])
    subjects = np.repeat(["a", "b", "c"], 2)
    threshold = baseline.select_threshold(y, probability)
    assert 0 <= threshold <= 1
    result = baseline.ap_bootstrap(y, probability, subjects, n_boot=20)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(20):
        draw = rng.integers(3, size=3)
        rows = np.concatenate([np.flatnonzero(subjects == ["a", "b", "c"][i]) for i in draw])
        values.append(average_precision_score(y[rows], probability[rows]))
    np.testing.assert_allclose(result["average_precision"], np.percentile(values, [2.5, 97.5]))
    assert result["valid_draws"] == 20


def test_saved_labels_must_match_embedding_timestamps(tmp_path):
    embeddings, labels = tmp_path / "embeddings", tmp_path / "labels"
    embeddings.mkdir()
    labels.mkdir()
    np.savez(embeddings / "s.npz", X=np.zeros((2, 1536)), epoch_idx=[0, 1], onset_s=[0.0, 30.0])
    np.savez(labels / "s.npz", y=[0, 1], epoch_idx=[0, 1], onset_s=[1.0, 31.0])
    with pytest.raises(ValueError, match="alignés"):
        baseline.load_partition(["s"], embeddings, labels)


def test_output_cannot_replace_input_caches(tmp_path):
    cache = tmp_path / "embeddings"
    cache.mkdir()
    original = cache / "s.npz"
    original.write_bytes(b"input must remain unchanged")
    args = SimpleNamespace(embeddings_dir=cache, labels_dir=cache, out=tmp_path / "out")
    with pytest.raises(ValueError, match="séparés"):
        baseline.run(args)
    assert original.read_bytes() == b"input must remain unchanged"


def test_pipeline_on_synthetic_embeddings_keeps_staging_and_refuses_overwrite(tmp_path, monkeypatch):
    contract = {"checkpoint_sha256": "synthetic", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "manifest.json").write_text(json.dumps(contract))
    split = {"train": [f"s{i:02d}" for i in range(10)], "test": ["test1", "test2"]}
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split))
    staging_bytes = {}
    for subject in split["train"] + split["test"]:
        y = np.tile([0, 1, -1, 0, 1], 4)
        X = np.zeros((20, 1536), dtype=np.float32)
        X[:, 0] = y
        path = embeddings / f"{subject}.npz"
        np.savez(path, X=X, y=np.arange(20) % 5, epoch_idx=np.arange(20), onset_s=np.arange(20) * 30.0)
        staging_bytes[path] = path.read_bytes()

    def fake_labels(record, onsets):
        return {
            "y": y.copy(),
            "overlap_s": np.where(y == 1, 15.0, 0.0),
            "intervals_s": np.empty((0, 2)),
            "event_counts": {},
        }

    annotations = tmp_path / "annotations"
    for subject in split["train"] + split["test"]:
        folder = annotations / subject
        folder.mkdir(parents=True)
        for ext in ("hea", "arousal"):
            (folder / f"{subject}.{ext}").write_text("fixture")
    monkeypatch.setattr(baseline, "read_respiratory_labels", fake_labels)
    monkeypatch.setattr(baseline, "N_BOOT", 10)
    original_bootstrap = baseline.ap_bootstrap
    monkeypatch.setattr(baseline, "ap_bootstrap", lambda *a: original_bootstrap(*a, n_boot=10))
    args = SimpleNamespace(
        embeddings_dir=embeddings,
        annotations_dir=annotations,
        labels_dir=tmp_path / "labels",
        split_file=split_file,
        protocol=Path(__file__),
        download=False,
        out=tmp_path / "out/run",
    )
    report = baseline.run(args)
    assert report["status"] == "terminé" and report["metrics"]["average_precision"] == 1.0
    assert report["coverage"]["test"]["excluded"] == 8
    assert all(path.read_bytes() == before for path, before in staging_bytes.items())
    with pytest.raises(ValueError, match="existante"):
        baseline.run(args)
