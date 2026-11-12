"""Référence de majorité stricte, exclusions et protection du parcours figé."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_grid_transitions as evaluation
from src.data.cache_contract import file_sha256
from src.data.majority_grid_labels import labels_from_counts, read_majority_grid


def test_uniform_majority_ties_and_partial_undefined_are_distinguished():
    counts = np.array([[20, 0, 0, 0, 0], [20, 10, 0, 0, 0], [0, 15, 5, 10, 0], [0, 0, 0, 30, 0]])
    r = labels_from_counts(counts, np.array([10, 0, 0, 0]), 1)
    assert r["category"].tolist() == ["partial_undefined", "transition_majority", "transition_no_majority", "uniform"]
    assert r["y_uniform"].tolist() == [-1, -1, -1, 3]
    assert r["y_majority"].tolist() == [-1, 0, -1, 3]
    np.testing.assert_allclose(r["stage_duration_s"].sum(1) + r["undefined_s"], 30.0)


def test_exact_half_is_excluded_and_one_extra_sample_defines_majority():
    half = labels_from_counts(np.array([[3000, 3000, 0, 0, 0]]), np.array([0]), 200)
    assert half["y_majority"].tolist() == [-1]
    shifted = labels_from_counts(np.array([[3001, 2999, 0, 0, 0]]), np.array([0]), 200)
    assert shifted["y_majority"].tolist() == [0] and shifted["category"].tolist() == ["transition_majority"]
    near = labels_from_counts(np.array([[5999, 1, 0, 0, 0]]), np.array([0]), 200)
    assert near["y_uniform"].tolist() == [-1] and near["category"].tolist() == ["near_uniform"]


def test_annotations_without_stages_stay_undefined():
    r = labels_from_counts(np.zeros((1, 5), dtype=int), np.array([30]), 1)
    assert r["category"].tolist() == ["undefined"] and r["y_majority"].tolist() == [-1]


@pytest.mark.parametrize(
    "counts,undefined",
    [
        ([[30, 1, 0, 0, 0]], [0]),
        ([[-1, 31, 0, 0, 0]], [0]),
        ([[30.0, 0, 0, 0, 0]], [0]),
        ([[30, 0, 0, 0]], [0]),
    ],
)
def test_invalid_counts_fail(counts, undefined):
    with pytest.raises(ValueError):
        labels_from_counts(np.asarray(counts), np.asarray(undefined), 1)


def test_h5_reader_uses_sample_indicators_and_rejects_invalid_one_hot(tmp_path):
    import h5py

    from scripts.generate_synthetic_physionet2018 import write_hea
    from src.data.epoch_dataset import STAGE_NAME_TO_LABEL

    record = tmp_path / "night"
    write_hea(str(record.with_suffix(".hea")), "night", 2, 120)
    h5 = tmp_path / "night-arousal.mat"
    with h5py.File(h5, "w") as f:
        group = f.create_group("data/sleep_stages")
        for name in [*STAGE_NAME_TO_LABEL, "undefined"]:
            values = np.zeros((1, 120), dtype=np.int16)
            if name == "wake":
                values[0, :60] = 1
            if name == "nonrem1":
                values[0, 60:] = 1
            group.create_dataset(name, data=values)
    r = read_majority_grid(record, np.array([0.0, 30.0]))
    assert r["y_uniform"].tolist() == [0, 1]
    with h5py.File(h5, "r+") as f:
        f["data/sleep_stages/wake"][0, 0] = 0
    with pytest.raises(ValueError, match="Un seul stade"):
        read_majority_grid(record, np.array([0.0, 30.0]))


def test_empty_or_degenerate_score_is_explicit():
    assert evaluation.stage_scores(np.array([]), np.array([]), np.array([])) is None
    r = evaluation.stage_scores(np.array([1, 1]), np.array([1, 1]), np.array(["s", "s"]))
    assert r["cohen_kappa"] is None and r["ci95"] is None and r["accuracy"] == 1.0


def test_wrong_grid_times_are_refused():
    data = {
        "X": np.zeros((2, 1536), dtype=np.float32),
        "y_grid": np.array([0, 1]),
        "epoch_idx": np.array([0, 1]),
        "onset_s": np.array([0.0, 31.0]),
    }
    with pytest.raises(ValueError, match="grille"):
        evaluation.check_grid(data)


def test_frozen_pipeline_validates_old_labels_before_any_new_score(tmp_path, monkeypatch):
    cache, annotations, models = [tmp_path / name for name in ("cache", "annotations", "models")]
    for directory in (cache, annotations, models):
        directory.mkdir()
    contract = {"checkpoint_sha256": "synthetic", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
    (cache / "manifest.json").write_text(json.dumps(contract | {"epoch_selection": "fixed_30s_from_recording_start"}))
    subjects = ["test1", "test2"]
    (cache / "run_status.json").write_text(json.dumps({"complete": True, "done": subjects, "rejected": {}}))
    (models / "manifest.json").write_text("{}")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["train1"], "test": subjects}))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {"contract": contract, "split_sha256": file_sha256(split), "models": {"logistic": "a", "sequence": "b"}}
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "models_manifest_sha256": file_sha256(models / "manifest.json"),
                "windows": {"total": 8},
                "grid": {k: {"cohen_kappa": 1.0, "macro_f1": 1.0} for k in evaluation.PROBES},
            }
        )
    )
    sources = {}
    for subject in subjects:
        X = np.zeros((4, 1536), dtype=np.float32)
        X[:, 0] = np.arange(4)
        np.savez(
            cache / f"{subject}.npz", X=X, y_grid=[0, -1, -1, 3], epoch_idx=np.arange(4), onset_s=np.arange(4) * 30.0
        )
        folder = annotations / subject
        folder.mkdir()
        for extension in ("hea", "arousal"):
            (folder / f"{subject}.{extension}").write_text("fixture")
        (folder / f"{subject}-arousal.mat").write_bytes(b"fixture")
        sources[subject] = {
            "header_sha256": file_sha256(folder / f"{subject}.hea"),
            "annotation_sha256": file_sha256(folder / f"{subject}.arousal"),
        }
    annotation_manifest = tmp_path / "annotations.json"
    annotation_manifest.write_text(json.dumps({"sources": sources}))
    args = SimpleNamespace(
        embeddings_dir=cache,
        annotations_dir=annotations,
        models=models,
        split_file=split,
        frozen=frozen,
        reference=reference,
        annotation_manifest=annotation_manifest,
        plan=Path(__file__),
        out=tmp_path / "output/run",
    )

    class FrozenProbe:
        classes_ = np.arange(5)

        def fit(self, *a):
            pytest.fail("Aucun entraînement")

        def predict_proba(self, X):
            return np.eye(5)[X[:, 0].astype(int)]

    def labels(*a):
        return {
            "y_uniform": np.array([0, -1, -1, 3]),
            "y_majority": np.array([0, 1, -1, 3]),
            "category": np.array(["uniform", "transition_majority", "transition_no_majority", "uniform"]),
        }

    monkeypatch.setattr(evaluation, "read_majority_grid", labels)
    monkeypatch.setattr(
        evaluation, "load_probes", lambda *a, **k: ({p: FrozenProbe() for p in evaluation.PROBES}, {"loaded": {}})
    )
    before = {p: p.read_bytes() for p in [reference, split, *cache.glob("*.npz")]}
    report = evaluation.run(args)
    assert report["counts"]["uniform"] == 4 and report["counts"]["transition_majority"] == 2
    assert report["metrics"]["sequence"]["all_majority"]["windows"] == 6
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(ValueError, match="existante"):
        evaluation.run(args)
    args.out = tmp_path / "failed/run"
    old_labels = labels()
    old_labels["y_uniform"][0] = -1
    monkeypatch.setattr(evaluation, "read_majority_grid", lambda *a: old_labels)
    monkeypatch.setattr(evaluation, "load_probes", lambda *a, **k: pytest.fail("Labels à valider avant modèle"))
    with pytest.raises(ValueError, match="uniformes différents"):
        evaluation.run(args)
    assert not args.out.parent.exists()
