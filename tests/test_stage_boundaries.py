"""Frontières de stade : positions, référence figée et décompte des erreurs."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import analyse_stage_boundaries as analysis
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL
from src.data.majority_grid_labels import labels_from_counts


def fixture_grid():
    grid = np.zeros((6, 30), dtype=np.int8)
    grid[1:, :] = 1
    grid[1, 0] = 0
    grid[2, -1] = 2
    grid[3, 12] = 3
    grid[4, :16] = 2
    grid[5, :] = -1
    counts = np.stack([(grid == i).sum(1) for i in range(5)], axis=1)
    labels = labels_from_counts(counts, (grid == -1).sum(1), 1)
    return grid, labels


def test_first_last_and_inside_are_distinguished_and_neighbors_stay_within_night():
    grid, labels = fixture_grid()
    r = analysis.boundary_profiles(grid, 1, labels["y_majority"], labels["category"])
    assert r["minority"].tolist() == [-1, 0, 2, 3, -1, -1]
    assert r["position"].tolist() == ["na", "first", "last", "inside", "na", "na"]
    assert r["previous"].tolist() == [-1, 0, 1, 1, 1, 2]
    assert r["following"].tolist() == [1, 1, 1, 2, -1, -1]


def test_different_archived_labels_or_categories_stop_analysis():
    grid, labels = fixture_grid()
    wrong = labels["y_majority"].copy()
    wrong[1] = 2
    with pytest.raises(ValueError, match="archivé"):
        analysis.boundary_profiles(grid, 1, wrong, labels["category"])
    wrong = labels["category"].copy()
    wrong[1] = "uniform"
    with pytest.raises(ValueError, match="archivé"):
        analysis.boundary_profiles(grid, 1, labels["y_majority"], wrong)


@pytest.mark.parametrize("rate", [0, 1.5, True])
def test_invalid_sampling_rate_is_rejected(rate):
    grid, labels = fixture_grid()
    with pytest.raises(ValueError, match="invalide"):
        analysis.boundary_profiles(grid, rate, labels["y_majority"], labels["category"])


def test_outcomes_are_exclusive_and_empty_groups_have_no_rates():
    y = np.array([0, 1, 2, 3])
    minority = np.array([1, 0, 1, 2])
    pred = {"sequence": np.array([0, 0, 4, 3])}
    s = np.array(["night1", "night1", "night2", "night2"])
    r = analysis.outcome_summary(y, minority, pred, s, np.ones(4, dtype=bool), bootstrap=True)
    m = r["models"]["sequence"]
    assert [m[k]["count"] for k in analysis.ISSUES] == [2, 1, 1]
    assert m["majority"]["rate"] == 0.5
    # Independent explicit-row bootstrap, including complete nights.
    rng = np.random.default_rng(42)
    values = []
    for _ in range(500):
        rows = np.concatenate([np.flatnonzero(s == sid) for sid in rng.choice(np.unique(s), 2)])
        values.append((pred["sequence"][rows] == y[rows]).mean())
    np.testing.assert_allclose(m["majority"]["ci95"], np.percentile(values, [2.5, 97.5]))
    empty = analysis.outcome_summary(y, minority, pred, s, np.zeros(4, dtype=bool))
    assert empty == {"windows": 0, "nights": 0, "models": {}}


def test_h5_reconstruction_rejects_overlapping_stage_indicators(tmp_path):
    from scripts.generate_synthetic_physionet2018 import write_hea

    record = tmp_path / "night"
    write_hea(str(record.with_suffix(".hea")), "night", 1, 60)
    with h5py.File(tmp_path / "night-arousal.mat", "w") as f:
        g = f.create_group("data/sleep_stages")
        for name in [*STAGE_NAME_TO_LABEL, "undefined"]:
            a = np.zeros((1, 60), dtype=np.int8)
            if name == "wake":
                a[:] = 1
            g.create_dataset(name, data=a)
    samples, rate = analysis.read_samples(record, 2)
    assert samples.shape == (2, 30) and rate == 1 and not samples.any()
    with h5py.File(tmp_path / "night-arousal.mat", "r+") as f:
        f["data/sleep_stages/nonrem1"][0, 0] = 1
    with pytest.raises(ValueError, match="Un seul stade"):
        analysis.read_samples(record, 2)


def test_existing_output_is_protected_before_reading_inputs(tmp_path):
    out = tmp_path / "result"
    out.with_suffix(".json").write_text("keep")
    with pytest.raises(ValueError, match="Sortie existante"):
        analysis.run(SimpleNamespace(out=out))
    assert out.with_suffix(".json").read_text() == "keep"


def test_pipeline_checks_source_hashes_and_reconstructed_labels(tmp_path, monkeypatch):
    import json

    from sklearn.metrics import confusion_matrix

    from src.data.cache_contract import file_sha256

    grid, target = fixture_grid()
    y = np.tile(target["y_majority"], 2)
    category = np.tile(target["category"], 2)
    subjects = ["tr03-0001", "tr03-0002"]
    arrays = {
        "subject": np.repeat(subjects, len(grid)),
        "epoch_idx": np.tile(np.arange(len(grid)), 2),
        "onset_s": np.tile(np.arange(len(grid)) * 30.0, 2),
        "y_majority": y,
        "category": category,
        "pred_logistic": np.zeros(len(y), dtype=int),
        "pred_sequence": np.zeros(len(y), dtype=int),
    }
    np.savez(tmp_path / "source.predictions.npz", **arrays)
    inventory = {}
    annotations = tmp_path / "annotations"
    for subject in subjects:
        record = annotations / subject
        record.mkdir(parents=True)
        h5 = record / f"{subject}-arousal.mat"
        h5.write_bytes(b"h5 fixture")
        hea = record / f"{subject}.hea"
        hea.write_bytes(b"header fixture")
        inventory[subject] = {"h5_sha256": file_sha256(h5), "header_sha256": file_sha256(hea)}
    (tmp_path / "source.inputs.json").write_text(json.dumps(inventory))
    source = tmp_path / "source.json"
    report = {
        "status": "terminé",
        "windows": len(y),
        "nights": 2,
        "counts": {c: int((category == c).sum()) for c in np.unique(category)},
        "artifacts": {
            ext: {"file": f"source.{ext}", "sha256": file_sha256(tmp_path / f"source.{ext}")}
            for ext in ["predictions.npz", "inputs.json"]
        },
        "code_files": {
            "src/data/majority_grid_labels.py": file_sha256(
                analysis.Path(analysis.__file__).resolve().parents[1] / "src/data/majority_grid_labels.py"
            )
        },
        "metrics": {
            kind: {
                group: {
                    "confusion_matrix": confusion_matrix(
                        y[mask], arrays[f"pred_{kind}"][mask], labels=range(5)
                    ).tolist()
                }
                for group, mask in [
                    ("uniform", category == "uniform"),
                    ("near_uniform", category == "near_uniform"),
                    ("all_majority", y >= 0),
                ]
            }
            for kind in analysis.PROBES
        },
    }
    source.write_text(json.dumps(report))
    plan = tmp_path / "plan.md"
    plan.write_text("fixed plan")
    args = SimpleNamespace(source=source, annotations_dir=annotations, plan=plan, out=tmp_path / "result")
    monkeypatch.setattr(analysis, "read_samples", lambda record, n: (grid, 1))
    result = analysis.run(args)
    assert result["overall"]["windows"] == 6
    assert result["position"] == {"first": 2, "last": 2, "inside": 2}
    # Corrupt reconstruction must fail before writing any result.
    changed = grid.copy()
    changed[0, 0] = 4
    monkeypatch.setattr(analysis, "read_samples", lambda record, n: (changed, 1))
    args.out = tmp_path / "bad_labels"
    with pytest.raises(ValueError, match="archivé"):
        analysis.run(args)
    assert not (tmp_path / "bad_labels.json").exists()
    # A modified artifact is refused even when np.load could still read it.
    with (tmp_path / "source.predictions.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Artefact source modifié"):
        analysis.run(args)
