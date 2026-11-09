"""L'égalité numérique ne prouve pas l'identité temporelle des fenêtres."""

import sys

import numpy as np
import pytest

from scripts.compare_embedding_caches import cache_subjects, compare_one, main, validate_tolerances


def save(directory, *, onsets=(0.0, 30.0), **changes):
    directory.mkdir(exist_ok=True)
    arrays = {"X": np.ones((2, 1536), np.float32), "y": np.array([0, 1]), "epoch_idx": np.array([0, 1])}
    if onsets is not None:
        arrays["onset_s"] = np.array(onsets)
    np.savez(directory / "s1.npz", **(arrays | changes))


@pytest.mark.parametrize(
    "onsets,status", [((0.0, 30.0), "identique"), ((3600.0, 3630.0), "différent"), (None, "non vérifié")]
)
def test_origin_times_control_the_claim_of_identity(tmp_path, onsets, status):
    old, new = tmp_path / "old", tmp_path / "new"
    save(old)
    save(new, onsets=onsets)
    actual, reason = compare_one(old, new, "s1", atol=0.001)
    assert actual == status
    if status != "identique":
        assert "temps" in reason


def test_time_tolerance_has_no_relative_component(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    save(old, onsets=(1_000_000.0, 1_000_030.0))
    save(new, onsets=(1_000_000.005, 1_000_030.005))
    assert compare_one(old, new, "s1", 0.001)[0] == "identique"
    save(new, onsets=(1_000_000.5, 1_000_030.5))
    assert compare_one(old, new, "s1", 0.001)[0] == "différent"


def test_real_embedding_difference_is_reported_even_without_times(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    save(old, onsets=None)
    save(new, onsets=None, X=np.zeros((2, 1536), np.float32))
    status, reason = compare_one(old, new, "s1", 0.001)
    assert status == "différent" and "embeddings différents" in reason


def test_rows_are_aligned_by_epoch_index(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    save(old)
    save(new, onsets=(30.0, 0.0), y=np.array([1, 0]), epoch_idx=np.array([1, 0]))
    assert compare_one(old, new, "s1", 0.001)[0] == "identique"


def test_temporaries_are_not_subjects_and_unknown_times_are_not_counted_identical(tmp_path, monkeypatch, capsys):
    old, new = tmp_path / "old", tmp_path / "new"
    save(old, onsets=None)
    save(new)
    for directory in (old, new):
        (directory / "s2.tmp.npz").write_bytes("écriture interrompue".encode())
        assert cache_subjects(directory) == {"s1"}
    monkeypatch.setattr(sys, "argv", ["compare", str(old), str(new)])
    main()
    output = capsys.readouterr().out
    assert "NON VÉRIFIÉ s1" in output
    assert "0 / 1" in output
    assert "s2" not in output


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_bad_tolerances_are_refused(value):
    with pytest.raises(ValueError):
        validate_tolerances(value, 0.01)
    with pytest.raises(ValueError):
        validate_tolerances(0.001, value)
