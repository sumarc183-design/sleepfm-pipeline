"""Inférence sans annotations : grille, signal, contrats et sorties."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat, savemat
from sklearn.dummy import DummyClassifier

from run_baseline import build_random_checkpoint, embeddings_manifest
from scripts.generate_synthetic_physionet2018 import generate_subject
from scripts.predict_night import write_predictions
from src.data.cache_contract import file_sha256
from src.data.extract_physionet2018 import process_one_subject
from src.data.raw_night import EPOCH_SAMPLES, RawNightDataset
from src.models.probe_export import save_probes


@pytest.fixture
def night(tmp_path):
    generate_subject(str(tmp_path), "raw-night", 32, 2, 30, 2, np.random.default_rng(0))
    subject = tmp_path / "raw-night"
    # Ce fichier volontairement illisible ne doit jamais être ouvert.
    (subject / "raw-night-arousal.mat").write_bytes(b"annotations interdites")
    return subject


def test_raw_grid_ignores_annotations_and_drops_tail(night):
    dataset = RawNightDataset(night)
    assert len(dataset) == 2
    assert dataset[0].shape == (13, EPOCH_SAMPLES)
    assert dataset[0].dtype == torch.float32
    np.testing.assert_array_equal(dataset.onset_s, [0.0, 30.0])
    assert dataset.metadata()["discarded_tail_s"] == 2
    assert dataset.metadata()["annotations_used"] is False
    with pytest.raises(IndexError):
        dataset[2]
    (night / "raw-night-arousal.mat").unlink()
    np.testing.assert_array_equal(RawNightDataset(night)[0], dataset[0])


def test_raw_reorders_channels_without_changing_values(night):
    reference = RawNightDataset(night)[0]
    hea, mat = night / "raw-night.hea", night / "raw-night.mat"
    lines = hea.read_text().splitlines()
    hea.write_text("\n".join([lines[0], *lines[1:][::-1]]) + "\n")
    savemat(mat, {"val": loadmat(mat)["val"][::-1]})
    np.testing.assert_allclose(RawNightDataset(night)[0], reference, rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["nan", "shape", "missing", "duplicate", "short", "almost_complete"])
def test_raw_rejects_invalid_or_incomplete_input(night, problem):
    hea, mat = night / "raw-night.hea", night / "raw-night.mat"
    lines = hea.read_text().splitlines()
    signal = loadmat(mat)["val"].astype(float)
    if problem == "nan":
        signal[0, 0] = np.nan
    elif problem == "shape":
        signal = signal[:, :-1]
    elif problem == "missing":
        lines[-1] = lines[-1].replace("ECG", "OTHER")
    elif problem == "duplicate":
        lines[-1] = lines[-1].replace("ECG", "ABD")
    else:
        signal = signal[:, : 959 if problem == "almost_complete" else 32]
        fields = lines[0].split()
        fields[3] = str(signal.shape[1])
        lines[0] = " ".join(fields)
    hea.write_text("\n".join(lines) + "\n")
    savemat(mat, {"val": signal})
    with pytest.raises(ValueError):
        RawNightDataset(night)


def test_raw_signal_matches_annotated_processing_when_grid_matches(tmp_path):
    generate_subject(str(tmp_path), "aligned", 32, 2, 30, 0, np.random.default_rng(1))
    subject = tmp_path / "aligned"
    processed = tmp_path / "processed"
    process_one_subject(
        {
            "hea": str(subject / "aligned.hea"),
            "mat": str(subject / "aligned.mat"),
            "arousal_mat": str(subject / "aligned-arousal.mat"),
        },
        str(processed),
        30,
        256,
    )
    reference = np.load(processed / "X" / "aligned" / "aligned_0.npy").astype(np.float32)
    np.testing.assert_array_equal(RawNightDataset(subject)[0], reference)


def test_resampling_rounding_never_pads_an_incomplete_epoch(night):
    # 30 s moins un échantillon à 512 Hz : MNE arrondirait à 7680 à 256 Hz.
    hea, mat = night / "raw-night.hea", night / "raw-night.mat"
    lines = hea.read_text().splitlines()
    lines[0] = "raw-night 13 512 15359"
    hea.write_text("\n".join(lines) + "\n")
    savemat(mat, {"val": np.zeros((13, 15359))})
    with pytest.raises(ValueError, match="aucune fenêtre complète"):
        RawNightDataset(night)


def test_csv_failure_preserves_previous_file(tmp_path):
    class InvalidProbe:
        classes_ = np.arange(5)

        def predict_proba(self, X):
            return np.full((len(X), 5), np.nan)

    path = tmp_path / "night.csv"
    path.write_text("ancien résultat\n")
    groups = (np.array(["s"]), np.array([0]), np.array([0.0]))
    with pytest.raises(ValueError, match="probabilités invalides"):
        write_predictions({"bad": InvalidProbe()}, np.zeros((1, 1536)), groups, path)
    assert path.read_text() == "ancien résultat\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["night.csv"]


@pytest.mark.slow
def test_raw_cli_with_real_encoders_and_exported_probe(night, tmp_path):
    checkpoint = tmp_path / "random.pt"
    build_random_checkpoint(checkpoint)
    model_dir, out = tmp_path / "models", tmp_path / "predictions"
    probe = DummyClassifier(strategy="prior").fit(np.zeros((5, 1536)), np.arange(5))
    save_probes(model_dir, {"dummy": probe}, 1536, embeddings_manifest(checkpoint))
    command = [
        sys.executable,
        "scripts/predict_raw_night.py",
        "raw-night",
        "--raw-data",
        str(tmp_path),
        "--models",
        str(model_dir),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--batch-size",
        "1",
    ]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with (out / "raw-night.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["epoch_idx"]) for row in rows] == [0, 1]
    assert [float(row["onset_s"]) for row in rows] == [0.0, 30.0]
    assert all(row["dummy_stade"] == "wake" for row in rows)
    metadata = json.loads((out / "raw-night.json").read_text())
    assert metadata["annotations_used"] is False
    assert metadata["csv_sha256"] == file_sha256(out / "raw-night.csv")
    assert metadata["source_files"]["raw-night.mat"] == file_sha256(night / "raw-night.mat")
    assert metadata["signal_contract"] == embeddings_manifest(checkpoint)
    # Refus d'un autre checkpoint avant même la lecture du signal.
    checkpoint.write_bytes(b"autres poids")
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "checkpoint_sha256" in result.stderr
    assert (out / "raw-night.csv").read_bytes()  # la sortie précédente reste présente
