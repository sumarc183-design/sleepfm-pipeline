"""Un run complet exige des artefacts valides, pas seulement des noms de fichiers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_baseline as pipeline

CONTRACT = {"checkpoint_sha256": "test", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}


def cache(directory, subject="s1", **changes):
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(CONTRACT))
    arrays = {"X": np.zeros((2, 1536), np.float32), "y": [0, 1], "epoch_idx": [0, 1], "onset_s": [0.0, 30.0]}
    np.savez(directory / f"{subject}.npz", **(arrays | changes))


@pytest.mark.parametrize("problem", ["truncated", "shape", "nonfinite", "times"])
def test_corrupt_cache_never_counts_as_embedded(tmp_path, problem):
    cache(tmp_path)
    if problem == "truncated":
        (tmp_path / "s1.npz").write_bytes(b"incomplet")
    elif problem == "shape":
        cache(tmp_path, X=np.zeros((2, 8)))
    elif problem == "nonfinite":
        cache(tmp_path, X=np.full((2, 1536), np.nan))
    else:
        cache(tmp_path, onset_s=[float("nan"), float("nan")])
    before = (tmp_path / "s1.npz").read_bytes()
    status = pipeline.write_run_status(tmp_path, ["s1"], 0.0)
    assert status["complete"] is False
    assert status["embedded"] == [] and "s1" in status["invalid"]
    assert status["missing"] == []  # présent, mais inutilisable
    assert (tmp_path / "s1.npz").read_bytes() == before
    assert json.loads((tmp_path / "run_status.json").read_text())["invalid"] == status["invalid"]


@pytest.mark.parametrize("reason", [b"", b" \n", b"\xff"])
def test_empty_or_unreadable_rejection_is_not_a_valid_result(tmp_path, reason):
    marker = tmp_path / "s1.skipped"
    marker.write_bytes(reason)
    status = pipeline.write_run_status(tmp_path, ["s1"], 0.0)
    assert not status["complete"] and status["rejected"] == {}
    assert "s1" in status["invalid_rejections"]
    assert marker.read_bytes() == reason


def test_valid_rejections_and_foreign_files_do_not_hide_requested_results(tmp_path, monkeypatch):
    cache(tmp_path)
    (tmp_path / "s2.skipped").write_text("canaux manquants\n")
    (tmp_path / "foreign.npz").write_bytes(b"corrompu, ne pas lire")
    (tmp_path / "other.skipped").write_bytes(b"\xff")
    check = pipeline.cache_problem

    def only_requested(path, **kwargs):
        assert path.stem != "foreign"
        return check(path, **kwargs)

    monkeypatch.setattr(pipeline, "cache_problem", only_requested)
    status = pipeline.write_run_status(tmp_path, ["s1", "s2"], 0.0)
    assert status["complete"] is True
    assert status["embedded"] == ["s1"]
    assert status["rejected"] == {"s2": "canaux manquants"}
    assert status["not_requested"] == ["foreign", "other"]


@pytest.mark.parametrize("final_exists", [False, True])
def test_requested_temporary_file_prevents_completion(tmp_path, final_exists):
    if final_exists:
        cache(tmp_path)
    (tmp_path / "s1.tmp.npz").write_bytes(b"partiel")
    (tmp_path / "foreign.tmp.npz").write_bytes(b"autre sujet")
    status = pipeline.write_run_status(tmp_path, ["s1"], 0.0)
    assert not status["complete"] and status["temporary"] == ["s1"]
    assert status["not_requested"] == ["foreign"]
    assert not any(s.endswith(".tmp") for s in status["embedded"])


@pytest.mark.parametrize("embed_only", [False, True])
def test_invalid_final_status_stops_before_success_or_training(tmp_path, monkeypatch, embed_only):
    raw, out = tmp_path / "raw", tmp_path / "cache"
    (raw / "s1").mkdir(parents=True)
    cache(out, X=np.zeros((2, 8)))
    monkeypatch.setattr(pipeline, "SleepFMEncoders", lambda *a, **k: SimpleNamespace(to=lambda *a: None))
    monkeypatch.setattr(pipeline, "embeddings_manifest", lambda *a: CONTRACT)
    monkeypatch.setattr(pipeline, "embed_subjects_in_chunks", lambda *a, **k: CONTRACT)
    monkeypatch.setattr(pipeline, "load_or_extend_split", lambda *a: pytest.fail("Le découpage ne doit pas changer"))
    monkeypatch.setattr(pipeline, "select_and_evaluate", lambda *a: pytest.fail("Aucun entraînement attendu"))
    args = SimpleNamespace(
        subjects_file=None,
        physionet_subjects=None,
        raw_data=raw,
        device="cpu",
        embeddings_dir=out,
        chunk_size=1,
        num_threads=1,
        batch_size=1,
        retry_skipped=False,
        num_workers=0,
        embed_only=embed_only,
    )
    with pytest.raises(SystemExit, match="Bilan du run incomplet ou invalide"):
        pipeline.run_in_chunks(args, tmp_path / "unused.pt", tmp_path)
    assert json.loads((out / "run_status.json").read_text())["complete"] is False


def test_notebook_reports_invalid_cache_before_export(tmp_path):
    cache(tmp_path, X=np.zeros((2, 8)))
    pipeline.write_run_status(tmp_path, ["s1"], 0.0)
    notebook = Path(__file__).resolve().parents[1] / "notebooks/sleepfm_embeddings_physionet2018.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "status_path =" in "".join(c.get("source", [])))
    # Exécute le contrôle avant la première commande IPython ; aucun export ni pip freeze.
    prefix = source.split("!python", 1)[0]
    context = {
        "WORK": str(tmp_path),
        "N_SUBJECTS": 1,
        "EMB_DIR": str(tmp_path),
        "run_started": 0.0,
        "os": os,
        "json": json,
    }
    with pytest.raises(RuntimeError, match="caches invalides.*s1"):
        exec(compile(prefix, str(notebook), "exec"), context)
