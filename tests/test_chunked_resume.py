"""Une reprise ne modifie pas les données avant contrôle de leur contrat."""

import json
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from run_baseline import embed_subjects_in_chunks
from src.data.cache_contract import import_embeddings

CONTRACT = {"checkpoint_sha256": "aaa", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}


def run(directory, subjects, **overrides):
    def unexpected_fetch(*args):
        pytest.fail("Aucun téléchargement attendu")

    arguments = {
        "subject_ids": subjects,
        "fetch_subject": unexpected_fetch,
        "encoders": None,
        "embeddings_dir": directory,
        "chunk_size": 2,
        "num_threads": 1,
        "batch_size": 1,
        "device": "cpu",
        "manifest": CONTRACT,
        "num_workers": 0,
    }
    embed_subjects_in_chunks(**(arguments | overrides))


@pytest.mark.parametrize("mismatch", ["checkpoint_sha256", "preprocessing_version"])
def test_contract_mismatch_preserves_corrupt_cache_and_markers(tmp_path, mismatch):
    (tmp_path / "manifest.json").write_text(json.dumps(CONTRACT))
    (tmp_path / "s1.npz").write_bytes(b"corrompu")
    (tmp_path / "s1.tmp.npz").write_bytes(b"partiel")
    (tmp_path / "s1.skipped").write_text("rejet historique")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    changed = CONTRACT | {mismatch: "autre" if mismatch == "checkpoint_sha256" else 3}
    with pytest.raises(SystemExit):
        run(tmp_path, ["s1"], manifest=changed, retry_skipped=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_retry_and_cleanup_only_touch_requested_subjects(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(CONTRACT))
    np.savez(tmp_path / "s1.npz", X=np.zeros((1, 1536)), y=np.array([0]), epoch_idx=[0], onset_s=[0.0])
    for subject in ("s1", "foreign"):
        (tmp_path / f"{subject}.skipped").write_text("rejet")
        (tmp_path / f"{subject}.tmp.npz").write_bytes(b"partiel")
    run(tmp_path, ["s1"], retry_skipped=True)
    assert not (tmp_path / "s1.skipped").exists()
    assert not (tmp_path / "s1.tmp.npz").exists()
    assert (tmp_path / "foreign.skipped").read_text() == "rejet"
    assert (tmp_path / "foreign.tmp.npz").read_bytes() == b"partiel"


@pytest.mark.parametrize("filename", ["s1.npz", "s1.skipped", "s1.tmp.npz"])
def test_missing_manifest_never_assigns_provenance_or_deletes_files(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"origine inconnue")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit, match="provenance inconnue"):
        run(tmp_path, ["s1"], retry_skipped=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_incomplete_expected_contract_is_refused_before_import(tmp_path):
    out = tmp_path / "import"
    with pytest.raises(ValueError, match="Contrat attendu incomplet"):
        import_embeddings([], out, CONTRACT | {"checkpoint_sha256": None})
    assert not out.exists()


@pytest.mark.parametrize("option", ["chunk_size", "batch_size", "num_threads", "num_workers"])
def test_invalid_chunk_settings_do_not_create_output(tmp_path, option):
    out = tmp_path / "cache"
    with pytest.raises(ValueError):
        run(out, ["s1"], **{option: -1})
    assert not out.exists()


@pytest.mark.parametrize("embedding_error", [False, True])
def test_prefetch_overlaps_embeddings_and_cleans_up_on_error(tmp_path, monkeypatch, embedding_error):
    import run_baseline as pipeline

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    embedding_started, second_downloaded = threading.Event(), threading.Event()
    raw_counts = []

    def fetch(subject, raw):
        if subject == "s2":
            assert embedding_started.wait(5), "Le téléchargement doit chevaucher les embeddings du premier lot"
        (raw / subject).mkdir()
        raw_counts.append(len(list(scratch.glob("*/raw"))))
        if subject == "s2":
            second_downloaded.set()

    def extract(raw, processed, num_threads):
        processed.mkdir()
        (processed / "subject").write_text(next(raw.iterdir()).name)

    def dataset(processed):
        return SimpleNamespace(subject=(processed / "subject").read_text())

    def embed(encoders, data, *args):
        if data.subject == "s1":
            embedding_started.set()
            assert second_downloaded.wait(5), "Le lot suivant doit arriver pendant les embeddings"
            if embedding_error:
                raise RuntimeError("échec embeddings")
        return np.zeros((1, 1536), dtype=np.float32), np.array([0])

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr(pipeline, "PhysioNet2018EpochDataset", dataset)
    monkeypatch.setattr(pipeline, "embed_dataset", embed)
    monkeypatch.setattr(pipeline, "epoch_groups", lambda data: (np.array([data.subject]), np.array([0])))
    monkeypatch.setattr(pipeline, "epoch_onsets", lambda *args: np.array([0.0]))
    out = tmp_path / "cache"

    def execute():
        run(out, ["s1", "s2", "s3"], fetch_subject=fetch, chunk_size=1)

    if embedding_error:
        with pytest.raises(RuntimeError, match="échec embeddings"):
            execute()
        assert not list(out.glob("*.npz"))
    else:
        execute()
        assert sorted(p.stem for p in out.glob("*.npz")) == ["s1", "s2", "s3"]
        assert max(raw_counts) == 2
    assert list(scratch.iterdir()) == []
