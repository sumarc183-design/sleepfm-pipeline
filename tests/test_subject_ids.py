"""Les entrées invalides sont refusées avant réseau, poids ou écritures."""

import json
import sys

import pytest

import run_baseline
from scripts.download_physionet2018 import download_subject
from src.data.subject_ids import load_subjects_file, validate_subject_ids


@pytest.mark.parametrize(
    "subjects", [[], "s1", None, [1], [""], ["s1", "s1"], ["../s1"], ["/s1"], ["a/b"], [".."], ["a?b"]]
)
def test_bad_subjects_are_refused_without_files(tmp_path, subjects):
    with pytest.raises(ValueError):
        validate_subject_ids(subjects)
    with pytest.raises(ValueError):
        run_baseline.embed_subjects_in_chunks(subjects, None, None, tmp_path / "out", 1, 1, 1, "cpu")
    assert list(tmp_path.iterdir()) == []


def test_valid_subjects_and_file_metadata_are_preserved(tmp_path):
    subjects = ["tr03-0005", "synthetic_0001"]
    path = tmp_path / "subjects.json"
    path.write_text(json.dumps({"subjects": subjects, "description": "protocole"}))
    assert load_subjects_file(path) == subjects
    assert validate_subject_ids([], allow_empty=True) == []


@pytest.mark.parametrize("payload", [{}, [], {"subjects": []}, {"subjects": "s1"}, {"subjects": ["s1", "s1"]}])
def test_invalid_subject_file_is_refused_before_checkpoint(tmp_path, monkeypatch, payload):
    path = tmp_path / "subjects.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        ["baseline", "--subjects-file", str(path), "--embed-only", "--embeddings-dir", str(tmp_path / "out")],
    )
    monkeypatch.setattr(run_baseline, "build_random_checkpoint", lambda *args: pytest.fail("Poids non nécessaires"))
    with pytest.raises(SystemExit) as error:
        run_baseline.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--chunk-size", "0"),
        ("--chunk-size", "-1"),
        ("--batch-size", "0"),
        ("--num-threads", "0"),
        ("--num-workers", "-1"),
        ("--physionet-subjects", "0"),
        ("--test-fraction", "nan"),
    ],
)
def test_invalid_cli_settings_stop_before_work(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["baseline", option, value])
    monkeypatch.setattr(run_baseline, "build_random_checkpoint", lambda *args: pytest.fail("Poids non nécessaires"))
    with pytest.raises(SystemExit) as error:
        run_baseline.main()
    assert error.value.code == 2


@pytest.mark.parametrize("subject,retries", [("../outside", 3), ("s1", 0)])
def test_downloader_rejects_bad_input_before_network_or_output(tmp_path, monkeypatch, subject, retries):
    from scripts import download_physionet2018 as downloader

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Réseau non nécessaire"))
    with pytest.raises(ValueError):
        download_subject(subject, tmp_path / "out", retries=retries)
    assert list(tmp_path.iterdir()) == []


def test_download_cli_rejects_invalid_limit_before_listing(tmp_path, monkeypatch):
    from scripts import download_physionet2018 as downloader

    monkeypatch.setattr(sys, "argv", ["download", "--out", str(tmp_path / "out"), "--limit", "-1"])
    monkeypatch.setattr(downloader, "list_subjects", lambda *args: pytest.fail("Réseau non nécessaire"))
    with pytest.raises(SystemExit) as error:
        downloader.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
