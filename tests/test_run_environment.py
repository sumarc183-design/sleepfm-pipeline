"""Le commit annoncé par un run doit contenir le code exécuté."""

import subprocess

import pytest

from src.models.probe_export import run_environment


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "suivi.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repository_is_not_modified(repo):
    env = run_environment(root=repo)
    assert env["code_modified"] is False
    assert env["untracked_code"] == []


def test_untracked_script_counts_as_modified(repo):
    (repo / "scripts" / "nouveau.py").write_text("y = 2\n")
    env = run_environment(root=repo)
    assert env["code_modified"] is True
    assert env["untracked_code"] == ["scripts/nouveau.py"]


def test_untracked_non_code_file_is_ignored(repo):
    (repo / "notes.md").write_text("brouillon\n")
    (repo / "scripts" / "sortie.json").write_text("{}\n")
    assert run_environment(root=repo)["code_modified"] is False


def test_tracked_modification_still_detected(repo):
    (repo / "scripts" / "suivi.py").write_text("x = 3\n")
    assert run_environment(root=repo)["code_modified"] is True


def test_outside_git_is_unknown(tmp_path):
    env = run_environment(root=tmp_path / "absent")
    assert env["code_modified"] is None
    assert env["code_commit"] == "inconnu"
