"""Téléchargement NSRR local sans réseau : MD5, seconde tentative, accès refusé, jeton jamais affiché."""

import hashlib
import io
import json

import pytest

from scripts import run_nsrr_local as nsrr

TOKEN = "12345-secretjeton"


class Response(io.BytesIO):
    def __init__(self, body: bytes, ctype: str = "application/octet-stream"):
        super().__init__(body)
        self.headers = {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def entry(body: bytes, name="mesa-sleep-0001.edf"):
    return {"size": len(body), "md5": hashlib.md5(body).hexdigest(), "full_path": f"polysomnography/edfs/{name}"}


def serve(monkeypatch, *bodies):
    calls = []

    def urlopen(request, timeout, context=None):
        calls.append(request.full_url)
        body = bodies[min(len(calls), len(bodies)) - 1]
        if isinstance(body, Exception):
            raise body
        return body

    monkeypatch.setattr(nsrr.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(nsrr.time, "sleep", lambda _: None)
    return calls


def test_download_verifies_md5_and_uses_gem_address(monkeypatch, tmp_path):
    body = b"EDF" * 1000
    calls = serve(monkeypatch, Response(body))
    nsrr.download_file("mesa", entry(body), tmp_path / "a.edf", TOKEN)
    assert (tmp_path / "a.edf").read_bytes() == body and not (tmp_path / "a.edf.part").exists()
    assert calls == [
        f"https://sleepdata.org/datasets/mesa/files/a/{TOKEN}/m/nsrr-gem-v8-0-0/polysomnography/edfs/mesa-sleep-0001.edf"
    ]


def test_corrupt_file_is_downloaded_again(monkeypatch, tmp_path):
    body = b"EDF" * 1000
    calls = serve(monkeypatch, Response(b"tronque"), Response(body))
    nsrr.download_file("mesa", entry(body), tmp_path / "a.edf", TOKEN)
    assert len(calls) == 2 and (tmp_path / "a.edf").read_bytes() == body


def test_two_failures_raise_without_token_or_file(monkeypatch, tmp_path):
    body = b"EDF" * 1000
    serve(monkeypatch, OSError(f"coupure https://x/a/{TOKEN}/"), Response(b"faux"))
    with pytest.raises(RuntimeError) as error:
        nsrr.download_file("mesa", entry(body), tmp_path / "a.edf", TOKEN)
    assert "MD5" in str(error.value) and TOKEN not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_html_page_means_access_not_granted_and_stops_the_lot(monkeypatch, tmp_path, capsys):
    serve(monkeypatch, Response(b"<!DOCTYPE html>", "text/html; charset=utf-8"))
    index = {"mesa-sleep-0001.edf": entry(b"x")}
    with pytest.raises(nsrr.AccessError, match="accès non ouvert"):
        nsrr.download_lot("mesa", ["mesa-sleep-0001"], index, tmp_path, TOKEN)
    assert TOKEN not in capsys.readouterr().out and list(tmp_path.iterdir()) == []


def test_lot_reports_subjects_absent_from_nsrr(monkeypatch, tmp_path):
    body = b"EDF" * 10
    serve(monkeypatch, Response(body))
    missing = nsrr.download_lot(
        "mesa", ["mesa-sleep-0001", "mesa-sleep-9999"], {"mesa-sleep-0001.edf": entry(body)}, tmp_path, TOKEN
    )
    assert missing == ["mesa-sleep-9999"] and (tmp_path / "mesa-sleep-0001.edf").exists()


def test_folder_index_keeps_files_only(monkeypatch):
    listing = [
        {"file_name": "shhs1", "is_file": False},
        {"file_name": "a.edf", "is_file": True, "file_size": 3, "file_checksum_md5": "m", "full_path": "p/a.edf"},
    ]
    serve(monkeypatch, Response(json.dumps(listing).encode()))
    assert nsrr.folder_index("shhs", "p", TOKEN) == {"a.edf": {"size": 3, "md5": "m", "full_path": "p/a.edf"}}


def test_pending_skips_done_and_rejected(tmp_path):
    (tmp_path / "a.npz").touch(), (tmp_path / "a.json").touch()  # NPZ et sa fiche
    (tmp_path / "b.skipped").touch()
    assert nsrr.pending(["a", "b", "c"], tmp_path) == ["c"]


def test_only_success_or_unfinished_list_is_accepted(monkeypatch, capsys, tmp_path):
    """Audit A3 : code 1 accepté seulement si l'extracteur vient d'écrire un statut incomplet."""
    import subprocess

    def fake(code, status=None):
        def run(*a, **k):
            if status is not None:
                (tmp_path / "run_status.json").write_text(json.dumps({"complete": status}))
            return subprocess.CompletedProcess([], code, "ligne\n[1/2] x : EDF absent\n", "contrat différent")

        return run

    monkeypatch.setattr(nsrr.subprocess, "run", fake(0))
    nsrr.run_lot(["embed"], tmp_path)
    monkeypatch.setattr(nsrr.subprocess, "run", fake(1, status=False))
    nsrr.run_lot(["embed"], tmp_path)
    assert "EDF absent" not in capsys.readouterr().out
    old = tmp_path / "run_status.json"
    nsrr.os.utime(old, (0, 0))  # statut d'un lot précédent : l'erreur de ce lot n'est pas masquée
    for case in (fake(1), fake(1, status=True), fake(2, status=False)):
        nsrr.os.utime(old, (0, 0))
        monkeypatch.setattr(nsrr.subprocess, "run", case)
        with pytest.raises(SystemExit, match="contrat différent"):
            nsrr.run_lot(["embed"], tmp_path)


def test_npz_without_its_record_is_still_pending(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"")
    (tmp_path / "b.npz").write_bytes(b"")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "c.skipped").write_text("EDF tronqué")
    assert nsrr.pending(["a", "b", "c", "d"], tmp_path) == ["a", "d"]


def test_tls_context_verifies_with_the_joined_intermediate():
    """sleepdata.org sans intermédiaire (20-11) : vérification conservée, intermédiaire public joint."""
    import hashlib
    import ssl

    assert nsrr.TLS.verify_mode == ssl.CERT_REQUIRED and nsrr.TLS.check_hostname
    der = ssl.PEM_cert_to_DER_cert(nsrr.INTERMEDIATE.read_text(encoding="ascii"))
    assert hashlib.sha256(der).hexdigest().upper() == (
        "8C54C334B66BA4E426772AF4A3F9136C19A1AEC729FDB28C535C07A5A4EF22E0"
    )
