"""Téléchargement PSG-Audio sans réseau : reprise sur coupure, ordre des octets en parallèle, .rml vérifié d'abord."""

import hashlib
import http.client
import random
import time

import pytest

from scripts import download_psg_audio as dl

RATES = [2, 4, 48]  # deux voies gardées, une voie « audio » retirée (seuil abaissé pour le test)


def edf(start: str, n_records: int, seed: int) -> bytes:
    """EDF synthétique : 3 voies, enregistrements d'une seconde."""
    n = len(RATES)
    fixed = (
        f"{'0':<8}{'X':<80}{'Y':<80}{'01.01.20':<8}{start:<8}{256 * (1 + n):<8}{'':<44}{n_records:<8}{'1':<8}{n:<4}"
    ).encode()
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    values = {0: [f"V{i}" for i in range(n)], 8: [str(r) for r in RATES]}
    body = b"".join(
        b"".join(f"{values.get(k, ['0'] * n)[i]:<{w}}".encode() for i in range(n)) for k, w in enumerate(widths)
    )
    data = random.Random(seed).randbytes(2 * sum(RATES) * n_records)
    return fixed + body + data


@pytest.fixture
def server(monkeypatch):
    files = {"s1": edf("22.00.00", 7, 1), "s2": edf("22.00.07", 5, 2), "rml": b"<rml/>"}
    calls = []
    jitter = random.Random(0)

    def fetch(file_id, ranges=None, retries=5):
        calls.append(file_id)
        if ranges is None:
            return files[file_id]
        time.sleep(jitter.random() / 200)  # réponses dans le désordre entre fils
        return [files[file_id][a : b + 1] for a, b in ranges]

    monkeypatch.setattr(dl, "fetch", fetch)
    monkeypatch.setattr(dl, "AUDIO_MIN_RATE", 10.0)
    monkeypatch.setattr(dl, "RANGES_PER_REQUEST", 2)  # plusieurs requêtes par tranche
    monkeypatch.setattr(
        dl,
        "patient_segments",
        lambda _: [{"id": "s1", "fileName": "p[001].edf"}, {"id": "s2", "fileName": "p[002].edf"}],
    )
    return files, calls


def expected_edf(files) -> bytes:
    kept = 2 * sum(RATES[:2])
    record = 2 * sum(RATES)
    out = b""
    for name, n in (("s1", 7), ("s2", 5)):
        raw = files[name]
        header = 256 * (1 + len(RATES))
        out += b"".join(raw[header + i * record : header + i * record + kept] for i in range(n))
    return out


@pytest.mark.parametrize("workers", [1, 4])
def test_parallel_download_writes_records_in_order(server, tmp_path, workers):
    files, _ = server
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    entry = dl.download_patient("p", rml, tmp_path, workers=workers)
    written = (tmp_path / "p.edf").read_bytes()
    header_bytes = 256 * (1 + 2)
    assert written[header_bytes:] == expected_edf(files)
    assert int(written[236:244]) == 12 and int(written[252:256]) == 2
    assert entry["edf_sha256"] == hashlib.sha256(written).hexdigest()
    assert entry["kept_signals"] == ["V0", "V1"] and entry["duration_s"] == 12.0
    assert (tmp_path / "p.rml").read_bytes() == files["rml"]
    assert not (tmp_path / "p.edf.part").exists()


def test_wrong_rml_fails_before_any_edf_request(server, tmp_path):
    _, calls = server
    with pytest.raises(RuntimeError, match="MD5 du .rml"):
        dl.download_patient("p", {"id": "rml", "md5": "0" * 32}, tmp_path)
    assert calls == ["rml"] and list(tmp_path.iterdir()) == []


def test_fetch_retries_after_incomplete_read(monkeypatch):
    attempts = []

    class Response:
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    def urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise http.client.IncompleteRead(b"par", 3)
        return Response()

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    assert dl.fetch("f") == b"ok" and len(attempts) == 2
