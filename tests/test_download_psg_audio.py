"""Téléchargement PSG-Audio sans réseau : reprise sur coupure, ordre des octets en parallèle, .rml vérifié d'abord."""

import hashlib
import http.client
import random
import time

import pytest

from scripts import download_psg_audio as dl

RATES = [2, 4, 48]  # deux voies gardées, une voie « audio » retirée (seuil abaissé pour le test)


def edf(start: str, n_records: int, seed: int, pmin: str = "0", rates: list[int] = RATES) -> bytes:
    """EDF synthétique : 3 voies, enregistrements d'une seconde."""
    n = len(rates)
    fixed = (
        f"{'0':<8}{'X':<80}{'Y':<80}{'01.01.20':<8}{start:<8}{256 * (1 + n):<8}{'':<44}{n_records:<8}{'1':<8}{n:<4}"
    ).encode()
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    values = {0: [f"V{i}" for i in range(n)], 3: [pmin] + ["0"] * (n - 1), 8: [str(r) for r in rates]}
    body = b"".join(
        b"".join(f"{values.get(k, ['0'] * n)[i]:<{w}}".encode() for i in range(n)) for k, w in enumerate(widths)
    )
    data = random.Random(seed).randbytes(2 * sum(rates) * n_records)
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


def test_uppercase_md5_from_repository_is_accepted(server, tmp_path):
    """93 des 287 .rml du dépôt ont un MD5 annoncé en majuscules (ex. 00001579)."""
    files, _ = server
    md5 = hashlib.md5(files["rml"]).hexdigest()
    entry = dl.download_patient("p", {"id": "rml", "md5": md5.upper()}, tmp_path)
    assert entry["rml_md5"] == md5 and (tmp_path / "p.rml").exists()


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

        def __init__(self):
            self.parts = [b"ok", b""]

        def read1(self, n):
            return self.parts.pop(0)

    def urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise http.client.IncompleteRead(b"par", 3)
        return Response()

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    assert dl.fetch("f") == b"ok" and len(attempts) == 2


def http_error(code, headers=None):
    return dl.urllib.error.HTTPError("u", code, "", headers or {}, None)


def test_throttling_waits_longer_than_a_cut():
    assert dl.retry_delay(http.client.IncompleteRead(b"", 1), 0) == 10
    assert [dl.retry_delay(http_error(429), a) for a in range(6)] == [60, 120, 240, 480, 900, 900]
    assert dl.retry_delay(http_error(504), 1) == 120
    assert dl.retry_delay(http_error(429, {"Retry-After": "30"}), 3) == 30
    assert dl.retry_delay(http_error(404), 0) == 10  # pas une limitation : attente courte


def test_trickling_response_is_abandoned_at_the_deadline(monkeypatch):
    clock = iter(range(0, 10_000, 400))  # chaque morceau arrive 400 s après le précédent
    monkeypatch.setattr(dl.time, "monotonic", lambda: next(clock))

    class Trickle:
        def read1(self, n):
            return b"x"

    with pytest.raises(TimeoutError, match="trop lente"):
        dl.read_with_deadline(Trickle(), deadline=dl.time.monotonic() + dl.REQUEST_DEADLINE)
    assert isinstance(TimeoutError(), OSError)  # donc réessayée par fetch


def test_budget_uses_the_exact_size_before_writing(server, tmp_path):
    """Audit A4 : taille exacte connue après les en-têtes, comparée avant tout octet écrit."""
    files, _ = server
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    size = 256 * 3 + 12 * 2 * sum(RATES[:2])
    with pytest.raises(dl.BudgetExceeded):
        dl.download_patient("p", rml, tmp_path, max_bytes=size - 1)
    assert list(tmp_path.iterdir()) == []
    dl.download_patient("p", rml, tmp_path, max_bytes=size)
    assert (tmp_path / "p.edf").stat().st_size == size


def test_frozen_order_comes_from_the_approved_list():
    approved = [f"s{i:03d}" for i in range(20)]
    forward = dl.frozen_order(approved, set(approved) | {"nouveau"})  # sujet ajouté au dépôt : ignoré
    assert sorted(forward) == approved
    assert dl.frozen_order(approved, set(approved), reverse=True, limit=5) == forward[::-1][:5]
    with pytest.raises(SystemExit, match="Index ScienceDB modifié"):
        dl.frozen_order(approved, set(approved[1:]))


def test_finished_pair_is_rechecked_on_resume(server, tmp_path):
    files, _ = server
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    entry = dl.download_patient("p", rml, tmp_path)
    assert dl.intact(tmp_path, "p", entry) is None
    edf = tmp_path / "p.edf"
    edf.write_bytes(edf.read_bytes()[:-1] + b"\x00")
    assert "SHA-256" in dl.intact(tmp_path, "p", entry)
    (tmp_path / "p.rml").unlink()
    assert dl.intact(tmp_path, "p", entry) == "fichier absent"


def test_normal_response_is_read_whole():
    class Body:
        parts = [b"ab", b"cd", b""]

        def read1(self, n):
            return self.parts.pop(0)

    assert dl.read_with_deadline(Body(), deadline=float("inf")) == b"abcd"


def test_deadline_is_checked_even_if_bytes_trickle(monkeypatch):
    """Régression du 19-11 : `read(n)` attendait n octets et ne rendait jamais la main."""
    clock = iter(range(0, 10_000, 400))
    monkeypatch.setattr(dl.time, "monotonic", lambda: next(clock))

    class Trickle:
        def read(self, n):  # ne doit plus être appelée
            raise AssertionError("read(n) bloquerait jusqu'à n octets")

        def read1(self, n):
            return b"x"

    with pytest.raises(TimeoutError):
        dl.read_with_deadline(Trickle(), deadline=dl.time.monotonic() + dl.REQUEST_DEADLINE)


@pytest.mark.parametrize(
    ("second", "field"),
    [
        (edf("22.00.07", 5, 2, pmin="-100"), "pmin"),  # calibre différent sur une voie gardée
        (edf("22.00.07", 5, 2, rates=[4, 2, 48]), "samples"),  # fréquences permutées, même taille
    ],
)
def test_incompatible_segment_is_refused_before_any_edf_byte(server, tmp_path, second, field):
    """Audit A1 du 19-11 : tranches de mêmes libellés mais de calibrations différentes."""
    files, _ = server
    files["s2"] = second
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    with pytest.raises(RuntimeError, match=rf"p\[002\]\.edf : voie V[01], {field}"):
        dl.download_patient("p", rml, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_audio_voice_may_differ_and_source_headers_are_traced(server, tmp_path):
    files, _ = server
    files["s2"] = edf("22.00.07", 5, 2, rates=[2, 4, 96])  # seule la voie audio retirée change
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    entry = dl.download_patient("p", rml, tmp_path)
    assert entry["duration_s"] == 12.0
    headers = [hashlib.sha256(files[s][: 256 * 4]).hexdigest() for s in ("s1", "s2")]
    assert entry["source_headers_sha256"] == headers


def test_existing_edf_is_qualified_against_source_headers(server, tmp_path, monkeypatch):
    from scripts import check_psg_audio_headers as check

    files, _ = server
    rml = {"id": "rml", "md5": hashlib.md5(files["rml"]).hexdigest()}
    dl.download_patient("p", rml, tmp_path)
    monkeypatch.setattr(check, "source_header", lambda file_id: files[file_id][: 256 * 4])
    assert check.check_patient("p", tmp_path / "p.edf")["compatible"]
    files["s2"] = edf("22.00.07", 5, 2, pmin="-100")  # tranche source recalibrée après coup
    result = check.check_patient("p", tmp_path / "p.edf")
    assert not result["compatible"] and "pmin" in result["reason"]


def test_end_of_response_after_the_deadline_is_refused(monkeypatch):
    """Audit A4 : la fin de réponse arrivée après l'échéance n'est plus acceptée."""
    clock = iter([0, 1000])  # morceau à 0 s, fin de réponse à 1 000 s
    monkeypatch.setattr(dl.time, "monotonic", lambda: next(clock))

    class Late:
        parts = [b"ab", b""]

        def read1(self, n):
            return self.parts.pop(0)

    with pytest.raises(TimeoutError):
        dl.read_with_deadline(Late(), deadline=900)
