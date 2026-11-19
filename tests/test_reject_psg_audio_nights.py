"""Exclusions PSG-Audio du plan du 19-11 : motifs de données seulement, rejets motivés à l'extraction."""

import json

from scripts import reject_psg_audio_nights as reject


def test_only_data_failures_become_rejections(tmp_path):
    download, out = tmp_path / "dl", tmp_path / "emb"
    download.mkdir(), out.mkdir()
    failed = {
        "a": "a[007].edf : tranche non contiguë (03:29:38 au lieu de 03:18:41)",
        "b": "HTTP Error 504: Gateway Time-out",  # réseau : reprise, pas exclusion
        "c": "c[002].edf : voie V0, pmin b'-100' au lieu de b'0'",
    }
    (download / "download_log.json").write_text(json.dumps({"done": {"d": {}}, "failed": failed}))
    checks = {"d": {"compatible": False, "reason": "d[003].edf : voie V1, samples"}, "e": {"compatible": True}}
    (download / "source_header_check.json").write_text(json.dumps(checks))
    for suffix in (".npz", ".json"):
        (out / f"d{suffix}").write_text("x")
    found = reject.exclusions(download)
    assert sorted(found) == ["a", "c", "d"] and "non contiguë" in found["a"]
    reject.apply(found, out)
    assert not (out / "d.npz").exists() and not (out / "d.json").exists()
    assert (out / "a.skipped").read_text(encoding="utf-8").startswith("enregistrement exclu")
    assert not (out / "b.skipped").exists()
