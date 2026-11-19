"""Qualifie les EDF PSG-Audio déjà assemblés (audit A1 du 19-11) d'après les en-têtes des tranches source.

Pour chaque patient terminé du journal : en-tête de chaque tranche relu sur ScienceDB (une requête
de quelques Ko), voies gardées comparées à la première tranche (`check_compatible`), puis en-tête
de la première tranche comparé à celui de l'EDF local pour les voies gardées. Résultat par patient
dans <out>/source_header_check.json, repris à la relance ; aucun EDF n'est modifié.

    python scripts/check_psg_audio_headers.py --out data/raw/psg_audio
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import download_psg_audio as dl

HEADER_PROBE = 256 * 65  # en-tête complet jusqu'à 64 voies


def source_header(file_id: str) -> bytes:
    raw = dl.fetch(file_id, [(0, HEADER_PROBE - 1)])[0]
    return raw[: int(raw[184:192])]


def check_patient(patient: str, local_edf: Path) -> dict:
    infos = []
    for seg in dl.patient_segments(f"/V3/APNEA_EDF/{patient}"):
        raw = source_header(seg["id"])
        infos.append(dl.parse_header(raw) | {"name": seg["fileName"], "sha256": hashlib.sha256(raw).hexdigest()})
    with local_edf.open("rb") as handle:
        fixed = handle.read(256)
        local = dl.parse_header(fixed + handle.read(256 * int(fixed[252:256])))
    keep = local["n"]
    result = {"segments": [i["name"] for i in infos], "source_headers_sha256": [i["sha256"] for i in infos]}
    try:
        for name in dl.COMPARED_FIELDS:  # EDF local = voies gardées de la première tranche
            if [v.strip() for v in local["fields"][name]] != [v.strip() for v in infos[0]["fields"][name][:keep]]:
                raise RuntimeError(f"EDF local : {name} différent de la première tranche")
        for info in infos[1:]:
            dl.check_compatible(infos[0], info, keep)
    except RuntimeError as exc:
        return result | {"compatible": False, "reason": str(exc)}
    return result | {"compatible": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = json.loads((args.out / "download_log.json").read_text(encoding="utf-8"))
    report_path = args.out / "source_header_check.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    todo = [p for p in sorted(log["done"]) if p not in report]
    print(f"{len(report)} déjà contrôlés, {len(todo)} à contrôler", flush=True)
    for index, patient in enumerate(todo, 1):
        try:
            report[patient] = check_patient(patient, args.out / f"{patient}.edf")
        except Exception as exc:  # réseau : laissé à la relance
            print(f"[{index}/{len(todo)}] {patient} : non contrôlé ({exc})", flush=True)
            continue
        state = "compatible" if report[patient]["compatible"] else f"INCOMPATIBLE : {report[patient]['reason']}"
        print(f"[{index}/{len(todo)}] {patient} : {len(report[patient]['segments'])} tranches, {state}", flush=True)
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(report_path)
    bad = sorted(p for p, r in report.items() if not r["compatible"])
    print(f"Fin : {len(report)} contrôlés sur {len(log['done'])}, {len(bad)} incompatible(s) {bad}", flush=True)


if __name__ == "__main__":
    main()
