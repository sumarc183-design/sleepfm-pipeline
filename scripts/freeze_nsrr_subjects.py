"""Liste figée des nuits NSRR pour la confirmation de l'IAH (plan validé le 13 novembre 2026).

Règle du plan : toutes les nuits qui ont un EDF et des annotations XML NSRR. Le seuil d'au moins
1 h de sommeil s'applique à l'évaluation (`evaluate_nsrr_ahi.py`), qui publie les nuits écartées.
Ici, aucune donnée n'est lue : seuls les noms de fichiers comptent. Les fichiers sans partenaire
(EDF sans XML, XML sans EDF) sont publiés dans la liste. Une liste existante n'est jamais écrasée.
Les EDF MESA ne tenant pas sur le disque, `--edf-dir` accepte aussi un index texte de leurs noms.

    python scripts/freeze_nsrr_subjects.py --cohort mesa --edf-dir mesa/polysomnography/edfs \\
        --annotations-dir mesa/polysomnography/annotations-events-nsrr \\
        --out configs/nsrr_mesa_subjects.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import write_json_atomic  # noqa: E402

# Noms de la documentation NSRR, à vérifier sur les premiers fichiers. SHHS : visite 1 seulement.
SUBJECT_PATTERNS = {"mesa": r"mesa-sleep-\d{4}", "shhs1": r"shhs1-\d{6}"}
XML_SUFFIX = "-nsrr.xml"


def file_names(source: Path) -> list[str]:
    """Noms des fichiers d'un dossier, ou d'un index texte (un nom par ligne, liste du site NSRR)."""
    if source.is_dir():
        return sorted(path.name for path in source.iterdir())
    return sorted({Path(line.strip()).name for line in source.read_text().splitlines() if line.strip()})


def subjects_in(source: Path, cohort: str, suffix: str) -> tuple[set[str], list[str]]:
    """Sujets reconnus et fichiers ignorés (nom inattendu) pour un suffixe donné."""
    pattern = re.compile(SUBJECT_PATTERNS[cohort])
    found, ignored = set(), []
    for name in file_names(source):
        if not name.endswith(suffix):
            continue
        subject = name[: -len(suffix)]
        if pattern.fullmatch(subject):
            found.add(subject)
        else:
            ignored.append(name)
    return found, ignored


def freeze(cohort: str, edf_dir: Path, annotations_dir: Path) -> dict:
    edfs, edf_ignored = subjects_in(edf_dir, cohort, ".edf")
    xmls, xml_ignored = subjects_in(annotations_dir, cohort, XML_SUFFIX)
    subjects = sorted(edfs & xmls)
    if not subjects:
        raise SystemExit(f"{cohort} : aucune nuit avec EDF et XML (noms attendus {SUBJECT_PATTERNS[cohort]})")
    return {
        "cohort": cohort,
        "subjects": subjects,
        "rule": "EDF et annotations XML NSRR présents ; 1 h de sommeil vérifiée à l'évaluation",
        "plan": "docs/plans/Plan confirmation IAH NSRR SleepFM OSF 13-11.md",
        "edf_without_xml": sorted(edfs - xmls),
        "xml_without_edf": sorted(xmls - edfs),
        "ignored_files": edf_ignored + xml_ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=sorted(SUBJECT_PATTERNS), required=True)
    parser.add_argument("--edf-dir", type=Path, required=True, help="dossier des EDF ou index texte de leurs noms")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default="", help="origine des noms (dossier local, index public NSRR daté)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} existe déjà : une liste figée ne se réécrit pas")
    payload = freeze(args.cohort, args.edf_dir, args.annotations_dir) | {"source": args.source}
    write_json_atomic(args.out, payload)
    print(
        f"{len(payload['subjects'])} nuit(s) retenue(s) ; EDF sans XML : {len(payload['edf_without_xml'])} ; "
        f"XML sans EDF : {len(payload['xml_without_edf'])} ; fichiers ignorés : {len(payload['ignored_files'])}"
    )
    print(json.dumps({k: v[:5] for k, v in payload.items() if isinstance(v, list)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
