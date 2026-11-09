"""Vérifie un NPZ d'époques OSF, sans charger de modèle ni écrire de signaux.

Champs requis : epochs (n, canaux, 1920), channels (chaînes), sample_rate
(scalaire). --mapping accepte un JSON {nom OSF: nom source}. Le contrôle
porte sur le format ; les unités et l'équivalence physiologique restent à valider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.osf_input import prepare_osf_epochs


def check_archive(path: Path, mapping: dict | None = None) -> dict:
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Archive NPZ attendue")
    with archive as data:
        missing = {"epochs", "channels", "sample_rate"} - set(data.files)
        if missing:
            raise ValueError(f"Champs NPZ absents : {', '.join(sorted(missing))}")
        names, rate = data["channels"], data["sample_rate"]
        if names.ndim != 1 or names.dtype.kind not in "US":
            raise ValueError("channels doit être un vecteur de chaînes Unicode")
        if names.dtype.kind == "S":
            raise ValueError("channels doit être encodé en Unicode, pas en bytes")
        if rate.shape != ():
            raise ValueError("sample_rate doit être un scalaire")
        _, metadata = prepare_osf_epochs(data["epochs"], names.tolist(), rate.item(), mapping)
    return {"source": str(path), "source_sha256": file_sha256(path), "format_valid": True, **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--mapping", type=Path, help="Correspondance explicite nom OSF -> nom source")
    parser.add_argument("--report", type=Path, help="Écrit uniquement le rapport JSON de contrôle")
    args = parser.parse_args()
    if args.report is not None and args.report.resolve() in {
        args.archive.resolve(),
        *([args.mapping.resolve()] if args.mapping else []),
    }:
        parser.error("Le rapport ne peut pas remplacer une entrée")
    try:
        mapping = json.loads(args.mapping.read_text()) if args.mapping else None
        report = check_archive(args.archive, mapping)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
