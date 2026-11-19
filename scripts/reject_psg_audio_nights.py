"""Exclusions PSG-Audio fixées au plan du 19-11, écrites à l'extraction (`<sujet>.skipped` motivé).

Deux sources, sans lecture d'annotation ni de prédiction :
- journal de téléchargement : échecs dus aux données (tranches non contiguës, voies incompatibles).
  Les échecs réseau (délai, HTTP, coupure) ne sont pas des exclusions : ces nuits sont reprises ;
- contrôle des en-têtes source (`source_header_check.json`, audit A1) : nuits déjà assemblées dont
  les tranches sont incompatibles. Leur NPZ et sa fiche sont retirés, puisque la nuit est exclue.

Sans `--apply`, la liste est seulement affichée. Relancer ensuite `embed_nsrr.py` met à jour
`run_status.json`, qui compte ces rejets.

    python scripts/reject_psg_audio_nights.py --download data/raw/psg_audio --out embeddings/psgaudio --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Motifs d'échec qui relèvent des données (download_psg_audio.py), pas du réseau.
DATA_FAILURES = (
    "tranche non contiguë",
    "voies audio pas en fin de bloc",
    "voies différentes de la première tranche",
    "durée d'enregistrement",
    " : voie ",
    "aucune EDF",
)


def exclusions(download_dir: Path) -> dict[str, str]:
    log = json.loads((download_dir / "download_log.json").read_text(encoding="utf-8"))
    out = {
        subject: f"enregistrement exclu (plan du 19-11) : {reason}"
        for subject, reason in log["failed"].items()
        if subject not in log["done"] and any(marker in reason for marker in DATA_FAILURES)
    }
    check = download_dir / "source_header_check.json"
    if check.exists():
        for subject, result in json.loads(check.read_text(encoding="utf-8")).items():
            if not result["compatible"]:
                out[subject] = f"tranches incompatibles (plan du 19-11, audit A1) : {result['reason']}"
    return out


def apply(found: dict[str, str], out: Path) -> None:
    for subject, reason in sorted(found.items()):
        for suffix in (".npz", ".json"):
            (out / f"{subject}{suffix}").unlink(missing_ok=True)
        (out / f"{subject}.skipped").write_text(reason + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", type=Path, required=True, help="dossier du téléchargement PSG-Audio")
    parser.add_argument("--out", type=Path, required=True, help="dossier des embeddings PSG-Audio")
    parser.add_argument("--apply", action="store_true", help="écrit les rejets (sinon : liste seulement)")
    args = parser.parse_args()
    found = exclusions(args.download)
    for subject, reason in sorted(found.items()):
        print(f"{subject} : {reason}")
    if args.apply:
        apply(found, args.out)
    print(f"{len(found)} nuit(s) exclue(s){'' if args.apply else ' (rien écrit, ajouter --apply)'}")


if __name__ == "__main__":
    main()
