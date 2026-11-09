"""Télécharge un sous-ensemble du PhysioNet/CinC Challenge 2018 (PSG + labels
de stades/arousals), sujets d'entraînement uniquement (le jeu de test n'a pas
de -arousal.mat, donc pas de labels).

Le jeu complet fait 994 sujets / ~130 Go ; utiliser --limit pour ne récupérer
qu'un sous-ensemble. Range chaque sujet dans <out>/<id>/ (3 fichiers), la
structure attendue par src/data/extract_physionet2018.py et run_baseline.py.
Pour 100+ sujets, préférer run_baseline.py --physionet-subjects, qui
télécharge au fil de l'eau sans tout garder sur disque.

Sur un réseau d'entreprise : si le téléchargement Python est bloqué (proxy
géré par Windows/le navigateur mais pas par Python), chaque sujet ne
nécessite que 3 fichiers, téléchargeables au navigateur depuis
https://physionet.org/files/challenge-2018/1.0.0/training/<id>/<id>.hea
(+ .mat + -arousal.mat), à ranger dans data/raw/physionet2018/training/<id>/.
"""

import argparse
import http.client
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.subject_ids import validate_subject_ids

PHYSIONET_2018_URL = "https://physionet.org/files/challenge-2018/1.0.0"
DOWNLOAD_TIMEOUT_SECONDS = 120  # sans réponse du serveur pendant ce délai : nouvel essai


def copy_complete(response, out) -> None:
    """Copie la réponse et vérifie sa taille annoncée : une fin de flux prématurée
    lève une erreur au lieu de passer pour un fichier complet."""
    written = 0
    while block := response.read(1 << 20):
        out.write(block)
        written += len(block)
    headers = getattr(response, "headers", None)
    announced = headers.get("Content-Length") if headers is not None else None
    if announced is not None and written != int(announced):
        raise OSError(f"Transfert incomplet : {written} octets reçus sur {announced}")


def list_subjects(split: str = "training") -> list[str]:
    """Identifiants des sujets (ex. "tr03-0005"), d'après le fichier RECORDS."""
    with urllib.request.urlopen(f"{PHYSIONET_2018_URL}/{split}/RECORDS", timeout=60) as response:
        return [line.strip().strip("/") for line in response.read().decode().split() if line.strip()]


def download_subject(subject_id: str, dest_dir: str | Path, split: str = "training", retries: int = 3) -> Path:
    """Télécharge les 3 fichiers d'un sujet (.hea, .mat, -arousal.mat) en HTTPS
    direct vers dest_dir/<subject_id>/, sans passer par wfdb (-arousal.mat est
    un fichier MATLAB, pas un enregistrement WFDB). Les fichiers déjà présents
    ne sont pas retéléchargés. Écriture dans un `.part` renommé à la fin : un
    transfert interrompu ou tronqué n'est jamais pris pour un fichier complet."""
    validate_subject_ids([subject_id])
    if retries <= 0:
        raise ValueError("retries doit être positif")
    if split not in ("training", "test"):
        raise ValueError("split doit être training ou test")
    subject_dir = Path(dest_dir) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{subject_id}.hea", f"{subject_id}.mat", f"{subject_id}-arousal.mat"):
        dest = subject_dir / name
        partial = subject_dir / f"{name}.part"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for attempt in range(retries):
            try:
                with (
                    urllib.request.urlopen(
                        f"{PHYSIONET_2018_URL}/{split}/{subject_id}/{name}", timeout=DOWNLOAD_TIMEOUT_SECONDS
                    ) as response,
                    open(partial, "wb") as out,
                ):
                    copy_complete(response, out)
                os.replace(partial, dest)
                break
            except (OSError, http.client.HTTPException):
                partial.unlink(missing_ok=True)
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
    return subject_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Répertoire de sortie (un sous-dossier par sujet)")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de sujets à télécharger")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit doit être positif")

    subjects = list_subjects("training")[: args.limit]
    print(f"{len(subjects)} sujet(s) à télécharger vers {args.out}")
    for i, subject in enumerate(subjects, 1):
        download_subject(subject, args.out)
        print(f"[{i}/{len(subjects)}] {subject}")


if __name__ == "__main__":
    main()
