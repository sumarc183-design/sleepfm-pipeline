"""Embeddings NSRR sur ce PC, par lots, sans cloud : variante locale du notebook Kaggle (même logique).

Pour chaque lot de la liste figée : EDF téléchargés depuis sleepdata.org (adresse de la gem officielle,
taille et MD5 de l'API NSRR vérifiés, seconde tentative en cas d'écart), embeddings par
`scripts/embed_nsrr.py` (reprise automatique), puis EDF du lot supprimés. Les annotations XML,
légères, se téléchargent à part (`--annotations`) pour l'évaluation. Aucun score n'est calculé ici.
Le jeton personnel est lu dans `--token-file` (ou `NSRR_TOKEN`) et n'est jamais affiché.

    python scripts/run_nsrr_local.py --cohort mesa --limit 2           # essai court
    python scripts/run_nsrr_local.py --cohort mesa --annotations       # XML de toute la liste
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WEBSITE = "https://sleepdata.org"
GEM_VERSION = "8-0-0"  # adresse de téléchargement de la gem nsrr 8.0.0
DATASET = {"mesa": "mesa", "shhs1": "shhs"}
EDF_PATH = {"mesa": "polysomnography/edfs", "shhs1": "polysomnography/edfs/shhs1"}
XML_PATH = {"mesa": "polysomnography/annotations-events-nsrr", "shhs1": "polysomnography/annotations-events-nsrr/shhs1"}
HEADERS = {"User-Agent": "sleepfm-pipeline research download"}
# sleepdata.org n'envoie pas son certificat intermédiaire (Sectigo DV R36, vérifié le 20-11) : Windows
# le retrouve seul, pas Linux (Kaggle). Intermédiaire public joint, vérification complète conservée
# jusqu'à la racine Sectigo R46 du magasin système ou de certifi. SHA-256 de l'intermédiaire :
# 8C:54:C3:34:B6:6B:A4:E4:26:77:2A:F4:A3:F9:13:6C:19:A1:AE:C7:29:FD:B2:8C:53:5C:07:A5:A4:EF:22:E0
INTERMEDIATE = Path(__file__).resolve().parents[1] / "configs/certs/sectigo_dv_r36.pem"


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    context.load_verify_locations(INTERMEDIATE)
    return context


TLS = tls_context()
LOT = 10


class AccessError(RuntimeError):
    """Le serveur répond par une page HTML : accès à la cohorte non ouvert."""


def read_token(token_file: Path) -> str:
    token = os.environ.get("NSRR_TOKEN") or (
        token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""
    )
    if not token:
        raise SystemExit(f"Jeton NSRR absent : {token_file} ou variable NSRR_TOKEN")
    return token


def redact(text: str, token: str) -> str:
    return str(text).replace(token, "***")


def folder_index(dataset: str, path: str, token: str) -> dict[str, dict]:
    """{nom de fichier: {size, md5, full_path}} d'un dossier NSRR (API publique de la gem)."""
    query = urllib.parse.urlencode({"auth_token": token, "path": path})
    request = urllib.request.Request(f"{WEBSITE}/api/v1/datasets/{dataset}/files.json?{query}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120, context=TLS) as response:
        entries = json.load(response)
    return {
        e["file_name"]: {"size": int(e["file_size"]), "md5": e["file_checksum_md5"], "full_path": e["full_path"]}
        for e in entries
        if e.get("is_file")
    }


def download_file(dataset: str, entry: dict, dest: Path, token: str, attempts: int = 2) -> None:
    """Fichier complet vers `dest` (via `.part`), taille et MD5 vérifiés ; AccessError si page HTML."""
    url = f"{WEBSITE}/datasets/{dataset}/files/a/{token}/m/nsrr-gem-v{GEM_VERSION}/{entry['full_path']}"
    partial = dest.with_name(dest.name + ".part")
    problem = ""
    for _ in range(attempts):
        digest, size = hashlib.md5(), 0
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=600, context=TLS) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise AccessError(f"{entry['full_path']} : page HTML au lieu du fichier (accès non ouvert ?)")
                with partial.open("wb") as handle:
                    for chunk in iter(lambda: response.read(1 << 20), b""):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        except OSError as exc:
            problem = redact(exc, token)
            partial.unlink(missing_ok=True)
            time.sleep(5)
            continue
        if size == entry["size"] and digest.hexdigest() == entry["md5"]:
            os.replace(partial, dest)
            return
        problem = f"taille {size} / {entry['size']}, MD5 {digest.hexdigest()} / {entry['md5']}"
        partial.unlink(missing_ok=True)
    raise RuntimeError(f"{entry['full_path']} : {problem}")


def download_lot(cohort: str, lot: list[str], index: dict, folder: Path, token: str, workers: int = 1) -> list[str]:
    """EDF du lot dans `folder` ; renvoie les sujets non obtenus (absents de NSRR ou en échec), dans l'ordre.

    `workers` fichiers à la fois : NSRR limite le débit par connexion (environ 2,5 Mo/s mesurés sur
    Kaggle le 20-11), le GPU attendait les EDF les trois quarts du temps. Accès refusé : arrêt du lot.
    """

    def one(subject: str) -> str | None:
        entry = index.get(f"{subject}.edf")
        if entry is None:
            return subject
        try:
            download_file(DATASET[cohort], entry, folder / f"{subject}.edf", token)
        except RuntimeError as exc:
            if isinstance(exc, AccessError):
                raise
            print(f"  {subject} non obtenu : {redact(exc, token)}", flush=True)
            return subject
        return None

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return [s for s in pool.map(one, lot) if s is not None]


def pending(subjects: list[str], out: Path) -> list[str]:
    """Nuits à télécharger : ni NPZ accompagné de sa fiche, ni rejet motivé (audit A3)."""
    done = [s for s in subjects if (out / f"{s}.npz").exists() and (out / f"{s}.json").exists()]
    return [s for s in subjects if s not in set(done) and not (out / f"{s}.skipped").exists()]


def unfinished_run(out: Path, started: float) -> bool:
    """Vrai si l'extracteur a écrit, pendant ce lot, un run_status.json « liste pas encore finie ».

    `embed_nsrr` sort avec le code 1 dans ce cas normal, mais aussi pour une erreur bloquante
    (contrat différent, cache abîmé) avant d'écrire ce statut : seul le premier cas est accepté
    (audit A3). Le code de l'extracteur n'est pas modifié, car il fait partie du contrat des caches.
    """
    path = out / "run_status.json"
    if not path.exists() or path.stat().st_mtime < started:
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("complete") is False


def run_lot(command: list[str], out: Path) -> None:
    """Extraction d'un lot ; seul « liste pas encore finie » est accepté en plus du succès."""
    started = time.time() - 1  # marge pour la résolution des dates de fichier
    result = subprocess.run(command, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "EDF absent" not in line:
            print(line, flush=True)
    if result.returncode == 0 or (result.returncode == 1 and unfinished_run(out, started)):
        return
    raise SystemExit(f"embed_nsrr a échoué (code {result.returncode}) :\n{result.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=sorted(DATASET), required=True)
    parser.add_argument("--limit", type=int, default=None, help="nombre de nuits (essai court)")
    parser.add_argument(
        "--annotations", action="store_true", help="télécharge les XML de la liste figée, puis s'arrête"
    )
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".nsrr_token")
    parser.add_argument("--work", type=Path, default=Path("data/nsrr"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    token = read_token(args.token_file)
    subjects = json.loads(Path(f"configs/nsrr_{args.cohort}_subjects.json").read_text(encoding="utf-8"))["subjects"]
    dataset = DATASET[args.cohort]

    if args.annotations:
        folder = args.work / args.cohort / "annotations-events-nsrr"
        folder.mkdir(parents=True, exist_ok=True)
        index = folder_index(dataset, XML_PATH[args.cohort], token)
        todo = [s for s in subjects if not (folder / f"{s}-nsrr.xml").exists()]
        for i, subject in enumerate(todo, 1):
            download_file(dataset, index[f"{subject}-nsrr.xml"], folder / f"{subject}-nsrr.xml", token)
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} annotations", flush=True)
        return

    out = args.out or Path(f"embeddings/nsrr_{args.cohort}")
    edf_dir = args.work / args.cohort / "edf_lot"
    index = folder_index(dataset, EDF_PATH[args.cohort], token)
    todo = pending(subjects, out)[: args.limit] if args.limit else pending(subjects, out)
    command = [
        sys.executable, "-u", "scripts/embed_nsrr.py", "--cohort", args.cohort, "--edf-dir", str(edf_dir),
        "--subjects", f"configs/nsrr_{args.cohort}_subjects.json", "--out", str(out),
        "--checkpoint", "third_party/sleepfm/sleepfm/checkpoint/best.pt", "--respiratory-rule", "physionet_median_v2",
        "--osf-checkpoint", "third_party/osf/osf_backbone.pth",
    ]  # fmt: skip
    unavailable = []
    for start in range(0, len(todo), LOT):
        shutil.rmtree(edf_dir, ignore_errors=True)
        edf_dir.mkdir(parents=True)
        lot = todo[start : start + LOT]
        unavailable += download_lot(args.cohort, lot, index, edf_dir, token)
        run_lot(command, out)
        print(
            f"--- lot {start // LOT + 1} : {len(subjects) - len(pending(subjects, out))}/{len(subjects)} nuits",
            flush=True,
        )
    shutil.rmtree(edf_dir, ignore_errors=True)
    print(f"Non obtenus : {unavailable}")


if __name__ == "__main__":
    main()
