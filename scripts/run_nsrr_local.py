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
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

WEBSITE = "https://sleepdata.org"
GEM_VERSION = "8-0-0"  # adresse de téléchargement de la gem nsrr 8.0.0
DATASET = {"mesa": "mesa", "shhs1": "shhs"}
EDF_PATH = {"mesa": "polysomnography/edfs", "shhs1": "polysomnography/edfs/shhs1"}
XML_PATH = {"mesa": "polysomnography/annotations-events-nsrr", "shhs1": "polysomnography/annotations-events-nsrr/shhs1"}
HEADERS = {"User-Agent": "sleepfm-pipeline research download"}
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
    with urllib.request.urlopen(request, timeout=120) as response:
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
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=600) as response:
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


def download_lot(cohort: str, lot: list[str], index: dict, folder: Path, token: str) -> list[str]:
    """EDF du lot dans `folder` ; renvoie les sujets non obtenus (absents de NSRR ou en échec)."""
    missing = []
    for subject in lot:
        entry = index.get(f"{subject}.edf")
        if entry is None:
            missing.append(subject)
            continue
        try:
            download_file(DATASET[cohort], entry, folder / f"{subject}.edf", token)
        except RuntimeError as exc:
            if isinstance(exc, AccessError):
                raise
            print(f"  {subject} non obtenu : {redact(exc, token)}", flush=True)
            missing.append(subject)
    return missing


def pending(subjects: list[str], out: Path) -> list[str]:
    return [s for s in subjects if not (out / f"{s}.npz").exists() and not (out / f"{s}.skipped").exists()]


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
        result = subprocess.run(command, capture_output=True, text=True)  # code 1 tant que la liste n'est pas finie
        for line in result.stdout.splitlines():
            if "EDF absent" not in line:
                print(line, flush=True)
        if result.returncode not in (0, 1):
            raise SystemExit(f"embed_nsrr a échoué :\n{result.stderr[-2000:]}")
        print(
            f"--- lot {start // LOT + 1} : {len(subjects) - len(pending(subjects, out))}/{len(subjects)} nuits",
            flush=True,
        )
    shutil.rmtree(edf_dir, ignore_errors=True)
    print(f"Non obtenus : {unavailable}")


if __name__ == "__main__":
    main()
