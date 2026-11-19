"""PSG-Audio (ScienceDB, CC BY 4.0) sans le son : une EDF allégée et le .rml des médecins par patient.

Contrôle externe validé le 13 novembre (plan dans Journal). Les EDF du dépôt contiennent deux voies
audio à 48 kHz (97 % du volume, 918 Go). Le serveur accepte les requêtes à plages multiples : pour
chaque seconde enregistrée, seuls les octets des voies non audio (préfixe contigu de chaque bloc)
sont téléchargés. Les tranches d'une heure d'un patient sont recollées en une seule EDF après
contrôle de leur continuité (heure de début de chaque tranche = fin de la précédente). Reprise
après arrêt : un patient terminé n'est jamais refait ; un patient en échec est noté et repris.

    python scripts/download_psg_audio.py --out data/raw/psg_audio [--limit 1]

Ordre de téléchargement aléatoire, figé par la graine 42 (voir `main`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cache_contract import write_json_atomic  # noqa: E402

DATASET = "778740145531650048"
LIST_URL = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DOWNLOAD_URL = "https://download.scidb.cn/download?fileId={}"
AUDIO_MIN_RATE = 1000.0  # voies à 48 kHz retirées
RANGES_PER_REQUEST = 300
WORKERS = 4  # requêtes simultanées : débit x3 environ (mesuré le 16 nov.)
HEADERS = {"User-Agent": "sleepfm-pipeline research download"}
# Serveur saturé ou limitant le débit (vu le 17 nov. au soir : 429 et 504 en série) : attendre bien plus
# longtemps qu'après une simple coupure, pour ne pas sauter les patients les uns après les autres.
THROTTLED = {429, 502, 503, 504}
RETRIES = 8
# Le délai d'une lecture réseau ne borne pas une requête dont le serveur distille quelques octets :
# blocage de plus de 5 h vu le 18 nov. à 03h13. Durée totale bornée (une requête prend d'ordinaire < 1 min).
SOCKET_TIMEOUT = 120
REQUEST_DEADLINE = 900.0
APPROVED = ROOT / "configs/psgaudio_subjects.json"  # univers de sélection figé (audit A4)
PASSES = 3  # passages successifs : les patients en échec sont repris, au plus PASSES fois au total


def list_folder(path: str) -> list[dict]:
    body = json.dumps({"dataSetId": DATASET, "version": "V3", "path": path, "lastFileName": None, "pageSize": 1000})
    request = urllib.request.Request(
        LIST_URL, data=body.encode(), headers=HEADERS | {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    if payload.get("code") != 20000:
        raise RuntimeError(f"Liste refusée pour {path} : {payload.get('messageEn')}")
    return [f for f in payload["data"] if f["version"] == "V3"]


def retry_delay(exc: Exception, attempt: int) -> float:
    """Secondes avant le prochain essai : Retry-After du serveur s'il est donné, sinon 1, 2, 4… min (15 au plus)."""
    if isinstance(exc, urllib.error.HTTPError) and exc.code in THROTTLED:
        after = (exc.headers or {}).get("Retry-After", "")
        if after.strip().isdigit():
            return min(float(after), 900.0)
        return min(60.0 * 2**attempt, 900.0)
    return 10.0 * (attempt + 1)


def read_with_deadline(response, deadline: float) -> bytes:
    """Corps lu par morceaux ; au-delà de l'échéance, abandon (réessayé par `fetch`).

    L'échéance porte sur la lecture du corps. La connexion et l'attente des en-têtes sont bornées
    séparément par SOCKET_TIMEOUT à chaque opération réseau.

    `read1` rend ce qui est déjà arrivé : `read(n)` attendrait n octets, et un serveur qui
    distille quelques octets par minute ne rendrait jamais la main (blocage de 7 h le 19-11).
    """
    chunks = []
    while chunk := response.read1(1 << 16):
        chunks.append(chunk)
        if time.monotonic() > deadline:
            raise TimeoutError(f"réponse trop lente (plus de {REQUEST_DEADLINE:.0f} s)")
    if time.monotonic() > deadline:  # fin de réponse arrivée après l'échéance (audit A4)
        raise TimeoutError(f"réponse trop lente (plus de {REQUEST_DEADLINE:.0f} s)")
    return b"".join(chunks)


def fetch(file_id: str, ranges: list[tuple[int, int]] | None = None, retries: int = RETRIES) -> bytes | list[bytes]:
    """Fichier entier, ou liste des plages demandées (dans l'ordre) via une réponse multipart."""
    headers = dict(HEADERS)
    if ranges:
        headers["Range"] = "bytes=" + ",".join(f"{a}-{b}" for a, b in ranges)
    for attempt in range(retries):
        try:
            request = urllib.request.Request(DOWNLOAD_URL.format(file_id), headers=headers)
            with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT) as response:
                body = read_with_deadline(response, time.monotonic() + REQUEST_DEADLINE)
                ctype = response.headers.get("Content-Type", "")
            if not ranges:
                return body
            return parse_ranges(body, ctype, ranges)
        except (OSError, ValueError, http.client.HTTPException) as exc:  # IncompleteRead : coupure en cours de réponse
            if attempt == retries - 1:
                raise
            delay = retry_delay(exc, attempt)
            print(f"  nouvelle tentative dans {delay:.0f} s ({exc})", flush=True)
            time.sleep(delay)
    raise AssertionError


def parse_ranges(body: bytes, ctype: str, ranges: list[tuple[int, int]]) -> list[bytes]:
    if len(ranges) == 1 and not ctype.startswith("multipart/"):
        if len(body) != ranges[0][1] - ranges[0][0] + 1:
            raise ValueError("Plage unique de taille inattendue")
        return [body]
    match = re.search(r"boundary=([^;]+)", ctype)
    if not match:
        raise ValueError(f"Réponse non multipart : {ctype}")
    boundary = b"--" + match.group(1).strip().encode()
    parts = {}
    for chunk in body.split(boundary)[1:]:
        if chunk.startswith(b"--"):
            break
        head, _, data = chunk.partition(b"\r\n\r\n")
        found = re.search(rb"Content-Range:\s*bytes (\d+)-(\d+)/", head, re.IGNORECASE)
        if not found:
            raise ValueError("Partie sans Content-Range")
        start, end = int(found.group(1)), int(found.group(2))
        parts[(start, end)] = data[: end - start + 1]
    out = [parts.get(r) for r in ranges]
    if any(p is None or len(p) != r[1] - r[0] + 1 for p, r in zip(out, ranges, strict=True)):
        raise ValueError("Plages manquantes ou tronquées dans la réponse")
    return out


def parse_header(header: bytes) -> dict:
    n = int(header[252:256])
    if len(header) < 256 * (1 + n):
        raise ValueError("En-tête incomplet")
    fields, offset = {}, 256
    for name, width in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8),
                        ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32)):  # fmt: skip
        fields[name] = [header[offset + width * i : offset + width * (i + 1)] for i in range(n)]
        offset += width * n
    record_s = float(header[244:252])
    samples = [int(s) for s in fields["samples"]]
    return {
        "n": n,
        "n_records": int(header[236:244]),
        "record_s": record_s,
        "start": dt.datetime.strptime(header[168:184].decode(), "%d.%m.%y%H.%M.%S"),
        "fields": fields,
        "samples": samples,
        "rates": [s / record_s for s in samples],
    }


def light_header(info: dict, keep: int, n_records: int) -> bytes:
    """En-tête EDF réduit aux `keep` premières voies."""
    fixed = bytearray(info["raw"][:256])
    fixed[184:192] = f"{256 * (1 + keep):<8}".encode()
    fixed[236:244] = f"{n_records:<8}".encode()
    fixed[252:256] = f"{keep:<4}".encode()
    body = b"".join(b"".join(info["fields"][name][:keep]) for name in info["fields"])
    return bytes(fixed) + body


def patient_segments(patient_dir: str) -> list[dict]:
    files = [f for f in list_folder(patient_dir) if f["fileName"].lower().endswith(".edf")]
    return sorted(files, key=lambda f: int(re.search(r"\[(\d+)\]", f["fileName"]).group(1)))


COMPARED_FIELDS = ("label", "transducer", "unit", "pmin", "pmax", "dmin", "dmax", "prefilter", "samples")


def check_compatible(first: dict, info: dict, keep: int) -> None:
    """Refuse une tranche dont les voies gardées diffèrent de la première (audit A1, 19-11).

    L'EDF assemblé garde l'en-tête de la première tranche et découpe chaque enregistrement
    d'après ses tailles : calibre, unité, fréquence ou durée d'enregistrement différents
    fausseraient le signal sans erreur. Les voies audio retirées peuvent différer.
    """
    if info["record_s"] != first["record_s"]:
        raise RuntimeError(f"{info['name']} : durée d'enregistrement {info['record_s']} au lieu de {first['record_s']}")
    for name in COMPARED_FIELDS:
        for index, (a, b) in enumerate(zip(first["fields"][name][:keep], info["fields"][name][:keep], strict=True)):
            if a.strip() != b.strip():
                voice = first["fields"]["label"][index].decode(errors="replace").strip()
                raise RuntimeError(f"{info['name']} : voie {voice}, {name} {b.strip()!r} au lieu de {a.strip()!r}")


class BudgetExceeded(Exception):
    """Le prochain EDF, de taille connue après lecture des en-têtes, dépasserait le volume maximal."""


def download_patient(
    patient: str, rml: dict, out: Path, workers: int = WORKERS, max_bytes: float | None = None
) -> dict:
    rml_bytes = fetch(rml["id"])  # vérifié avant l'EDF : un .rml faux échoue en quelques secondes
    if hashlib.md5(rml_bytes).hexdigest() != rml["md5"].lower():  # le dépôt donne 93 MD5 en majuscules
        raise RuntimeError("MD5 du .rml différent de celui du dépôt")
    segments = patient_segments(f"/V3/APNEA_EDF/{patient}")
    if not segments:
        raise RuntimeError("aucune EDF")
    infos, keep = [], None
    for seg in segments:
        first = fetch(seg["id"], [(0, 255)])[0]
        size = int(first[184:192])
        raw = fetch(seg["id"], [(0, size - 1)])[0]
        info = parse_header(raw) | {"raw": raw, "header_bytes": size, "id": seg["id"], "name": seg["fileName"]}
        audio = [i for i, r in enumerate(info["rates"]) if r >= AUDIO_MIN_RATE]
        this_keep = info["n"] - len(audio)
        if audio != list(range(this_keep, info["n"])):
            raise RuntimeError(f"{seg['fileName']} : voies audio pas en fin de bloc")
        if keep is not None and (
            this_keep != keep or info["fields"]["label"][:keep] != infos[0]["fields"]["label"][:keep]
        ):
            raise RuntimeError(f"{seg['fileName']} : voies différentes de la première tranche")
        if keep is not None:
            check_compatible(infos[0], info, keep)
        keep = this_keep
        infos.append(info)
    for previous, current in zip(infos[:-1], infos[1:], strict=True):
        expected = previous["start"] + dt.timedelta(seconds=previous["n_records"] * previous["record_s"])
        if current["start"] != expected:
            raise RuntimeError(f"{current['name']} : tranche non contiguë ({current['start']} au lieu de {expected})")
    record_bytes = [2 * sum(info["samples"]) for info in infos]
    kept_bytes = 2 * sum(infos[0]["samples"][:keep])
    total = sum(info["n_records"] for info in infos)
    size = 256 * (1 + keep) + total * kept_bytes
    if max_bytes is not None and size > max_bytes:  # taille exacte, avant tout octet écrit (audit A4)
        raise BudgetExceeded(f"{patient} : {size / 1e9:.2f} Go pour {max_bytes / 1e9:.2f} Go restants")
    partial = out / f"{patient}.edf.part"
    digest = hashlib.sha256()
    jobs = []
    for info, size in zip(infos, record_bytes, strict=True):
        offsets = [info["header_bytes"] + i * size for i in range(info["n_records"])]
        for start in range(0, len(offsets), RANGES_PER_REQUEST):
            jobs.append((info["id"], [(o, o + kept_bytes - 1) for o in offsets[start : start + RANGES_PER_REQUEST]]))
    with partial.open("wb") as handle, ThreadPoolExecutor(workers) as pool:
        header = light_header(infos[0], keep, total)
        handle.write(header)
        digest.update(header)
        # `map` rend les réponses dans l'ordre des requêtes : le fichier est écrit dans l'ordre.
        for parts in pool.map(lambda job: fetch(*job), jobs):
            data = b"".join(parts)
            handle.write(data)
            digest.update(data)
    os.replace(partial, out / f"{patient}.edf")
    (out / f"{patient}.rml").write_bytes(rml_bytes)
    return {
        "edf_sha256": digest.hexdigest(),
        "segments": [{"name": i["name"], "records": i["n_records"]} for i in infos],
        "kept_signals": [label.decode().strip() for label in infos[0]["fields"]["label"][:keep]],
        "duration_s": total * infos[0]["record_s"],
        "rml_md5": rml["md5"].lower(),
        "source_headers_sha256": [hashlib.sha256(i["raw"]).hexdigest() for i in infos],
    }


def frozen_order(approved: list[str], index: set[str], reverse: bool = False, limit: int | None = None) -> list[str]:
    """Ordre aléatoire figé (graine 42) sur la liste approuvée ; sens inverse appliqué avant la limite."""
    missing = sorted(set(approved) - index)
    if missing:
        raise SystemExit(f"Index ScienceDB modifié : {len(missing)} sujets approuvés absents ({missing[:3]})")
    ordered = sorted(approved)
    patients = [ordered[i] for i in np.random.default_rng(42).permutation(len(ordered))]
    return (patients[::-1] if reverse else patients)[:limit]


def intact(out: Path, patient: str, entry: dict) -> str | None:
    """Raison de reprendre un patient marqué terminé, ou None si ses deux fichiers sont conformes."""
    edf, rml = out / f"{patient}.edf", out / f"{patient}.rml"
    if not edf.exists() or not rml.exists():
        return "fichier absent"
    if hashlib.md5(rml.read_bytes()).hexdigest() != entry["rml_md5"]:
        return "MD5 du .rml différent du journal"
    digest = hashlib.sha256()
    with edf.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return None if digest.hexdigest() == entry["edf_sha256"] else "SHA-256 de l'EDF différent du journal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--passes", type=int, default=PASSES)
    # Second téléchargement depuis un autre réseau (Kaggle) : liste prise par la fin, volume borné.
    parser.add_argument("--reverse", action="store_true", help="parcourir l'ordre figé depuis la fin")
    parser.add_argument("--max-gb", type=float, default=None, help="arrêt avant de dépasser ce volume d'EDF")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "download_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"done": {}, "failed": {}}
    rmls = {
        f["fileName"].removesuffix(".rml"): f for f in list_folder("/V3/APNEA_RML") if f["fileName"].endswith(".rml")
    }
    # Ordre aléatoire figé (graine 42) : tout préfixe téléchargé est un tirage au sort sans biais,
    # si le téléchargement complet (environ 3 jours à 4 requêtes simultanées) devait être arrêté avant la fin.
    approved = json.loads(APPROVED.read_text(encoding="utf-8"))["subjects"]
    patients = frozen_order(approved, set(rmls), args.reverse, args.limit)
    for patient in [p for p in log["done"] if p in patients]:  # reprise : paires revérifiées (audit A4)
        if reason := intact(args.out, patient, log["done"][patient]):
            print(f"{patient} repris : {reason}", flush=True)
            del log["done"][patient]
    sizes = [(args.out / f"{p}.edf").stat().st_size for p in log["done"] if (args.out / f"{p}.edf").exists()]
    budget = None if args.max_gb is None else args.max_gb * 1e9
    stop = False
    for passage in range(1, args.passes + 1):
        todo = [p for p in patients if p not in log["done"]]
        print(f"Passage {passage} : {len(patients)} patients, {len(todo)} à télécharger", flush=True)
        if not todo or stop:
            break
        for index, patient in enumerate(todo):
            remaining = None if budget is None else budget - sum(sizes)
            begin = time.time()
            try:
                entry = download_patient(patient, rmls[patient], args.out, args.workers, remaining)
                log["done"][patient] = entry
                log["failed"].pop(patient, None)
                sizes.append((args.out / f"{patient}.edf").stat().st_size)
                print(f"[{index + 1}/{len(todo)}] {patient} : {time.time() - begin:.0f} s", flush=True)
            except BudgetExceeded as exc:
                print(f"Volume maximal atteint ({sum(sizes) / 1e9:.1f} Go) : arrêt ({exc})", flush=True)
                stop = True
                break
            except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
                log["failed"][patient] = str(exc)
                print(f"[{index + 1}/{len(todo)}] {patient} ÉCHEC : {exc}", flush=True)
            write_json_atomic(log_path, log)
    print(f"Fin : {len(log['done'])} faits, {len(log['failed'])} en échec {sorted(log['failed'])}", flush=True)


if __name__ == "__main__":
    main()
