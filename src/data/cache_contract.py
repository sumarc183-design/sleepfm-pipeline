"""Contrat d'un cache d'embeddings et import sûr depuis d'autres dossiers.

Le manifeste d'un cache décrit ce qui détermine ses embeddings. Deux caches ne
se réunissent, et des sondes ne prédisent sur un cache, que si ce contrat est
identique : checkpoint, prétraitement, canaux dans le même ordre, fréquence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

CONTRACT_KEYS = ("checkpoint_sha256", "preprocessing_version", "channels", "sample_rate")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_contract_keys(manifest: dict | None) -> list[str]:
    """Clés du contrat absentes ou vides dans un manifeste."""
    if not isinstance(manifest, dict):
        return list(CONTRACT_KEYS)
    return [key for key in CONTRACT_KEYS if manifest.get(key) in (None, "", [])]


def contract_differences(a: dict, b: dict) -> list[str]:
    """Clés du contrat sur lesquelles deux manifestes diffèrent."""
    return [key for key in CONTRACT_KEYS if a.get(key) != b.get(key)]


def write_json_atomic(path: Path, value) -> None:
    """Écrit un JSON sous un nom temporaire puis le renomme : jamais de fichier tronqué."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def import_embeddings(source_manifests: list[Path], destination: Path, expected: dict) -> dict:
    """Copie dans `destination` les caches des dossiers dont le manifeste
    respecte exactement le contrat `expected`.

    - Source au contrat différent ou incomplet : ignorée entièrement, sans
      aucune copie, et signalée.
    - Destination au contrat différent : arrêt. Destination avec des caches mais
      sans manifeste : arrêt, leur provenance inconnue ne devient pas attestée.
    - Même fichier avec un contenu différent, entre sources ou avec la
      destination : arrêt avant toute copie.

    Les fichiers sont copiés dans un dossier temporaire, puis la provenance
    (source et SHA-256 de chaque fichier) et le manifeste sont écrits, puis les
    fichiers sont publiés. Après une interruption, le dossier ne contient donc
    que des fichiers vérifiés sous ce contrat, et une relance complète le reste.
    Renvoie {"copied", "ignored"}.
    """
    missing = missing_contract_keys(expected)
    if missing:
        raise ValueError(f"Contrat attendu incomplet : {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text())
        differs = missing_contract_keys(current) or contract_differences(current, expected)
        if differs:
            raise ValueError(f"{destination} : cache existant au contrat différent ou incomplet ({', '.join(differs)})")
    else:
        orphans = sorted(p.name for pattern in ("*.npz", "*.skipped") for p in destination.glob(pattern))
        if orphans:
            raise ValueError(
                f"{destination} : {len(orphans)} fichier(s) sans manifeste ({orphans[0]}…), provenance inconnue. "
                "Importer dans un dossier vide."
            )

    planned, ignored = {}, []
    for source_manifest in sorted(source_manifests):
        source = source_manifest.parent
        manifest = json.loads(source_manifest.read_text())
        problems = missing_contract_keys(manifest) or contract_differences(manifest, expected)
        if problems:
            print(f"Ignoré : {source} (contrat différent ou incomplet : {', '.join(problems)})")
            ignored.append(str(source))
            continue
        for path in sorted([*source.glob("*.npz"), *source.glob("*.skipped")]):
            digest = file_sha256(path)
            target = destination / path.name
            if target.exists():
                if file_sha256(target) != digest:
                    raise ValueError(f"Conflit : {path.name} diffère entre {source} et {destination}")
            elif path.name in planned:
                if planned[path.name]["sha256"] != digest:
                    raise ValueError(f"Conflit : {path.name} diffère entre {planned[path.name]['source']} et {source}")
            else:
                planned[path.name] = {"source": str(source), "sha256": digest, "path": path}
    if not planned:
        return {"copied": 0, "ignored": ignored}

    staging = destination / ".import_en_cours"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for name, entry in planned.items():
        shutil.copyfile(entry["path"], staging / name)
        if file_sha256(staging / name) != entry["sha256"]:
            raise OSError(f"{name} : copie différente de sa source")
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance |= {name: {"source": e["source"], "sha256": e["sha256"]} for name, e in planned.items()}
    write_json_atomic(provenance_path, provenance)
    write_json_atomic(manifest_path, expected)
    for name in planned:
        os.replace(staging / name, destination / name)
    staging.rmdir()
    return {"copied": len(planned), "ignored": ignored}
