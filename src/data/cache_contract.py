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


def import_embeddings(source_manifests: list[Path], destination: Path, expected: dict) -> dict:
    """Copie dans `destination` les caches des dossiers dont le manifeste
    respecte exactement le contrat `expected`.

    - Source au contrat différent ou incomplet : ignorée entièrement, sans
      aucune copie, et signalée.
    - Dossier de destination au contrat différent : arrêt.
    - Même fichier présent avec un contenu différent : arrêt (conflit).

    La destination reçoit le manifeste attendu et `provenance.json`, qui indique
    la source de chaque fichier importé. Renvoie {"copied", "ignored"}.
    """
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        differs = contract_differences(json.loads(manifest_path.read_text()), expected)
        if differs:
            raise ValueError(f"{destination} : cache existant au contrat différent ({', '.join(differs)})")

    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    copied, ignored = 0, []
    for source_manifest in sorted(source_manifests):
        source = source_manifest.parent
        manifest = json.loads(source_manifest.read_text())
        problems = missing_contract_keys(manifest) or contract_differences(manifest, expected)
        if problems:
            print(f"Ignoré : {source} (contrat différent ou incomplet : {', '.join(problems)})")
            ignored.append(str(source))
            continue
        for path in sorted([*source.glob("*.npz"), *source.glob("*.skipped")]):
            target = destination / path.name
            if target.exists():
                if file_sha256(target) != file_sha256(path):
                    raise ValueError(f"Conflit : {path.name} diffère entre {source} et {destination}")
                continue
            temporary = destination / f".{path.name}.tmp"
            shutil.copyfile(path, temporary)
            os.replace(temporary, target)
            provenance[path.name] = str(source)
            copied += 1
    if copied:
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"copied": copied, "ignored": ignored}
