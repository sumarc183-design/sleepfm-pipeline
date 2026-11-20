"""Réunit les parties Kaggle d'une extraction NSRR (`embeddings_nsrr_<cohorte>_part<N>.zip`).

Contrôles avant toute écriture :
- chaque zip ne contient que des fichiers à plat (`.npz`, `.json`, `.skipped`), sans chemin ;
- toutes les parties portent le même `manifest.json`, identique à celui du dossier de sortie s'il
  existe (même liste figée, mêmes poids, même règle, même code de préparation) ;
- une nuit présente dans deux sources doit y être identique octet pour octet, sinon refus ;
- `run_status.json` des parties est ignoré : relancer `embed_nsrr.py` sans EDF le recalcule.

Affiche la couverture de la liste figée (faites, rejetées, manquantes).

    python scripts/merge_embedding_parts.py --out embeddings/nsrr_mesa ~/Downloads/embeddings_nsrr_mesa_part*.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEPT_SUFFIXES = (".npz", ".json", ".skipped")
IGNORED = {"run_status.json"}


def read_parts(zips: list[Path]) -> tuple[dict, dict[str, tuple[bytes, str]]]:
    """Manifeste commun et fichiers {nom: (octets, zip d'origine)} ; lève une erreur sinon."""
    manifest, files = None, {}
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            unsafe = [n for n in names if "/" in n or "\\" in n or n.startswith(".") or ".." in n]
            if unsafe:
                raise ValueError(f"{path.name} : chemins refusés {unsafe[:3]}")
            if "manifest.json" not in names:
                raise ValueError(f"{path.name} : manifest.json absent")
            this = json.loads(archive.read("manifest.json"))
            if manifest is not None and this != manifest:
                raise ValueError(f"{path.name} : manifeste différent des autres parties")
            manifest = this
            for name in names:
                if name == "manifest.json" or name in IGNORED or ".tmp" in name:
                    continue
                if not name.endswith(KEPT_SUFFIXES):
                    raise ValueError(f"{path.name} : fichier inattendu {name}")
                data = archive.read(name)
                if name in files and files[name][0] != data:
                    raise ValueError(f"{name} : différent entre {files[name][1]} et {path.name}")
                files[name] = (data, path.name)
    if manifest is None:
        raise ValueError("Aucune partie")
    return manifest, files


def merge(zips: list[Path], out: Path) -> dict:
    manifest, files = read_parts(zips)
    existing = out / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"{existing} : manifeste différent des parties (utiliser un dossier neuf)")
    conflicts = [n for n, (data, _) in files.items() if (out / n).exists() and (out / n).read_bytes() != data]
    if conflicts:
        raise ValueError(f"Fichiers déjà présents et différents dans {out} : {sorted(conflicts)[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    if not existing.exists():
        existing.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    for name, (data, _) in files.items():
        (out / name).write_bytes(data)
    subjects = json.loads((ROOT / f"configs/nsrr_{manifest['cohort']}_subjects.json").read_text(encoding="utf-8"))[
        "subjects"
    ]
    done = [s for s in subjects if (out / f"{s}.npz").exists() and (out / f"{s}.json").exists()]
    rejected = [s for s in subjects if (out / f"{s}.skipped").exists()]
    return {
        "cohort": manifest["cohort"],
        "parts": len(zips),
        "files_written": len(files),
        "done": len(done),
        "rejected": len(rejected),
        "missing": [s for s in subjects if s not in set(done) | set(rejected)],
        "sha256_of_parts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in zips},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("zips", type=Path, nargs="+")
    args = parser.parse_args()
    report = merge(sorted(args.zips), args.out)
    missing = report["missing"]
    print(
        f"{report['cohort']} : {report['parts']} partie(s), {report['done']} nuits faites, "
        f"{report['rejected']} rejetées, {len(missing)} manquantes {missing[:5]}"
    )
    print("Ensuite : relancer embed_nsrr.py sur ce dossier (sans EDF) pour recalculer run_status.json.")


if __name__ == "__main__":
    main()
