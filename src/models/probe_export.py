"""Export et rechargement des sondes entraînées.

`save_probes` écrit, dans un dossier :
- `<sonde>.joblib` : la sonde ajustée (régression logistique, MLP, XGBoost,
  BiGRU), prête à prédire ;
- `manifest.json` : contrat d'entrée (embeddings SleepFM de dimension 1536,
  empreinte du checkpoint, version du prétraitement, canaux, fréquence), stades
  dans l'ordre des classes, paramètres effectifs et nuits de validation de
  chaque sonde, commit du code et versions des bibliothèques.

`load_probes` relit le dossier et exige un contrat complet et identique avant
toute prédiction, sauf demande explicite de provenance non vérifiée.
Les fichiers .joblib sont des pickles : ne charger que des exports de
confiance (les nôtres).
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from src.data.cache_contract import contract_differences, missing_contract_keys

EXPORT_FORMAT = 2
READABLE_EXPORT_FORMATS = (1, 2)


def _git(*arguments: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_environment() -> dict:
    """Code et bibliothèques réellement utilisés par un run."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "code_commit": _git("rev-parse", "HEAD") or "inconnu",
        "code_modified": None if status is None else bool(status),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": _version("xgboost"),
    }


def _hyperparameters(model) -> dict:
    params = model.get_params()
    return {k: v if isinstance(v, (int, float, str, bool, type(None))) else repr(v) for k, v in params.items()}


def save_probes(directory: Path, models: dict, embedding_dim: int, embeddings_manifest: dict | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probes = {}
    for kind, model in models.items():
        # BiGRU : poids sur CPU, pour recharger sans GPU. Le MLP a aussi un
        # attribut model_, mais c'est un MLPClassifier, pas un module torch.
        if isinstance(getattr(model, "model_", None), torch.nn.Module):
            model.model_.to("cpu")
            model.device = "cpu"
        joblib.dump(model, directory / f"{kind}.joblib")
        probes[kind] = {
            "file": f"{kind}.joblib",
            "classes": [int(c) for c in model.classes_],
            "needs_groups": bool(getattr(model, "needs_groups", False)),
            "hyperparameters": _hyperparameters(model),
            "fit": getattr(model, "fit_info_", None),
        }
    manifest = {
        "export_format": EXPORT_FORMAT,
        "input": {
            "embedding_dim": embedding_dim,
            "embeddings_manifest": embeddings_manifest,
            "groups": "BiGRU : (sujet, numéro d'époque[, début en s]) par ligne",
        },
        "probes": probes,
        "environment": run_environment(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Sondes exportées : {directory} ({', '.join(models)})")


def load_probes(directory: Path, embeddings_manifest: dict | None, allow_unverified: bool = False) -> tuple[dict, dict]:
    """Renvoie ({sonde: modèle}, manifeste). Le manifeste du cache à prédire
    et celui de l'entraînement doivent être complets et identiques (checkpoint,
    prétraitement, canaux, fréquence). `allow_unverified=True` réserve aux
    caches historiques : le contrôle devient un avertissement."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("export_format") not in READABLE_EXPORT_FORMATS:
        raise ValueError(f"{directory} : format d'export {manifest.get('export_format')} non pris en charge")
    trained_on = manifest["input"]["embeddings_manifest"]
    problems = [f"{key} absent du cache" for key in missing_contract_keys(embeddings_manifest)]
    problems += [f"{key} absent de l'export" for key in missing_contract_keys(trained_on)]
    if not problems:
        problems = [
            f"{key} : {embeddings_manifest[key]} contre {trained_on[key]} à l'entraînement"
            for key in contract_differences(embeddings_manifest, trained_on)
        ]
    if problems:
        message = f"Embeddings incompatibles avec les sondes ou non vérifiables : {' ; '.join(problems)}"
        if not allow_unverified:
            raise ValueError(message)
        print(f"ATTENTION : provenance non vérifiée. {message}")
    models = {}
    for kind, info in manifest["probes"].items():
        models[kind] = joblib.load(directory / info["file"])
        if [int(c) for c in models[kind].classes_] != info["classes"]:
            raise ValueError(f"{kind} : classes du modèle différentes de celles du manifeste")
    return models, manifest
