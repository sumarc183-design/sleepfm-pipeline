"""Export et rechargement des sondes entraînées.

`save_probes` écrit, dans un dossier :
- `<sonde>.joblib` : la sonde ajustée (régression logistique, MLP, XGBoost,
  BiGRU), prête à prédire ;
- `manifest.json` : contrat d'entrée (embeddings SleepFM de dimension 1536,
  empreinte du checkpoint, version du prétraitement), stades dans l'ordre des
  classes, hyperparamètres, commit du code et versions des bibliothèques.

`load_probes` relit le dossier et vérifie le contrat avant toute prédiction.
Les fichiers .joblib sont des pickles : ne charger que des exports de
confiance (les nôtres).
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

EXPORT_FORMAT = 1


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "inconnu"


def _hyperparameters(model) -> dict:
    if hasattr(model, "get_params"):
        params = model.get_params()
    else:  # SequenceProbe
        keys = ("hidden_dim", "epochs", "lr", "dropout", "window", "batch_size", "seed", "context", "class_weight")
        params = {k: getattr(model, k, None) for k in keys}
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
        }
    manifest = {
        "export_format": EXPORT_FORMAT,
        "input": {
            "embedding_dim": embedding_dim,
            "embeddings_manifest": embeddings_manifest,
            "groups": "BiGRU : (sujet, numéro d'époque[, début en s]) par ligne",
        },
        "probes": probes,
        "code_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Sondes exportées : {directory} ({', '.join(models)})")


def load_probes(directory: Path, embeddings_manifest: dict | None = None) -> tuple[dict, dict]:
    """Renvoie ({sonde: modèle}, manifeste). Si `embeddings_manifest` est
    fourni (celui du cache à prédire), le checkpoint et le prétraitement
    doivent être ceux de l'entraînement."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("export_format") != EXPORT_FORMAT:
        raise ValueError(f"{directory} : format d'export {manifest.get('export_format')} non pris en charge")
    trained_on = manifest["input"]["embeddings_manifest"]
    if embeddings_manifest is not None and trained_on is not None:
        for key in ("checkpoint_sha256", "preprocessing_version"):
            if embeddings_manifest.get(key) != trained_on.get(key):
                raise ValueError(
                    f"Embeddings incompatibles avec les sondes ({key} : {embeddings_manifest.get(key)} "
                    f"contre {trained_on.get(key)} à l'entraînement)"
                )
    models = {kind: joblib.load(directory / info["file"]) for kind, info in manifest["probes"].items()}
    return models, manifest
