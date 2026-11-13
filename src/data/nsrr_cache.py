"""Contrôle des caches d'embeddings NSRR, partagé par l'extraction et l'évaluation (audit du 13 novembre).

Chaque nuit extraite laisse deux fichiers : `<sujet>.npz` (embeddings, indices et débuts des
fenêtres) et `<sujet>.json` (fiche : empreintes du NPZ et de l'EDF, nombre de fenêtres, canaux).
La fiche est écrite avant que le NPZ prenne son nom définitif : un NPZ sans fiche, modifié après
coup ou incohérent est refusé, à la reprise comme à l'évaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.cache_contract import file_sha256

EMBEDDING_DIMS = {"sleepfm": 1536, "osf": 768}
EPOCH_S = 30.0


def validate_night_arrays(data, models, n_epochs: int | None = None) -> int:
    """Nombre de fenêtres d'une nuit après contrôle ; `ValueError` à la moindre incohérence."""
    index = np.asarray(data["epoch_idx"])
    onset = np.asarray(data["onset_s"], dtype=float)
    n = len(index)
    if n == 0 or index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("indices de fenêtres absents ou non entiers")
    if not np.array_equal(index, np.arange(n)):
        raise ValueError("indices de fenêtres hors de la grille 0..n-1")
    if onset.shape != (n,) or not np.allclose(onset, EPOCH_S * index):
        raise ValueError("débuts de fenêtres différents de 30 s × indice")
    for model in models:
        matrix = np.asarray(data[f"X_{model}"])
        if matrix.shape != (n, EMBEDDING_DIMS[model]):
            raise ValueError(f"X_{model} : forme {matrix.shape}, attendu ({n}, {EMBEDDING_DIMS[model]})")
        if not np.isfinite(matrix).all():
            raise ValueError(f"X_{model} : valeurs non finies")
    if n_epochs is not None and n != n_epochs:
        raise ValueError(f"{n} fenêtres pour {n_epochs} époques déclarées")
    return n


def record_path(folder: Path, subject: str) -> Path:
    return folder / f"{subject}.json"


def read_record(folder: Path, subject: str) -> dict:
    path = record_path(folder, subject)
    if not path.exists():
        raise ValueError(f"{subject} : NPZ sans fiche, provenance inconnue")
    return json.loads(path.read_text())


def npz_unchanged(folder: Path, subject: str, record: dict) -> bool:
    """Contrôle rapide (taille, date) ; l'empreinte complète n'est recalculée qu'en cas de doute."""
    stat = (folder / f"{subject}.npz").stat()
    if stat.st_size == record["npz_size"] and stat.st_mtime_ns == record["npz_mtime_ns"]:
        return True
    return file_sha256(folder / f"{subject}.npz") == record["npz_sha256"]


def check_cache(folder: Path, subject: str, models, full: bool = True) -> dict:
    """Fiche d'une nuit terminée ; `full` recalcule l'empreinte et relit les matrices."""
    record = read_record(folder, subject)
    npz = folder / f"{subject}.npz"
    if full:
        if file_sha256(npz) != record["npz_sha256"]:
            raise ValueError(f"{subject} : NPZ modifié après l'extraction")
        with np.load(npz) as data:
            validate_night_arrays(data, models, record["n_epochs"])
    elif not npz_unchanged(folder, subject, record):
        raise ValueError(f"{subject} : NPZ modifié après l'extraction")
    return record
