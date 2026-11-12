"""Référence des fenêtres mixtes depuis les mêmes indicateurs HDF5 que la grille uniforme."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.data.epoch_dataset import STAGE_NAME_TO_LABEL
from src.data.extract_physionet2018 import import_signal_names

CATEGORIES = (
    "uniform",
    "near_uniform",
    "transition_majority",
    "transition_no_majority",
    "undefined",
    "partial_undefined",
)


def labels_from_counts(stage_counts, undefined_counts, sample_rate: int) -> dict:
    """Majorité absolue ; les fenêtres à un échantillon minoritaire sont séparées."""
    counts, undefined = np.asarray(stage_counts), np.asarray(undefined_counts)
    if (
        type(sample_rate) is not int
        or sample_rate <= 0
        or counts.ndim != 2
        or counts.shape[1] != 5
        or undefined.shape != (len(counts),)
        or not np.issubdtype(counts.dtype, np.integer)
        or not np.issubdtype(undefined.dtype, np.integer)
        or (counts < 0).any()
        or (undefined < 0).any()
    ):
        raise ValueError("Décomptes de stades invalides")
    samples = sample_rate * 30
    if not np.all(counts.sum(axis=1) + undefined == samples):
        raise ValueError("Les durées ne couvrent pas exactement les fenêtres")
    dominant = counts.argmax(axis=1)
    duration = counts.max(axis=1)
    defined = undefined == 0
    uniform = defined & (duration == samples)
    majority = defined & (2 * duration > samples)
    near_uniform = majority & (samples - duration == 1)
    category = np.full(len(counts), "partial_undefined", dtype="U30")
    category[undefined == samples] = "undefined"
    category[defined & ~majority] = "transition_no_majority"
    category[majority & ~uniform & ~near_uniform] = "transition_majority"
    category[near_uniform] = "near_uniform"
    category[uniform] = "uniform"
    return {
        "y_uniform": np.where(uniform, dominant, -1).astype(np.int64),
        "y_majority": np.where(majority, dominant, -1).astype(np.int64),
        "category": category,
        "stage_duration_s": counts / sample_rate,
        "undefined_s": undefined / sample_rate,
    }


def read_majority_grid(record: Path, onsets) -> dict:
    """Lit .hea et -arousal.mat ; pas de signal, pas d'interpolation de marqueurs WFDB."""
    _, frequency, n_samples = import_signal_names(str(record.with_suffix(".hea")))
    onsets = np.asarray(onsets)
    if (
        frequency <= 0
        or n_samples <= 0
        or onsets.ndim != 1
        or not len(onsets)
        or not np.array_equal(onsets, np.arange(len(onsets)) * 30.0)
    ):
        raise ValueError("Grille fixe et en-tête valides requis")
    window = frequency * 30
    full = len(onsets) * window
    if full > n_samples:
        raise ValueError("Fenêtres au-delà de l'enregistrement")
    counts = np.zeros((len(onsets), 5), dtype=np.int64)
    undefined = np.zeros(len(onsets), dtype=np.int64)
    active = np.zeros(n_samples, dtype=np.uint8)
    with h5py.File(record.parent / f"{record.name}-arousal.mat") as f:
        stages = f["data/sleep_stages"]
        if set(stages) != {*STAGE_NAME_TO_LABEL, "undefined"}:
            raise ValueError("Colonnes de stades inattendues")
        for stage, dataset in stages.items():
            if dataset.shape != (1, n_samples):
                raise ValueError("Longueur d'annotation différente de l'en-tête")
            values = dataset[0]
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Indicateurs de stade non binaires")
            active += values.astype(np.uint8)
            by_window = values[:full].reshape(len(onsets), window).sum(axis=1).astype(np.int64)
            if stage == "undefined":
                undefined = by_window
            else:
                counts[:, STAGE_NAME_TO_LABEL[stage]] = by_window
    if not np.all(active == 1):
        raise ValueError("Un seul stade actif requis par échantillon")
    return labels_from_counts(counts, undefined, frequency)
