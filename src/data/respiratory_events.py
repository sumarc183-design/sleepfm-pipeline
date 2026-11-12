"""Événements WFDB respiratoires et labels sur les fenêtres SleepFM existantes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.data.extract_physionet2018 import import_signal_names

TARGET_EVENTS = (
    "resp_obstructiveapnea",
    "resp_centralapnea",
    "resp_mixedapnea",
    "resp_hypopnea",
)
MIN_OVERLAP_S = 10.0
EXCLUDED = -1


def event_intervals(samples, notes, n_samples: int) -> tuple[np.ndarray, dict]:
    """Intervalles [début, fin), en échantillons ; annotations mal appariées refusées."""
    samples = np.asarray(samples)
    if (
        samples.ndim != 1
        or len(samples) != len(notes)
        or not np.issubdtype(samples.dtype, np.integer)
        or (samples < 0).any()
        or (samples > n_samples).any()
        or (np.diff(samples) < 0).any()
    ):
        raise ValueError("Temps d'annotations invalides")
    opened, intervals, counts = {}, [], Counter()
    for sample, note in zip(samples, notes, strict=True):
        if not isinstance(note, str):
            raise ValueError("Annotation non textuelle")
        name = note[1:] if note.startswith("(") else note[:-1] if note.endswith(")") else None
        if name not in TARGET_EVENTS:
            continue
        if note.startswith("("):
            if name in opened:
                raise ValueError(f"Début en double : {name}")
            opened[name] = int(sample)
        else:
            if name not in opened or sample <= opened[name]:
                raise ValueError(f"Fin sans début valide : {name}")
            intervals.append((opened.pop(name), int(sample)))
            counts[name] += 1
    if opened:
        raise ValueError(f"Événements sans fin : {sorted(opened)}")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2), {k: counts[k] for k in TARGET_EVENTS}


def window_targets(onsets, intervals: np.ndarray, sample_rate: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Positif >= 10 s, négatif = 0 s, couverture intermédiaire exclue."""
    onsets = np.asarray(onsets, dtype=float)
    if sample_rate <= 0 or onsets.ndim != 1 or not np.isfinite(onsets).all() or (onsets < 0).any():
        raise ValueError("Fenêtres ou fréquence invalides")
    starts = np.rint(onsets * sample_rate).astype(np.int64)
    ends = starts + 30 * sample_rate
    if (ends > n_samples).any():
        raise ValueError("Fenêtre dépassant l'enregistrement")
    intervals = np.asarray(intervals)
    if intervals.shape != (len(intervals), 2) or not np.issubdtype(intervals.dtype, np.integer):
        raise ValueError("Intervalles entiers (n, 2) attendus")
    if len(intervals) and (
        (intervals[:, 0] < 0).any()
        or (intervals[:, 1] > n_samples).any()
        or (intervals[:, 1] <= intervals[:, 0]).any()
        or (intervals[1:, 0] < intervals[:-1, 1]).any()
    ):
        raise ValueError("Intervalles non triés, chevauchants ou hors enregistrement")
    if len(intervals):
        durations = intervals[:, 1] - intervals[:, 0]
        prefix = np.r_[0, np.cumsum(durations)]

        def covered_until(t):
            i = np.searchsorted(intervals[:, 0], t, side="right") - 1
            safe = np.maximum(i, 0)
            values = prefix[safe] + np.clip(t - intervals[safe, 0], 0, durations[safe])
            return np.where(i < 0, 0, values)

        overlap_samples = covered_until(ends) - covered_until(starts)
    else:
        overlap_samples = np.zeros(len(starts), dtype=np.int64)
    y = np.where(overlap_samples >= MIN_OVERLAP_S * sample_rate, 1, EXCLUDED)
    y[overlap_samples == 0] = 0
    return y.astype(np.int64), overlap_samples / sample_rate


def read_respiratory_labels(record: Path, onsets) -> dict:
    """Lit .hea et .arousal seulement ; jamais le signal ni les labels de stades."""
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Lecteur WFDB absent : voir requirements-respiratory.txt et le guide") from exc
    _, sample_rate, n_samples = import_signal_names(str(record.with_suffix(".hea")))
    annotation = wfdb.rdann(str(record), "arousal")
    if annotation.fs is not None and annotation.fs != sample_rate:
        raise ValueError("Fréquence des annotations différente de l'en-tête")
    intervals, counts = event_intervals(annotation.sample, annotation.aux_note, n_samples)
    y, overlap_s = window_targets(onsets, intervals, sample_rate, n_samples)
    return {"y": y, "overlap_s": overlap_s, "intervals_s": intervals / sample_rate, "event_counts": counts}
