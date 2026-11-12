"""Références individuelles, durées couvertes et appariement exploratoire d'événements."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.respiratory_events import TARGET_EVENTS, event_intervals


def individual_events(samples, notes, n_samples):
    """Conserve chaque paire d'annotations, sans fusionner les événements voisins."""
    event_intervals(samples, notes, n_samples)  # Contrôles communs du lecteur initial.
    opened, records = {}, []
    for sample, note in zip(samples, notes, strict=True):
        name = note[1:] if note.startswith("(") else note[:-1] if note.endswith(")") else None
        if name not in TARGET_EVENTS:
            continue
        if note.startswith("("):
            opened[name] = int(sample)
        else:
            records.append((opened.pop(name), int(sample), TARGET_EVENTS.index(name)))
    return np.asarray(sorted(records), dtype=np.int64).reshape(-1, 3)


def intervals_array(intervals):
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return np.empty((0, 2))
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not np.isfinite(values).all()
        or (values[:, 0] < 0).any()
        or (values[:, 1] <= values[:, 0]).any()
    ):
        raise ValueError("Intervalles temporels invalides")
    return values


def union_intervals(intervals):
    values = intervals_array(intervals)
    result = []
    for start, end in values[np.argsort(values[:, 0], kind="stable")]:
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return np.asarray(result, dtype=float).reshape(-1, 2)


def covered_duration(events, windows):
    """Intersection de chaque événement avec l'union des fenêtres, sans double compte."""
    events, windows = intervals_array(events), union_intervals(windows)
    if not len(windows):
        return np.zeros(len(events))
    duration = windows[:, 1] - windows[:, 0]
    prefix = np.r_[0.0, np.cumsum(duration)]

    def before(t):
        index = np.searchsorted(windows[:, 0], t, side="right") - 1
        safe = np.maximum(index, 0)
        return np.where(index >= 0, prefix[safe] + np.clip(t - windows[safe, 0], 0, duration[safe]), 0.0)

    return np.clip(before(events[:, 1]) - before(events[:, 0]), 0.0, events[:, 1] - events[:, 0])


def match_one_to_one(overlap, event_duration, run_duration, minimum=10.0, tolerance=0.01):
    """Maximum de paires admissibles, puis somme des IoU ; un index de séquence par événement."""
    overlap = np.asarray(overlap, dtype=float)
    event_duration, run_duration = np.asarray(event_duration), np.asarray(run_duration)
    if overlap.ndim != 2 or event_duration.ndim != 1 or run_duration.ndim != 1:
        raise ValueError("Matrice de recouvrement ou durées invalides")
    if (
        overlap.shape != (len(event_duration), len(run_duration))
        or not np.isfinite(overlap).all()
        or (overlap < 0).any()
        or not np.isfinite(event_duration).all()
        or not np.isfinite(run_duration).all()
        or (event_duration <= 0).any()
        or (run_duration <= 0).any()
        or not np.isfinite(minimum)
        or minimum <= 0
        or not np.isfinite(tolerance)
        or tolerance < 0
        or (overlap > np.minimum(event_duration[:, None], run_duration[None, :]) + 1e-7).any()
    ):
        raise ValueError("Matrice de recouvrement ou durées invalides")
    matches = np.full(len(event_duration), -1, dtype=np.int64)
    if not overlap.size:
        return matches
    allowed = overlap + tolerance >= minimum
    iou = overlap / (event_duration[:, None] + run_duration[None, :] - overlap)
    # Le bonus dépasse toute variation possible de somme des IoU.
    weights = np.where(allowed, min(overlap.shape) + 1 + iou, 0.0)
    rows, columns = linear_sum_assignment(weights, maximize=True)
    keep = allowed[rows, columns]
    matches[rows[keep]] = columns[keep]
    return matches
