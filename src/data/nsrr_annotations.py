"""Lecture des annotations XML NSRR (MESA, SHHS) : stades, événements respiratoires, IAH.

Préparé avant l'accès aux données, d'après le format public « NSRR XML » (`PSGAnnotation` >
`ScoredEvents` > `ScoredEvent` avec `EventType`, `EventConcept`, `Start`, `Duration`). Les
libellés exacts restent à vérifier sur les premiers fichiers reçus : un stade ou un concept
respiratoire inconnu est compté à part (`unknown_concepts`) et bloque l'IAH de référence ; il
n'est jamais converti en silence en absence d'événement. Un XML illisible lève `ValueError`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.respiratory_events import TARGET_EVENTS

# Concepts NSRR (partie avant « | », en minuscules) → stade AASM.
STAGE_CONCEPTS = {
    "wake": "W",
    "stage 1 sleep": "N1",
    "stage 2 sleep": "N2",
    "stage 3 sleep": "N3",
    "stage 4 sleep": "N3",  # Rechtschaffen-Kales : stades 3 et 4 regroupés en N3.
    "rem sleep": "R",
}
UNSCORED_STAGE_CONCEPTS = frozenset(("unscored", "movement"))
# Concepts NSRR → types d'événements utilisés pour PhysioNet 2018 (même ordre que TARGET_EVENTS).
RESPIRATORY_CONCEPTS = {
    "obstructive apnea": "resp_obstructiveapnea",
    "central apnea": "resp_centralapnea",
    "mixed apnea": "resp_mixedapnea",
    "hypopnea": "resp_hypopnea",
}
# Concepts respiratoires attendus mais non ciblés par la règle (désaturations, artefacts d'oxymétrie).
# Tout autre concept de la famille « Respiratory », ou contenant « apnea », est inconnu.
NON_TARGET_RESPIRATORY = frozenset(("spo2 desaturation", "spo2 artifact"))
SLEEP_STAGES = ("N1", "N2", "N3", "R")


def _concept(text: str | None) -> str:
    return (text or "").split("|", 1)[0].strip().lower()


def _number(event: ET.Element, tag: str) -> float:
    node = event.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Champ {tag} absent")
    value = float(node.text)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Champ {tag} invalide : {node.text}")
    return value


def parse_nsrr_xml(source: str | Path | bytes) -> dict:
    """Stades (début, durée, stade) et événements respiratoires (début, fin, type) en secondes."""
    try:
        root = ET.fromstring(source) if isinstance(source, bytes) else ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"XML illisible : {exc}") from exc
    if root.tag != "PSGAnnotation":
        raise ValueError("Racine PSGAnnotation attendue")
    epoch = root.findtext("EpochLength")
    epoch_length = float(epoch) if epoch else 30.0
    stages, events = [], []
    unscored, unknown, other = 0, Counter(), Counter()
    for event in root.iter("ScoredEvent"):
        family = _concept(event.findtext("EventType"))
        concept = _concept(event.findtext("EventConcept"))
        if family == "stages":
            if concept in STAGE_CONCEPTS:
                stages.append((_number(event, "Start"), _number(event, "Duration"), STAGE_CONCEPTS[concept]))
            elif concept in UNSCORED_STAGE_CONCEPTS:
                unscored += 1
            else:
                unknown[concept] += 1
        elif concept in RESPIRATORY_CONCEPTS:
            start, duration = _number(event, "Start"), _number(event, "Duration")
            if duration <= 0:
                raise ValueError("Événement respiratoire de durée nulle")
            events.append((start, start + duration, TARGET_EVENTS.index(RESPIRATORY_CONCEPTS[concept])))
        elif (family == "respiratory" and concept not in NON_TARGET_RESPIRATORY) or "apnea" in concept:
            unknown[concept] += 1
        elif concept:
            other[concept] += 1
    stages.sort()
    for (s0, d0, _), (s1, _, _) in zip(stages[:-1], stages[1:], strict=True):
        if s1 < s0 + d0 - 1e-6:
            raise ValueError("Stades qui se chevauchent")
    return {
        "epoch_length": epoch_length,
        "stages": stages,
        "events": np.asarray(sorted(events), dtype=float).reshape(-1, 3),
        "unscored_stage_events": unscored,
        "unknown_concepts": dict(unknown),
        "recording_end_s": max((start + duration for start, duration, _ in stages), default=0.0),
        "other_concepts": dict(other),
    }


def reference_ahi_nsrr(parsed: dict) -> dict:
    """Même règle que l'étude de faisabilité : 4 types d'événements annotés / heures de sommeil."""
    if parsed["unknown_concepts"]:
        raise ValueError(f"Concepts inconnus : {sorted(parsed['unknown_concepts'])}")
    sleep_h = sum(duration for _, duration, stage in parsed["stages"] if stage in SLEEP_STAGES) / 3600
    kinds = parsed["events"][:, 2].astype(int)
    counts = {kind: int((kinds == k).sum()) for k, kind in enumerate(TARGET_EVENTS)}
    total = sum(counts.values())
    return {
        "sleep_h": sleep_h,
        "event_counts": counts,
        "events": total,
        "ahi": total / sleep_h if sleep_h > 0 else float("nan"),
    }


def epoch_stages(parsed: dict, n_epochs: int) -> np.ndarray:
    """Stade par époque de 30 s ('' = non noté), pour aligner les fenêtres des modèles."""
    if parsed["epoch_length"] != 30.0:
        raise ValueError("Époques de 30 s attendues")
    labels = np.full(n_epochs, "", dtype="U2")
    for start, duration, stage in parsed["stages"]:
        first = start / 30.0
        count = duration / 30.0
        if abs(first - round(first)) > 1e-6 or abs(count - round(count)) > 1e-6:
            raise ValueError("Stade non aligné sur la grille de 30 s")
        labels[int(round(first)) : min(n_epochs, int(round(first + count)))] = stage
    return labels
