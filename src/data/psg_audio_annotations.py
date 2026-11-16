"""Annotations PSG-Audio (.rml Philips Sleepware) au même format que `parse_nsrr_xml`.

Contrôle externe validé le 13 novembre 2026 (plan dans Journal). Structure lue sur deux fichiers du
dépôt (libellés seulement) :
- stades notés par les médecins : `ScoringData/StagingData/UserStaging/NeuroAdultAASMStaging/Stage`
  (`Type`, `Start` en secondes ; un stade dure jusqu'au suivant, le dernier jusqu'à la fin de la
  session). La notation automatique (`MachineStaging`) n'est jamais lue.
- événements : `ScoringData/Events/Event` (`Family`, `Type`, `Start`, `Duration`). Famille
  `Respiratory` : 4 types ciblés ; respiration de Cheyne-Stokes, respiration périodique et RERA
  (README du dépôt) sont explicitement non ciblés ; tout autre type respiratoire est inconnu et
  bloque l'IAH de référence. Les autres familles (cardiaque, neuro, membres, ronflement, SpO2,
  utilisateur) ne sont pas comptées.
- durée : `Acquisition/Sessions/Session/Duration` (partie diagnostique seule dans le dépôt).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.respiratory_events import TARGET_EVENTS

STAGE_TYPES = {"Wake": "W", "NonREM1": "N1", "NonREM2": "N2", "NonREM3": "N3", "REM": "R"}
UNSCORED_TYPES = frozenset(("NotScored",))
EVENT_TYPES = {
    "ObstructiveApnea": "resp_obstructiveapnea",
    "CentralApnea": "resp_centralapnea",
    "MixedApnea": "resp_mixedapnea",
    "Hypopnea": "resp_hypopnea",
}
NON_TARGET_RESPIRATORY = frozenset(("CheyneStokesRespiration", "PeriodicRespiration", "RERA"))


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, *names: str) -> ET.Element:
    for name in names:
        found = next((c for c in element if _tag(c) == name), None)
        if found is None:
            raise ValueError(f"Élément {name} absent")
        element = found
    return element


def _number(value: str | None, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} invalide : {value!r}") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{what} invalide : {value!r}")
    return number


def parse_rml(source: str | Path | bytes) -> dict:
    raw = Path(source).read_bytes() if not isinstance(source, bytes) else source
    try:
        root = ET.fromstring(raw.decode("utf-8-sig").encode())
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise ValueError(f"RML illisible : {exc}") from exc
    session = _child(root, "Acquisition", "Sessions", "Session")
    duration = _number(_child(session, "Duration").text, "Durée de session")
    staging = _child(root, "ScoringData", "StagingData", "UserStaging", "NeuroAdultAASMStaging")
    marks, unknown = [], Counter()
    for stage in staging:
        if _tag(stage) == "Stage":
            marks.append((_number(stage.get("Start"), "Début de stade"), stage.get("Type") or ""))
    if not marks:
        raise ValueError("Aucun stade noté par les médecins")
    marks.sort()
    stages, unscored = [], 0
    for (start, kind), (end, _) in zip(marks, marks[1:] + [(duration, "")], strict=True):
        if end <= start:
            continue
        if kind in STAGE_TYPES:
            stages.append((start, end - start, STAGE_TYPES[kind]))
        elif kind in UNSCORED_TYPES:
            unscored += 1
        else:
            unknown[f"stage {kind}"] += 1
    events, other = [], Counter()
    for event in _child(root, "ScoringData", "Events"):
        if _tag(event) != "Event":
            continue
        family, kind = event.get("Family") or "", event.get("Type") or ""
        if family != "Respiratory":
            other[f"{family}:{kind}"] += 1
            continue
        if kind in NON_TARGET_RESPIRATORY:
            other[kind] += 1
            continue
        if kind not in EVENT_TYPES:
            unknown[kind] += 1
            continue
        start = _number(event.get("Start"), "Début d'événement")
        length = _number(event.get("Duration"), "Durée d'événement")
        if length <= 0:
            raise ValueError("Événement respiratoire de durée nulle")
        events.append((start, start + length, TARGET_EVENTS.index(EVENT_TYPES[kind])))
    return {
        "epoch_length": 30.0,
        "stages": stages,
        "events": np.asarray(sorted(events), dtype=float).reshape(-1, 3),
        "unscored_stage_events": unscored,
        "unknown_concepts": dict(unknown),
        "recording_end_s": duration,
        "other_concepts": dict(other),
    }
