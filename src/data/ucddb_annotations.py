"""Annotations UCDDB (PhysioNet) au même format que `parse_nsrr_xml`, pour `reference_ahi_nsrr`.

Contrôle externe validé le 13 novembre 2026 (plan dans Journal). Formats vérifiés sur la page et un
fichier publics :
- `<sujet>_stage.txt` : un code par époque de 30 s depuis le début de l'EDF ; 0 éveil, 1 REM,
  2 à 5 stades 1 à 4 (R&K, 3 et 4 regroupés en N3), 6 artefact et 7 indéterminé (non notés).
- `<sujet>_respevt.txt` : événements horodatés en heure d'horloge, recalés sur l'heure de début
  de l'EDF. `APNEA-O/C/M` et `HYP-O/C/M` (toutes les hypopnées comptent comme hypopnées) ; la
  colonne PB/CS (respiration périodique) n'est pas comptée, ni les types `POSSIBLE` et `PB EVENT`
  trouvés dans les fichiers ; le code de stade 8, non documenté, est traité comme non noté. Un type inconnu est signalé
  (`unknown_concepts`) et bloque l'IAH de référence, comme pour NSRR.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.respiratory_events import TARGET_EVENTS

EPOCH_S = 30.0
STAGE_CODES = {0: "W", 1: "R", 2: "N1", 3: "N2", 4: "N3", 5: "N3"}
# 8 : code présent dans 2 nuits, absent de la documentation ; traité comme non noté (décision du
# 13 novembre, prise sur les libellés seuls, avant tout résultat).
UNSCORED_CODES = frozenset((6, 7, 8))
EVENT_TYPES = {
    "APNEA-O": "resp_obstructiveapnea",
    "APNEA-C": "resp_centralapnea",
    "APNEA-M": "resp_mixedapnea",
    "HYP-O": "resp_hypopnea",
    "HYP-C": "resp_hypopnea",
    "HYP-M": "resp_hypopnea",
}
# Types non ciblés, explicitement écartés (13 novembre) : événement incertain, respiration périodique.
NON_TARGET_TYPES = frozenset(("POSSIBLE", "PB EVENT"))
EVENT_LINE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2})\s+(\S+(?: EVENT)?)\s+(?:(PB|CS)\s+)?(\d+(?:\.\d+)?)\b")


def edf_start_seconds(edf_path: str | Path) -> int:
    """Heure de début de l'EDF (champ hh.mm.ss de l'en-tête), en secondes depuis minuit."""
    with open(edf_path, "rb") as handle:
        header = handle.read(256)
    match = re.fullmatch(rb"(\d{2})\.(\d{2})\.(\d{2})", header[176:184].strip())
    if match is None:
        raise ValueError("Heure de début EDF illisible")
    h, m, s = (int(x) for x in match.groups())
    return 3600 * h + 60 * m + s


def parse_ucddb(stage_path: str | Path, respevt_path: str | Path, start_s: int) -> dict:
    """Stades (début, durée, stade) et événements (début, fin, type) en secondes depuis le début."""
    stages, unscored, unknown = [], 0, Counter()
    codes = [line.strip() for line in Path(stage_path).read_text().splitlines() if line.strip()]
    if not codes:
        raise ValueError("Fichier de stades vide")
    for epoch, text in enumerate(codes):
        if not text.isdigit():
            raise ValueError(f"Code de stade illisible : {text!r}")
        code = int(text)
        if code in STAGE_CODES:
            stages.append((epoch * EPOCH_S, EPOCH_S, STAGE_CODES[code]))
        elif code in UNSCORED_CODES:
            unscored += 1
        else:
            unknown[f"stage {code}"] += 1
    events, periodic, other = [], 0, Counter()
    for line in Path(respevt_path).read_text(errors="replace").splitlines():
        if not re.match(r"^\s*\d{2}:\d{2}:\d{2}", line):
            continue
        match = EVENT_LINE.match(line)
        if match is None:
            raise ValueError(f"Ligne d'événement illisible : {line.strip()!r}")
        h, m, s, kind, flag, duration = match.groups()
        periodic += flag is not None
        if kind in NON_TARGET_TYPES:
            other[kind] += 1
            continue
        if kind not in EVENT_TYPES:
            unknown[kind] += 1
            continue
        onset = (3600 * int(h) + 60 * int(m) + int(s) - start_s) % 86400
        length = float(duration)
        if length <= 0:
            raise ValueError("Événement respiratoire de durée nulle")
        events.append((float(onset), onset + length, TARGET_EVENTS.index(EVENT_TYPES[kind])))
    return {
        "epoch_length": EPOCH_S,
        "stages": stages,
        "events": np.asarray(sorted(events), dtype=float).reshape(-1, 3),
        "unscored_stage_events": unscored,
        "unknown_concepts": dict(unknown),
        "recording_end_s": len(codes) * EPOCH_S,
        "other_concepts": dict(other) | ({"periodic_breathing_flag": periodic} if periodic else {}),
    }
