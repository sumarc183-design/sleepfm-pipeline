"""Validation des identifiants utilisés dans les chemins et URL PhysioNet."""

from __future__ import annotations

import json
import re
from pathlib import Path


def validate_subject_ids(subjects, source: str = "Sujets", allow_empty: bool = False) -> list[str]:
    if not isinstance(subjects, list) or (not subjects and not allow_empty):
        raise ValueError(f"{source} : attendu une liste {'de sujets' if allow_empty else 'non vide de sujets'}")
    seen = set()
    for subject in subjects:
        if not isinstance(subject, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject) is None:
            raise ValueError(f"{source} : identifiant invalide {subject!r} (lettres, chiffres, tirets et underscores)")
        if subject in seen:
            raise ValueError(f"{source} : sujet en double {subject!r}")
        seen.add(subject)
    return subjects


def load_subjects_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "subjects" not in payload:
        raise ValueError(f'{path} : attendu un objet JSON {{"subjects": [...]}}')
    return validate_subject_ids(payload["subjects"], str(path))
