"""Analyse des réponses du médecin : contrôle du CSV et lectures fixées avant, sur des réponses fictives."""

import pytest

from scripts import analyze_hypopnea_review as ana

WINDOWS = [{"id": f"F{i:02d}", "group": g} for i, g in enumerate("AAABBC", start=1)]
HEADER = "fenetre,q1_evenement,q2_criteres_aasm,q3_qualite,commentaire,relecteur"


def csv_text(rows):
    return "﻿" + "\r\n".join([HEADER] + [",".join(r) + ',"",Dr X' for r in rows])


ROWS = [
    ("F01", "hypopnée", "oui", "bonne"),
    ("F02", "incertain", "non", "artefact"),
    ("F03", "aucun", "non", "bonne"),
    ("F04", "hypopnée", "oui", "bonne"),
    ("F05", "hypopnée", "oui", "capteur décroché"),
    ("F06", "apnée", "pas d'hypopnée annotée", "bonne"),
]


def test_summary_follows_the_fixed_reading():
    answers = ana.read_answers(csv_text(ROWS), {w["id"] for w in WINDOWS})
    result = ana.summarize(WINDOWS, answers)
    assert result["clear_hypopnea_criteria_met"]["A"] == {"n": 1, "total": 3, "part": 0.333}
    assert result["clear_hypopnea_criteria_met"]["B"] == {"n": 2, "total": 2, "part": 1.0}
    assert result["event_seen_in_false_alarms"] == {"n": 1, "total": 1, "part": 1.0}
    assert result["poor_signal"]["A"]["n"] == 1 and result["poor_signal"]["B"]["n"] == 1
    assert result["counts"]["A"]["q1_evenement"] == {"hypopnée": 1, "incertain": 1, "aucun": 1}
    assert result["reviewers"] == ["Dr X"]


def test_missing_or_extra_window_is_refused():
    with pytest.raises(ValueError, match="fenêtres attendues"):
        ana.read_answers(csv_text(ROWS[:-1]), {w["id"] for w in WINDOWS})
    with pytest.raises(ValueError, match="fenêtres attendues"):
        ana.read_answers(csv_text(ROWS + [ROWS[0]]), {w["id"] for w in WINDOWS})


def test_unanswered_or_unknown_choice_is_refused():
    rows = [("F01", "", "oui", "bonne")] + ROWS[1:]
    with pytest.raises(ValueError, match="hors choix"):
        ana.read_answers(csv_text(rows), {w["id"] for w in WINDOWS})
