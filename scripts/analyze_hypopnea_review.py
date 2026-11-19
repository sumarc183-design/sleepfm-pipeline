"""Analyse des réponses du médecin (examen des hypopnées, plan du 17 novembre), lecture fixée avant.

Entrées : l'échantillon figé (`results/hypopnea_review_sample.json`) et le CSV exporté par la page.
Le CSV est refusé s'il ne couvre pas exactement les 40 fenêtres ou contient une réponse hors des
choix proposés ; le plan doit être celui du tirage (empreinte). Le CSV porte l'empreinte du tirage
(colonne `tirage_sha256`, audit A7 du 19-11) : une empreinte différente est refusée ; un CSV sans
cette colonne (page envoyée avant le 19-11) est accepté et signalé « non vérifiable ».
Proportions par groupe, descriptives,
sans test (effectifs trop faibles) :
- A et B : part « hypopnée nette, critères remplis » (Q1 = hypopnée et Q2 = oui) ;
- C : part des fausses alertes où un événement est vu (Q1 = hypopnée ou apnée) ;
- chaque groupe : part de signal de mauvaise qualité (Q3 ≠ bonne).

    python scripts/analyze_hypopnea_review.py --sample results/hypopnea_review_sample.json \\
        --answers relecture_hypopnees_X.csv --plan "docs/plans/Plan examen hypopnées médecin 17-11.md" \\
        --out results/hypopnea_review_answers.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402

CHOICES = {
    "q1_evenement": {"aucun", "hypopnée", "apnée", "incertain"},
    "q2_criteres_aasm": {"oui", "non", "incertain", "pas d'hypopnée annotée"},
    "q3_qualite": {"bonne", "artefact", "capteur décroché"},
}
GROUPS = ("A", "B", "C")


def read_answers(text: str, ids: set[str]) -> dict[str, dict]:
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    answers = {r["fenetre"]: r for r in rows}
    if len(rows) != len(answers) or set(answers) != ids:
        raise ValueError(f"CSV : {len(rows)} lignes, fenêtres attendues {len(ids)} (doublons, manquantes ou inconnues)")
    for window, row in answers.items():
        for question, allowed in CHOICES.items():
            if row.get(question, "") not in allowed:
                raise ValueError(f"{window} : réponse « {row.get(question, '')} » hors choix pour {question}")
    return answers


def sample_check(answers: dict[str, dict], sample_sha256: str) -> str:
    """Rattachement du CSV au tirage : vérifié, refusé, ou non vérifiable pour une page antérieure."""
    found = {row.get("tirage_sha256") for row in answers.values()}
    if found == {None}:
        return "non vérifiable : CSV sans empreinte du tirage (page antérieure au 19-11)"
    if found != {sample_sha256}:
        raise ValueError(f"CSV d'un autre tirage : {sorted(str(f) for f in found)} au lieu de {sample_sha256}")
    return "vérifié"


def proportion(hits: int, total: int) -> dict:
    return {"n": hits, "total": total, "part": round(hits / total, 3) if total else None}


def summarize(windows: list[dict], answers: dict[str, dict]) -> dict:
    by_group = {g: [answers[w["id"]] for w in windows if w["group"] == g] for g in GROUPS}
    counts = {g: {q: dict(Counter(r[q] for r in rows)) for q in CHOICES} for g, rows in by_group.items()}
    clear = {
        g: proportion(sum(r["q1_evenement"] == "hypopnée" and r["q2_criteres_aasm"] == "oui" for r in by_group[g]),
                      len(by_group[g]))
        for g in ("A", "B")
    }  # fmt: skip
    seen = proportion(sum(r["q1_evenement"] in {"hypopnée", "apnée"} for r in by_group["C"]), len(by_group["C"]))
    poor = {g: proportion(sum(r["q3_qualite"] != "bonne" for r in rows), len(rows)) for g, rows in by_group.items()}
    return {
        "counts": counts,
        "clear_hypopnea_criteria_met": clear,
        "event_seen_in_false_alarms": seen,
        "poor_signal": poor,
        "reviewers": sorted({r.get("relecteur", "") for r in answers.values()}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    if file_sha256(args.plan) != sample["plan_sha256"]:
        raise SystemExit("Le plan ne correspond pas à celui du tirage")
    answers = read_answers(args.answers.read_text(encoding="utf-8"), {w["id"] for w in sample["windows"]})
    result = {
        "kind": "hypopnea_review_answers",
        "plan_sha256": sample["plan_sha256"],
        "sample_check": sample_check(answers, file_sha256(args.sample)),
        "source_sha256": {"sample": file_sha256(args.sample), "answers": file_sha256(args.answers)},
        **summarize(sample["windows"], answers),
    }
    write_json_atomic(args.out, result)
    c, s = result["clear_hypopnea_criteria_met"], result["event_seen_in_false_alarms"]
    print(f"Hypopnée nette, critères remplis : A {c['A']['n']}/{c['A']['total']}, B {c['B']['n']}/{c['B']['total']}")
    print(f"Événement vu dans les fausses alertes (C) : {s['n']}/{s['total']}")
    print(
        "Signal de mauvaise qualité : "
        + ", ".join(f"{g} {p['n']}/{p['total']}" for g, p in result["poor_signal"].items())
    )


if __name__ == "__main__":
    main()
