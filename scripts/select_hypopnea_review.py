"""Tire les 40 fenêtres de test à faire relire par le médecin (plan du 17 novembre, validé).

Groupes : A = hypopnées seules manquées par OSF (20), B = hypopnées seules détectées (10),
C = fausses alertes OSF sans événement annoté (10). Au plus une fenêtre par nuit dans chaque
groupe : nuits tirées d'abord, puis une fenêtre par nuit. Ordre d'affichage mélangé. Graine 42,
tirage unique : refuse d'écraser un échantillon existant.

    python scripts/select_hypopnea_review.py --plan "docs/plans/Plan examen hypopnées médecin 17-11.md" \\
        --out results/hypopnea_review_sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402

PREDICTIONS = Path("results/respiratory_osf_vs_sleepfm_physionet2018.predictions.npz")
GROUPS = Path("results/respiratory_errors_physionet2018.groups.npz")
THRESHOLDS = Path("results/respiratory_thresholds_physionet2018.json")
SIZES = {"A": 20, "B": 10, "C": 10}
SEED = 42


def eligible(group: str, event_group: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    hypopnea = event_group == "resp_hypopnea"
    masks = {"A": hypopnea & ~flagged, "B": hypopnea & flagged, "C": (event_group == "aucun") & flagged}
    return np.flatnonzero(masks[group])


def draw(subject: np.ndarray, event_group: np.ndarray, flagged: np.ndarray, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    picks = []
    for group, size in SIZES.items():
        rows = eligible(group, event_group, flagged)
        nights = sorted(set(subject[rows].tolist()))
        if len(nights) < size:
            raise ValueError(f"Groupe {group} : {len(nights)} nuits éligibles pour {size} fenêtres")
        for night in rng.choice(nights, size=size, replace=False):
            candidates = rows[subject[rows] == night]
            picks.append({"group": group, "row": int(rng.choice(candidates))})
    for position, index in enumerate(rng.permutation(len(picks))):
        picks[index]["order"] = position + 1
    return sorted(picks, key=lambda p: p["order"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} existe déjà : le tirage est unique")
    threshold = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["results"]["osf"]["initial_f1"]["threshold"]
    with np.load(PREDICTIONS) as p, np.load(GROUPS) as g:
        if not (np.array_equal(p["subject"], g["subject"]) and np.array_equal(p["onset_s"], g["onset_s"])):
            raise SystemExit("Prédictions et groupes ne sont pas alignés")
        subject, onset, epoch = p["subject"], p["onset_s"], p["epoch_idx"]
        probability, event_group = p["probability_osf"], g["event_group"]
    picks = draw(subject, event_group, probability >= threshold)
    windows = [
        {
            "id": f"F{p['order']:02d}",
            "order": p["order"],
            "group": p["group"],
            "subject": str(subject[p["row"]]),
            "epoch_idx": int(epoch[p["row"]]),
            "onset_s": float(onset[p["row"]]),
            "probability_osf": float(probability[p["row"]]),
        }
        for p in picks
    ]
    write_json_atomic(
        args.out,
        {
            "kind": "hypopnea_review_sample",
            "plan_sha256": file_sha256(args.plan),
            "source_sha256": {str(f): file_sha256(f) for f in (PREDICTIONS, GROUPS, THRESHOLDS)},
            "seed": SEED,
            "sizes": SIZES,
            "osf_threshold": threshold,
            "windows": windows,
        },
    )
    counts = {g: sum(w["group"] == g for w in windows) for g in SIZES}
    print(f"{len(windows)} fenêtres, {len({w['subject'] for w in windows})} nuits ; {counts} -> {args.out}")


if __name__ == "__main__":
    main()
