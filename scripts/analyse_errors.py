"""Analyse des erreurs de deux sondes à partir des prédictions sauvegardées.

Lit le `.predictions.npz` écrit par run_baseline.py (--results-json), sans
réentraîner, et compare une sonde à une référence (par défaut le BiGRU à la
baseline officielle) :

1. matrices de confusion des deux sondes, côte à côte ;
2. kappa par nuit : où le gain se fait-il, et sur quelles nuits ;
3. précision selon la distance à la transition de stade la plus proche
   (dans l'hypnogramme de l'expert) : les erreurs se concentrent-elles aux
   changements de stade ?
4. N1 en détail : rappel aux transitions et loin d'elles, et stades
   confondus.

    python scripts/analyse_errors.py results/results_physionet2018_100_auprc.predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402

STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
assert set(STAGES) == set(STAGE_NAME_TO_LABEL.values())
DISTANCE_BUCKETS = [
    (0, 0, "à la transition"),
    (1, 2, "1-2 époques"),
    (3, 10, "3-10 époques"),
    (11, None, "> 10 époques"),
]


def nights(subjects: np.ndarray, epoch_idx: np.ndarray) -> list[np.ndarray]:
    return [
        rows[np.argsort(epoch_idx[rows], kind="stable")]
        for rows in (np.flatnonzero(subjects == s) for s in dict.fromkeys(subjects))
    ]


def distance_to_transition(y_night: np.ndarray) -> np.ndarray:
    """Pour chaque époque, nombre d'époques jusqu'au changement de stade le
    plus proche : 0 = première ou dernière époque d'un segment."""
    change = np.flatnonzero(np.diff(y_night) != 0)
    edges = np.union1d(change, change + 1)  # dernière époque avant, première après
    if len(edges) == 0:
        return np.full(len(y_night), len(y_night))
    positions = np.arange(len(y_night))
    return np.abs(positions[:, None] - edges[None, :]).min(axis=1)


def print_confusions(y, preds: dict[str, np.ndarray]) -> None:
    labels = list(STAGES)
    blocks = {name: confusion_matrix(y, p, labels=labels) for name, p in preds.items()}
    print("1. Matrices de confusion (lignes = stade réel, colonnes = stade prédit), " + " | ".join(preds))
    header = "".join(f"{STAGES[c]:>7}" for c in labels)
    print(f"  {'':<6}" + "   |   ".join(header for _ in preds) + f"   {'n':>6}")
    for i, c in enumerate(labels):
        rows = "   |   ".join("".join(f"{v:>7d}" for v in m[i]) for m in blocks.values())
        print(f"  {STAGES[c]:<6}{rows}   {blocks[next(iter(blocks))][i].sum():>6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--reference", default="logistic", help="sonde de référence (défaut : baseline officielle)")
    parser.add_argument("--probe", default="sequence", help="sonde analysée (défaut : BiGRU)")
    args = parser.parse_args()

    data = np.load(args.predictions)
    y, subjects, epoch_idx = data["y_true"], data["subject"], data["epoch_idx"]
    ref, probe = data[f"pred_{args.reference}"], data[f"pred_{args.probe}"]
    names = {args.reference: ref, args.probe: probe}
    width = 100
    print("=" * width)
    print(f"Analyse des erreurs : {args.probe} contre {args.reference} ({len(y)} époques de test)")
    print("-" * width)
    print_confusions(y, names)

    print("-" * width)
    print("2. Kappa par nuit (trié par gain)")
    per_night = []
    distance = np.zeros(len(y), dtype=int)
    for rows in nights(subjects, epoch_idx):
        distance[rows] = distance_to_transition(y[rows])
        k_ref, k_probe = cohen_kappa_score(y[rows], ref[rows]), cohen_kappa_score(y[rows], probe[rows])
        n1 = (y[rows] == 1).mean()
        per_night.append((subjects[rows[0]], len(rows), k_ref, k_probe, n1))
    per_night.sort(key=lambda r: r[3] - r[2], reverse=True)
    print(f"  {'Nuit':<12}{'époques':>8}{args.reference:>12}{args.probe:>12}{'gain':>8}{'part de N1':>12}")
    for subject, n, k_ref, k_probe, n1 in per_night:
        print(f"  {subject:<12}{n:>8d}{k_ref:>12.3f}{k_probe:>12.3f}{k_probe - k_ref:>+8.3f}{n1:>12.1%}")
    gains = np.array([r[3] - r[2] for r in per_night])
    print(
        f"  Gain médian {np.median(gains):+.3f} ; nuits améliorées : {(gains > 0).sum()} sur {len(gains)} ; "
        f"corrélation gain / kappa de référence : {np.corrcoef(gains, [r[2] for r in per_night])[0, 1]:+.2f}"
    )

    print("-" * width)
    print("3. Précision selon la distance à la transition de stade la plus proche (hypnogramme de l'expert)")
    print(f"  {'Distance':<18}{'part':>8}{args.reference:>12}{args.probe:>12}{'gain':>8}")
    for low, high, label in DISTANCE_BUCKETS:
        mask = (distance >= low) & (distance <= (high if high is not None else distance.max()))
        acc_ref, acc_probe = (ref[mask] == y[mask]).mean(), (probe[mask] == y[mask]).mean()
        print(f"  {label:<18}{mask.mean():>8.1%}{acc_ref:>12.3f}{acc_probe:>12.3f}{acc_probe - acc_ref:>+8.3f}")

    print("-" * width)
    print("4. N1 : rappel selon la distance à la transition, et stades prédits à la place")
    is_n1 = y == 1
    print(f"  {'Distance':<18}{'époques N1':>11}{args.reference:>12}{args.probe:>12}")
    for low, high, label in DISTANCE_BUCKETS:
        mask = is_n1 & (distance >= low) & (distance <= (high if high is not None else distance.max()))
        if mask.any():
            print(f"  {label:<18}{mask.sum():>11d}{(ref[mask] == 1).mean():>12.3f}{(probe[mask] == 1).mean():>12.3f}")
    for name, pred in names.items():
        wrong = pred[is_n1 & (pred != 1)]
        counts = ", ".join(f"{STAGES[c]} {(wrong == c).sum()}" for c in STAGES if c != 1)
        print(f"  {name} : N1 pris pour {counts}")
    print("=" * width)


if __name__ == "__main__":
    main()
