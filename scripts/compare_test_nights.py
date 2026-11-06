"""Le score baisse-t-il parce que le modèle est moins bon, ou parce que les
nouvelles nuits de test sont plus difficiles ?

Quand le découpage figé est étendu, les anciennes nuits de test gardent leur
rôle. À partir des prédictions sauvegardées d'un run, compare le κ de la
baseline et du BiGRU sur les anciennes nuits de test, les nouvelles et
l'ensemble, avec IC à 95 % par bootstrap sur les sujets ; liste les nuits au
κ le plus bas avec leur répartition des stades (selon l'expert).

Le script vérifie si les anciennes nuits ont exactement les mêmes époques, aux
mêmes temps d'origine, et les mêmes stades dans les deux runs. Si oui, il donne
l'écart apparié entre les deux runs, pour une graine d'entraînement chacun. Sinon,
ou si les temps manquent, la comparaison reste descriptive : l'identité des entrées
n'est pas établie.

    python scripts/compare_test_nights.py \
        --predictions results/results_physionet2018_200.predictions.npz \
        --old-predictions results/results_physionet2018_100_auprc.predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.linear_probe import bootstrap_ci, paired_bootstrap_diff  # noqa: E402

PROBES = {"logistic": "Baseline", "sequence": "BiGRU"}
ONSET_TOLERANCE_S = 0.01
STAGES = "W/N1/N2/N3/REM"


def kappa_with_ci(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> str:
    low, high = bootstrap_ci(y, pred, subjects)["cohen_kappa"]
    return f"{cohen_kappa_score(y, pred):.3f} [{low:.3f} ; {high:.3f}]"


def sorted_rows(subjects: np.ndarray, epoch_idx: np.ndarray) -> np.ndarray:
    return np.lexsort((epoch_idx, subjects))


def same_epochs_problem(old, new, rows_old: np.ndarray, rows_new: np.ndarray) -> str | None:
    """Pourquoi les époques des deux runs ne sont pas établies identiques, ou None."""
    if len(rows_old) != len(rows_new):
        return f"{len(rows_old)} époques contre {len(rows_new)}"
    keys = len({(s, i) for s, i in zip(old["subject"][rows_old], old["epoch_idx"][rows_old], strict=True)})
    if keys != len(rows_old):
        return "couples sujet / époque en double"
    if not (
        np.array_equal(old["subject"][rows_old], new["subject"][rows_new])
        and np.array_equal(old["epoch_idx"][rows_old], new["epoch_idx"][rows_new])
    ):
        return "sujets ou numéros d'époque différents"
    if not np.array_equal(old["y_true"][rows_old], new["y_true"][rows_new]):
        return "stades différents"
    if "onset_s" not in old.files or "onset_s" not in new.files:
        return "temps d'origine absents d'une archive"
    onsets_old, onsets_new = old["onset_s"][rows_old], new["onset_s"][rows_new]
    if not (np.isfinite(onsets_old).all() and np.isfinite(onsets_new).all()):
        return "temps d'origine manquants"
    if (np.abs(onsets_old - onsets_new) > ONSET_TOLERANCE_S).any():
        return "temps d'origine différents"
    return None


def compare_same_epochs(old, new, is_old: np.ndarray) -> None:
    """Écart apparié entre les deux runs, seulement si les époques sont identiques."""
    rows_old = sorted_rows(old["subject"], old["epoch_idx"])
    rows_new = np.flatnonzero(is_old)[sorted_rows(new["subject"][is_old], new["epoch_idx"][is_old])]
    problem = same_epochs_problem(old, new, rows_old, rows_new)
    if problem:
        print(
            f"Identité des anciennes nuits non établie entre les runs ({problem}) : "
            "comparaison descriptive, pas d'écart attribuable au seul entraînement"
        )
        return
    print(f"Époques, temps et stades des anciennes nuits identiques entre les runs ({len(rows_old)} époques)")
    y, subjects = old["y_true"][rows_old], old["subject"][rows_old]
    for probe, name in PROBES.items():
        diff = paired_bootstrap_diff(
            y, old[f"pred_{probe}"][rows_old], new[f"pred_{probe}"][rows_new], subjects, seed=42
        )["cohen_kappa"]
        print(
            f"  {name} : κ ce run moins run précédent {diff['diff']:+.3f} "
            f"[{diff['ci95'][0]:+.3f} ; {diff['ci95'][1]:+.3f}] (une graine par run)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="prédictions du run étendu")
    parser.add_argument("--old-predictions", type=Path, required=True, help="prédictions du run précédent")
    parser.add_argument("--worst", type=int, default=6, help="nombre de nuits les moins bien classées à lister")
    args = parser.parse_args()

    new, old = np.load(args.predictions), np.load(args.old_predictions)
    old_test = set(old["subject"])
    is_old = np.isin(new["subject"], list(old_test))
    n_old, n_new = len(old_test), len(set(new["subject"])) - len(old_test)

    print("κ [IC 95 %, bootstrap sur les sujets]")
    print(f"{'Nuits de test':<36}{'époques':>8}" + "".join(f"{name:>26}" for name in PROBES.values()))
    print(
        f"{f'{n_old} anciennes (run précédent)':<36}{len(old['y_true']):>8}"
        + "".join(f"{kappa_with_ci(old['y_true'], old[f'pred_{p}'], old['subject']):>26}" for p in PROBES)
    )
    for label, mask in [
        (f"{n_old} anciennes (ce run)", is_old),
        (f"{n_new} nouvelles (ce run)", ~is_old),
        (f"{n_old + n_new} (ce run)", np.ones_like(is_old)),
    ]:
        y, subjects = new["y_true"][mask], new["subject"][mask]
        print(
            f"{label:<36}{mask.sum():>8}"
            + "".join(f"{kappa_with_ci(y, new[f'pred_{p}'][mask], subjects):>26}" for p in PROBES)
        )

    compare_same_epochs(old, new, is_old)

    nights = []
    for subject in sorted(set(new["subject"])):
        m = new["subject"] == subject
        y = new["y_true"][m]
        kappas = [cohen_kappa_score(y, new[f"pred_{p}"][m]) for p in PROBES]
        share = np.bincount(y, minlength=5) / m.sum()
        nights.append((kappas[0], kappas[1], subject, "ancienne" if subject in old_test else "nouvelle", share))
    nights.sort()
    for status in ("ancienne", "nouvelle"):
        median = np.median([n[0] for n in nights if n[3] == status])
        print(f"κ médian par nuit, baseline, nuits {status}s : {median:.3f}")
    wins = sum(n[1] > n[0] for n in nights)
    print(f"BiGRU au-dessus de la baseline sur {wins} nuits sur {len(nights)}")

    print(f"\nNuits au κ le plus bas (part des stades selon l'expert, {STAGES})")
    for k_base, k_seq, subject, status, share in nights[: args.worst]:
        shares = " ".join(f"{x:.2f}" for x in share)
        print(f"  {subject}  {status:<8}  baseline {k_base:.3f}  BiGRU {k_seq:.3f}  {shares}")


if __name__ == "__main__":
    main()
