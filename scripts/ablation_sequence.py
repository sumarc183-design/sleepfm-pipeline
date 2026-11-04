"""Ablation du contexte temporel du BiGRU, sur les embeddings en cache.

D'où vient le gain du BiGRU sur la baseline ? Même découpage figé et même
contrôle de provenance du cache que run_baseline.py, puis, pour plusieurs
graines :

- contexte à la prédiction (un modèle entraîné sur des fenêtres d'1 h) :
  nuit entière, blocs d'1 h, de 5 min, d'une époque. « 1 époque » n'est pas
  un modèle sans contexte équitable : le réseau a appris sur des séquences ;
- ordre des époques mélangé dans chaque nuit, à l'entraînement et au test :
  mêmes époques, chronologie détruite.

Chaque variante est comparée à la baseline officielle, et l'ordre réel à
l'ordre mélangé directement, par bootstrap apparié sur les mêmes nuits de
test. Les prédictions de chaque variante sont enregistrées (--out).

    python scripts/ablation_sequence.py --embeddings-dir embeddings/physionet2018 \
        --out results/ablation_sequence_physionet2018_100.predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_verified_split  # noqa: E402
from src.models.linear_probe import paired_bootstrap_diff, train_probe  # noqa: E402
from src.models.sequence_probe import SequenceProbe  # noqa: E402

CONTEXTS = [("nuit", None), ("1h", 120), ("5min", 10), ("1epoque", 1)]
LABELS = {
    "nuit": "Contexte à la prédiction : nuit entière",
    "1h": "Contexte à la prédiction : blocs d'1 h",
    "5min": "Contexte à la prédiction : blocs de 5 min",
    "1epoque": "Contexte à la prédiction : 1 époque",
    "melange": "Ordre mélangé (entraînement et test)",
}


def shuffle_within_nights(groups: tuple, seed: int) -> tuple:
    """Mêmes époques, numéros d'époque permutés au hasard dans chaque nuit
    (les temps d'origine sont retirés : ils trahiraient l'ordre)."""
    subjects, epoch_idx = groups[0], groups[1]
    rng = np.random.default_rng(seed)
    shuffled = epoch_idx.copy()
    for subject in dict.fromkeys(subjects):
        rows = np.flatnonzero(subjects == subject)
        shuffled[rows] = rng.permutation(epoch_idx[rows])
    return subjects, shuffled


def ci(d: dict) -> str:
    return f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f} ; {d['ci95'][1]:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("third_party/sleepfm/sleepfm/checkpoint/best.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="prédictions de chaque variante (.npz)")
    args = parser.parse_args()

    (X_train, y_train, train_groups), (X_test, y_test, test_groups) = load_verified_split(
        args.embeddings_dir, args.split_file, args.checkpoint
    )
    subjects_test = test_groups[0]
    print(f"{len(set(train_groups[0]))} nuits train / {len(set(subjects_test))} test, {len(y_test)} époques de test")

    baseline = train_probe("logistic", X_train, y_train, seed=42).predict(X_test)
    saved = {"y_true": y_test, "subject": subjects_test, "epoch_idx": test_groups[1], "pred_logistic": baseline}
    width = 112
    print("=" * width)
    print(
        f"Baseline officielle (logistique pondérée) : κ {cohen_kappa_score(y_test, baseline):.3f}, "
        f"F1 macro {f1_score(y_test, baseline, average='macro'):.3f}, "
        f"F1 N1 {f1_score(y_test, baseline, labels=[1], average='macro'):.3f}"
    )
    for seed in args.seeds:
        model = SequenceProbe(device=args.device, seed=seed).fit(X_train, y_train, train_groups)
        preds = {}
        for key, context in CONTEXTS:
            model.context = context
            preds[key] = model.predict(X_test, test_groups)
        shuffled_model = SequenceProbe(device=args.device, seed=seed).fit(
            X_train, y_train, shuffle_within_nights(train_groups, seed)
        )
        preds["melange"] = shuffled_model.predict(X_test, shuffle_within_nights(test_groups, seed + 1000))

        print("-" * width)
        print(f"Graine {seed}")
        print(f"  {'Variante':<44}{'Kappa':>7}{'F1 macro':>10}{'F1 N1':>7}   écart κ à la baseline [IC 95 %]")
        for key, pred in preds.items():
            d = paired_bootstrap_diff(y_test, baseline, pred, subjects_test, seed=seed)["cohen_kappa"]
            print(
                f"  {LABELS[key]:<44}{cohen_kappa_score(y_test, pred):>7.3f}"
                f"{f1_score(y_test, pred, average='macro'):>10.3f}"
                f"{f1_score(y_test, pred, labels=[1], average='macro'):>7.3f}   {ci(d)}"
            )
            saved[f"pred_{key}_seed{seed}"] = pred
        direct = paired_bootstrap_diff(y_test, preds["melange"], preds["nuit"], subjects_test, seed=seed)
        print(
            f"  Ordre réel moins ordre mélangé (nuit entière) : κ {ci(direct['cohen_kappa'])}, "
            f"F1 macro {ci(direct['macro_f1'])}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **saved)
        print(f"Prédictions des variantes : {args.out}")
    print("=" * width)


if __name__ == "__main__":
    main()
