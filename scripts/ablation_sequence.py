"""Ablation du contexte temporel du BiGRU, sur les embeddings en cache.

D'où vient le gain du BiGRU sur la baseline ? On garde le même découpage
train/test que run_baseline.py et on fait varier le contexte :

- à la prédiction (un seul modèle entraîné, sur des fenêtres d'1 h) : nuit
  entière, blocs d'1 h, de 5 min, puis d'une seule époque ;
- à l'entraînement et à la prédiction : ordre des époques mélangé dans
  chaque nuit, ce qui garde les mêmes époques mais détruit la chronologie.

Si le gain disparaît avec l'ordre mélangé ou un contexte d'une époque, il
vient bien du contexte de la nuit, pas de la tête récurrente elle-même.
Chaque variante est comparée à la baseline officielle par bootstrap apparié
sur les mêmes nuits de test.

    python scripts/ablation_sequence.py --embeddings-dir embeddings/physionet2018
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings, split_subject_ids  # noqa: E402
from src.models.linear_probe import paired_bootstrap_diff, train_probe  # noqa: E402
from src.models.sequence_probe import SequenceProbe  # noqa: E402


def shuffle_within_nights(groups: tuple[np.ndarray, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mêmes époques, numéros d'époque permutés au hasard dans chaque nuit."""
    subjects, epoch_idx = groups
    rng = np.random.default_rng(seed)
    shuffled = epoch_idx.copy()
    for subject in dict.fromkeys(subjects):
        rows = np.flatnonzero(subjects == subject)
        shuffled[rows] = rng.permutation(epoch_idx[rows])
    return subjects, shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    subjects = sorted(p.stem for p in args.embeddings_dir.glob("*.npz") if not p.stem.endswith(".tmp"))
    train_subjects, test_subjects = split_subject_ids(subjects, args.test_fraction, args.seed)
    X_train, y_train, train_groups = load_cached_embeddings(args.embeddings_dir, train_subjects)
    X_test, y_test, test_groups = load_cached_embeddings(args.embeddings_dir, test_subjects)
    print(f"{len(train_subjects)} nuits train / {len(test_subjects)} test, {len(y_train)} / {len(y_test)} époques")

    baseline = train_probe("logistic", X_train, y_train, seed=args.seed).predict(X_test)

    model = SequenceProbe(device=args.device, seed=args.seed).fit(X_train, y_train, train_groups)
    predictions = {}
    for label, context in [
        ("nuit entière", None),
        ("blocs d'1 h", 120),
        ("blocs de 5 min", 10),
        ("1 époque", 1),
    ]:
        model.context = context
        predictions[f"Contexte à la prédiction : {label}"] = model.predict(X_test, test_groups)

    shuffled = SequenceProbe(device=args.device, seed=args.seed).fit(
        X_train, y_train, shuffle_within_nights(train_groups, args.seed)
    )
    predictions["Ordre mélangé (entraînement et test)"] = shuffled.predict(
        X_test, shuffle_within_nights(test_groups, args.seed + 1)
    )

    subjects_test = test_groups[0]
    print("-" * 104)
    print(f"{'Variante':<44}{'Kappa':>7}{'F1 macro':>10}{'F1 N1':>7}   écart κ à la baseline [IC 95 %]")
    print(
        f"{'Baseline officielle (logistique pondérée)':<44}{cohen_kappa_score(y_test, baseline):>7.3f}"
        f"{f1_score(y_test, baseline, average='macro'):>10.3f}"
        f"{f1_score(y_test, baseline, labels=[1], average='macro'):>7.3f}"
    )
    for label, pred in predictions.items():
        diff = paired_bootstrap_diff(y_test, baseline, pred, subjects_test, seed=args.seed)["cohen_kappa"]
        verdict = "significatif" if diff["ci95"][0] > 0 or diff["ci95"][1] < 0 else "non significatif"
        print(
            f"{label:<44}{cohen_kappa_score(y_test, pred):>7.3f}{f1_score(y_test, pred, average='macro'):>10.3f}"
            f"{f1_score(y_test, pred, labels=[1], average='macro'):>7.3f}   {diff['diff']:+.3f} "
            f"[{diff['ci95'][0]:+.3f} ; {diff['ci95'][1]:+.3f}] {verdict}"
        )


if __name__ == "__main__":
    main()
