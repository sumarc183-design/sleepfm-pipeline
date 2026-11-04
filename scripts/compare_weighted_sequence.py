"""BiGRU avec perte pondérée par stade, contre le BiGRU et la baseline officielle.

Réentraîne seulement le BiGRU pondéré (class_weight="balanced") et réutilise
les prédictions sauvegardées de la baseline et du BiGRU, sur les mêmes nuits
de test. Écarts appariés par bootstrap sur les sujets.

    python scripts/compare_weighted_sequence.py \
        --embeddings-dir embeddings/physionet2018 \
        --predictions results/results_physionet2018_100_auprc.predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings, load_or_extend_split  # noqa: E402
from src.models.linear_probe import paired_bootstrap_diff  # noqa: E402
from src.models.sequence_probe import SequenceProbe  # noqa: E402


def describe(name: str, y: np.ndarray, pred: np.ndarray) -> str:
    per_stage = " ".join(f"{v:.2f}" for v in f1_score(y, pred, average=None))
    n1_recall = recall_score(y, pred, labels=[1], average="macro")
    n1_precision = precision_score(y, pred, labels=[1], average="macro")
    return (
        f"{name:<15} κ {cohen_kappa_score(y, pred):.3f}  F1 {f1_score(y, pred, average='macro'):.3f}  "
        f"F1/stade {per_stage}  N1 rappel {n1_recall:.2f} précision {n1_precision:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects = sorted(p.stem for p in args.embeddings_dir.glob("*.npz"))
    train, test = load_or_extend_split(args.split_file, subjects, 0.2, args.seed)
    saved = np.load(args.predictions)
    order = list(dict.fromkeys(saved["subject"]))  # ordre des nuits dans les prédictions sauvegardées
    if sorted(order) != sorted(test):
        raise SystemExit("Les prédictions sauvegardées ne portent pas sur les nuits de test du découpage figé")
    X_train, y_train, train_groups = load_cached_embeddings(args.embeddings_dir, train)
    X_test, y_test, test_groups = load_cached_embeddings(args.embeddings_dir, order)
    assert np.array_equal(saved["y_true"], y_test) and np.array_equal(saved["epoch_idx"], test_groups[1])

    baseline, plain = saved["pred_logistic"], saved["pred_sequence"]
    model = SequenceProbe(seed=args.seed, class_weight="balanced").fit(X_train, y_train, train_groups)
    weighted = model.predict(X_test, test_groups)

    print("F1/stade : Wake N1 N2 N3 REM ; écarts appariés par bootstrap sur les sujets (IC 95 %)")
    for name, pred in [("Baseline", baseline), ("BiGRU", plain), ("BiGRU pondéré", weighted)]:
        print(describe(name, y_test, pred))
    for ref_name, ref in [("baseline", baseline), ("BiGRU", plain)]:
        d = paired_bootstrap_diff(y_test, ref, weighted, test_groups[0], seed=args.seed)
        k, f = d["cohen_kappa"], d["macro_f1"]
        print(
            f"Pondéré - {ref_name} : κ {k['diff']:+.3f} [{k['ci95'][0]:+.3f} ; {k['ci95'][1]:+.3f}]   "
            f"F1 macro {f['diff']:+.3f} [{f['ci95'][0]:+.3f} ; {f['ci95'][1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
