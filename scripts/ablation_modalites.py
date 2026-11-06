"""Que perd-on en retirant des capteurs ? Ablation des blocs d'embeddings.

Un embedding SleepFM concatène trois blocs de 512 dimensions, issus de trois
encodeurs figés : respiration (CHEST, SaO2, ABD), EEG/EOG (C3, C4, O1, O2, E1)
et ECG. Chaque variante réentraîne la baseline et le BiGRU sur une partie des
blocs, avec le même découpage figé et le même contrôle du cache que
run_baseline.py, sans recalculer les embeddings.

Limites : analyse exploratoire, sur des nuits de test déjà examinées. Les
signaux restent des capteurs médicaux de polysomnographie : le résultat ne
simule ni un smartphone ni un objet connecté.

    python scripts/ablation_modalites.py --embeddings-dir embeddings/physionet2018_v2 \
        --out results/ablation_modalites_physionet2018_500.predictions.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import EMBEDDING_DIM, load_verified_split  # noqa: E402
from src.data.cache_contract import file_sha256  # noqa: E402
from src.models.linear_probe import (  # noqa: E402
    bootstrap_ci,
    paired_bootstrap_diff,
    predict_labels,
    predict_probabilities,
    train_probe,
)
from src.models.probe_export import run_environment  # noqa: E402

BLOCK_DIM = EMBEDDING_DIM // 3
# Même ordre que SleepFMEncoders.embed_combined.
BLOCKS = {"resp": 0, "eeg": 1, "ecg": 2}
VARIANTS = {
    "complet": ("resp", "eeg", "ecg"),
    "sans_ecg": ("resp", "eeg"),
    "sans_resp": ("eeg", "ecg"),
    "eeg": ("eeg",),
    "sans_eeg": ("resp", "ecg"),
    "resp": ("resp",),
    "ecg": ("ecg",),
}
LABELS = {
    "complet": "Respiration + EEG/EOG + ECG",
    "sans_ecg": "Respiration + EEG/EOG",
    "sans_resp": "EEG/EOG + ECG",
    "eeg": "EEG/EOG seul",
    "sans_eeg": "Respiration + ECG (sans EEG/EOG)",
    "resp": "Respiration seule",
    "ecg": "ECG seul",
}
PROBES = {"logistic": "Baseline", "sequence": "BiGRU"}


def select_blocks(X: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
    """Colonnes des blocs demandés, toujours dans l'ordre du vecteur complet."""
    if X.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings de dimension {X.shape[1]}, {EMBEDDING_DIM} attendus")
    starts = sorted(BLOCKS[block] * BLOCK_DIM for block in blocks)
    return np.concatenate([X[:, start : start + BLOCK_DIM] for start in starts], axis=1)


def save_archive(path: Path, arrays: dict, metadata: dict) -> None:
    """Archive écrite sous un nom temporaire puis renommée. `metadata["status"]`
    dit si le run est terminé : une archive partielle ne se présente pas comme finie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays, metadata=np.array(json.dumps(metadata, ensure_ascii=False, default=str)))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("third_party/sleepfm/sleepfm/checkpoint/best.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument(
        "--train-nights",
        type=int,
        default=None,
        help="n'entraîner que sur les N premières nuits du découpage figé (effet du volume, même test)",
    )
    parser.add_argument("--out", type=Path, default=None, help="prédictions de chaque variante (.npz)")
    args = parser.parse_args()
    if "complet" not in args.variants:
        raise SystemExit("La variante « complet » sert de référence : elle est obligatoire")

    (X_train, y_train, train_groups), (X_test, y_test, test_groups) = load_verified_split(
        args.embeddings_dir, args.split_file, args.checkpoint
    )
    if args.train_nights is not None:
        kept = json.loads(args.split_file.read_text())["train"][: args.train_nights]
        if len(kept) != args.train_nights:
            raise SystemExit(f"{len(kept)} nuits d'entraînement disponibles, {args.train_nights} demandées")
        rows = np.isin(train_groups[0], kept)
        X_train, y_train, train_groups = X_train[rows], y_train[rows], tuple(g[rows] for g in train_groups)
    subjects = test_groups[0]
    print(f"{len(set(train_groups[0]))} nuits train / {len(set(subjects))} test, {len(y_test)} époques de test")
    print(f"Graine d'entraînement {args.seed} ; écart à la variante complète de la même sonde, apparié par sujet")

    saved = {"y_true": y_test, "subject": subjects, "epoch_idx": test_groups[1], "onset_s": test_groups[2]}
    expected_runs = [f"{probe}_{name}" for name in args.variants for probe in PROBES]
    metadata = {
        "status": "en cours",
        "seed": args.seed,
        "device": args.device,
        "variants": {name: VARIANTS[name] for name in args.variants},
        "expected_runs": expected_runs,
        "completed_runs": [],
        "train_nights": args.train_nights,
        "train_subjects": sorted(set(train_groups[0].tolist())),
        "test_subjects": sorted(set(subjects.tolist())),
        "split_sha256": file_sha256(args.split_file),
        "embeddings_manifest": json.loads((args.embeddings_dir / "manifest.json").read_text()),
        "environment": run_environment(),
        "probes": {},
    }
    reference = {}
    for name in ["complet", *(v for v in args.variants if v != "complet")]:
        train, test = select_blocks(X_train, VARIANTS[name]), select_blocks(X_test, VARIANTS[name])
        print("-" * 104)
        print(LABELS[name])
        for probe, label in PROBES.items():
            start = time.time()
            groups = train_groups if probe == "sequence" else None
            model = train_probe(probe, train, y_train, seed=args.seed, device=args.device, groups=groups)
            pred = predict_labels(model, test, test_groups)
            proba = predict_probabilities(model, test, test_groups)
            saved[f"pred_{probe}_{name}"] = pred
            saved[f"proba_{probe}_{name}"] = proba.astype(np.float32)
            saved[f"classes_{probe}_{name}"] = model.classes_
            reference.setdefault(probe, pred)
            low, high = bootstrap_ci(y_test, pred, subjects)["cohen_kappa"]
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            line = (
                f"  {label:<9} κ {cohen_kappa_score(y_test, pred):.3f} [{low:.3f} ; {high:.3f}]"
                f"  F1 macro {f1_score(y_test, pred, average='macro'):.3f}  ROC-AUC {auc:.3f}"
            )
            if name != "complet":
                diff = paired_bootstrap_diff(y_test, reference[probe], pred, subjects, seed=args.seed)["cohen_kappa"]
                line += f"  écart κ {diff['diff']:+.3f} [{diff['ci95'][0]:+.3f} ; {diff['ci95'][1]:+.3f}]"
            print(f"{line}  ({time.time() - start:.0f} s)", flush=True)
            metadata["completed_runs"].append(f"{probe}_{name}")
            metadata["probes"][f"{probe}_{name}"] = {
                "hyperparameters": model.get_params(),
                "fit": getattr(model, "fit_info_", None),
            }
            if len(metadata["completed_runs"]) == len(expected_runs):
                metadata["status"] = "terminé"
            if args.out is not None:
                save_archive(args.out, saved, metadata)


if __name__ == "__main__":
    main()
