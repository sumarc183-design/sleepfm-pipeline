"""Prédit l'hypnogramme d'une ou plusieurs nuits avec des sondes exportées.

Lit des sondes exportées par `run_baseline.py --save-models DIR` et des
embeddings en cache (`<sujet>.npz` du mode par lots, avec ou sans stades),
exige le même contrat (checkpoint, prétraitement, canaux, fréquence, dimension),
puis écrit un CSV par nuit : numéro d'époque, début (s), stade prédit et
probabilités, pour chaque sonde.

Le cache doit déjà exister : l'extraction annotée choisit ses fenêtres à partir
des annotations. Pour une PSG brute, utiliser scripts/predict_raw_night.py.
Le BiGRU lit aussi les époques futures : traitement après enregistrement.

    python scripts/predict_night.py --models models/physionet2018_500 \
        --embeddings-dir embeddings/physionet2018_v2 tr03-0005 --out predictions/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402
from src.data.subject_ids import validate_subject_ids
from src.models.linear_probe import predict_probabilities  # noqa: E402
from src.models.probe_export import load_probes  # noqa: E402

LABEL_TO_STAGE = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}


def write_predictions(models: dict, X: np.ndarray, groups: tuple, path: Path) -> None:
    """Écrit un CSV complet par renommage ; une erreur conserve l'ancien CSV."""
    order = np.argsort(groups[1])
    columns = {"epoch_idx": groups[1][order], "onset_s": groups[2][order]}
    for kind, model in models.items():
        proba = predict_probabilities(model, X, groups)[order]
        if (
            proba.shape != (len(X), len(model.classes_))
            or not np.isfinite(proba).all()
            or (proba < 0).any()
            or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{kind} : probabilités invalides")
        columns[f"{kind}_stade"] = [LABEL_TO_STAGE[int(c)] for c in model.classes_[proba.argmax(1)]]
        for j, c in enumerate(model.classes_):
            columns[f"{kind}_p_{LABEL_TO_STAGE[int(c)]}"] = np.round(proba[:, j], 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values(), strict=True))
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-unverified-provenance",
        action="store_true",
        help="Cache historique sans contrat complet : prédit quand même, avec un avertissement.",
    )
    args = parser.parse_args()

    try:
        validate_subject_ids(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))
    manifest_path = args.embeddings_dir / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    models, export = load_probes(args.models, cache_manifest, args.allow_unverified_provenance)
    expected_dim = export["input"]["embedding_dim"]
    args.out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        X, _, groups = load_cached_embeddings(args.embeddings_dir, [subject], require_labels=False)
        if X.shape[1] != expected_dim:
            raise SystemExit(f"{subject} : embeddings de dimension {X.shape[1]}, {expected_dim} à l'entraînement")
        path = args.out / f"{subject}.csv"
        write_predictions(models, X, groups, path)
        print(f"{subject} : {len(X)} époques -> {path}")


if __name__ == "__main__":
    main()
