"""Prédit après enregistrement sur une PSG PhysioNet 2018 sans annotations.

Lit uniquement <sujet>.hea et <sujet>.mat dans --raw-data/<sujet>/.
Produit un CSV et un JSON de provenance par nuit. Aucun entraînement,
téléchargement ni modification des caches ou modèles existants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import EMBEDDING_DIM, embeddings_manifest
from scripts.predict_night import write_predictions
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.raw_night import RawNightDataset
from src.data.subject_ids import validate_subject_ids
from src.models.embeddings import SleepFMEncoders
from src.models.probe_export import load_probes, run_environment


@torch.no_grad()
def predict_raw_subject(
    subject_dir: Path, models: dict, encoders, out: Path, batch_size: int, device: str, provenance: dict
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size doit être positif")
    dataset = RawNightDataset(subject_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    X = np.concatenate([encoders.embed_combined(epochs.to(device)).cpu().numpy() for epochs in loader])
    if X.shape != (len(dataset), EMBEDDING_DIM) or not np.isfinite(X).all():
        raise ValueError("Embeddings invalides")
    groups = (np.full(len(dataset), dataset.subject), dataset.epoch_idx, dataset.onset_s)
    path = out / f"{dataset.subject}.csv"
    write_predictions(models, X, groups, path)
    write_json_atomic(
        path.with_suffix(".json"),
        provenance
        | dataset.metadata()
        | {
            "source_files": {p.name: file_sha256(p) for p in (dataset.header_path, dataset.signal_path)},
            "csv_sha256": file_sha256(path),
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size doit être positif")
    try:
        validate_subject_ids(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))
    contract = embeddings_manifest(args.checkpoint)
    # Même checkpoint et traitement du signal, sans contournement du contrat.
    # Le découpage indépendant des annotations reste tracé dans le JSON de sortie.
    models, export = load_probes(args.models, contract)
    if export["input"]["embedding_dim"] != EMBEDDING_DIM:
        raise ValueError(f"Dimension d'export incompatible : attendu {EMBEDDING_DIM}")
    encoders = SleepFMEncoders(args.checkpoint).to(args.device).eval()
    print("Fenêtres fixes depuis zéro ; performances sur fenêtres annotées non transposables sans évaluation.")
    provenance = {
        "signal_contract": contract,
        "models_manifest_sha256": file_sha256(args.models / "manifest.json"),
        "model_files": {info["file"]: file_sha256(args.models / info["file"]) for info in export["probes"].values()},
        "environment": run_environment(),
    }
    for subject in args.subjects:
        path = predict_raw_subject(
            args.raw_data / subject, models, encoders, args.out, args.batch_size, args.device, provenance
        )
        print(f"{subject} -> {path}")


if __name__ == "__main__":
    main()
