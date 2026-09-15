"""Bout en bout : époques prétraitées -> embeddings SleepFM figés -> sonde
logistique -> métriques comparées au chiffre publié.

Nécessite : data/processed/{X,Y} (via src/data/extract_physionet2018.py) et
un checkpoint SleepFM (third_party/sleepfm/checkpoints/best.pt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.epoch_dataset import PhysioNet2018EpochDataset
from src.models.embeddings import SleepFMEncoders
from src.models.linear_probe import evaluate_probe, train_linear_probe


def collect_embeddings(encoders: SleepFMEncoders, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for epochs, labels in loader:
            embeddings = encoders.embed_combined(epochs.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-train", required=True)
    parser.add_argument("--processed-test", required=True)
    parser.add_argument("--checkpoint", default="third_party/sleepfm/checkpoints/best.pt")
    parser.add_argument("--published-kappa", type=float, default=None, help="Chiffre publié à comparer, si connu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = SleepFMEncoders(args.checkpoint).to(device)

    train_loader = DataLoader(PhysioNet2018EpochDataset(args.processed_train), batch_size=args.batch_size)
    test_loader = DataLoader(PhysioNet2018EpochDataset(args.processed_test), batch_size=args.batch_size)

    print("Génération des embeddings (train)...")
    X_train, y_train = collect_embeddings(encoders, train_loader, device)
    print("Génération des embeddings (test)...")
    X_test, y_test = collect_embeddings(encoders, test_loader, device)

    print("Entraînement de la sonde logistique...")
    probe = train_linear_probe(X_train, y_train)
    results = evaluate_probe(probe, X_test, y_test)

    print(f"Accuracy: {results['accuracy']:.4f} | Cohen kappa: {results['cohen_kappa']:.4f} | Macro F1: {results['macro_f1']:.4f}")
    if args.published_kappa is not None:
        from src.models.linear_probe import compare_to_published

        compare_to_published(results, args.published_kappa)


if __name__ == "__main__":
    main()
