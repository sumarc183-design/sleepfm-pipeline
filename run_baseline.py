#!/usr/bin/env python3
"""Point d'entrée unique pour la baseline SleepFM sur PhysioNet 2018.

Enchaîne extraction -> embeddings -> sonde logistique -> évaluation en
réutilisant exclusivement le code déjà testé (src/data/extract_physionet2018.py,
src/models/embeddings.py, src/models/linear_probe.py) : pas de réimplémentation
parallèle.

Sans --raw-data : génère des données synthétiques (comme
tests/test_pipeline_smoke.py) pour un smoke test rapide.
Sans --checkpoint : utilise un checkpoint à poids aléatoires (résultats non
interprétables médicalement, plomberie uniquement -- clairement indiqué en
sortie).

Exemples :
    # Smoke test, tout synthétique
    python run_baseline.py

    # Vos vraies données, checkpoint aléatoire (validation de plomberie)
    python run_baseline.py --raw-data data/raw/physionet2018/training

    # Vos vraies données + vrai checkpoint SleepFM (premier vrai chiffre)
    python run_baseline.py --raw-data data/raw/physionet2018/training \\
        --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.generate_synthetic_physionet2018 import generate_subject
from src.data.epoch_dataset import PhysioNet2018EpochDataset
from src.data.extract_physionet2018 import main as extract_main
from src.models.effnet import EffNet
from src.models.embeddings import CHANNEL_INDICES, SleepFMEncoders
from src.models.linear_probe import evaluate_probe, train_linear_probe


def build_random_checkpoint(path: Path, embedding_dim: int = 512) -> None:
    def build_encoder(in_ch):
        m = EffNet(in_channel=in_ch, stride=2, dilation=1)
        m.fc = torch.nn.Linear(m.fc.in_features, embedding_dim)
        return m

    torch.save(
        {
            "respiratory_state_dict": build_encoder(len(CHANNEL_INDICES["respiratory"])).state_dict(),
            "sleep_stages_state_dict": build_encoder(len(CHANNEL_INDICES["sleep_stages"])).state_dict(),
            "ekg_state_dict": build_encoder(len(CHANNEL_INDICES["ekg"])).state_dict(),
        },
        path,
    )


def generate_synthetic_raw(raw_dir: Path, n_subjects: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_subjects):
        generate_subject(
            str(raw_dir), f"synthetic-{i:04d}", sample_rate=32,
            n_epochs=20, epoch_sec=30, lead_in_sec=2, rng=rng,
        )


def extract(raw_dir: Path, processed_dir: Path, num_threads: int) -> None:
    argv_backup, sys.argv = sys.argv, [
        "extract", "--data_path", str(raw_dir), "--save_path", str(processed_dir), "--num_threads", str(num_threads),
    ]
    try:
        extract_main()
    finally:
        sys.argv = argv_backup


def split_subjects_train_test(raw_dir: Path, test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split au niveau sujet (pas époque) pour éviter toute fuite -- un même
    sujet ne doit jamais apparaître à la fois en train et en test."""
    subjects = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_fraction)) if len(subjects) > 1 else 0
    return subjects[n_test:], subjects[:n_test]


def collect_embeddings(encoders: SleepFMEncoders, processed_dir: Path, batch_size: int = 32,
                       device: str = "cpu", num_workers: int = 0):
    dataset = PhysioNet2018EpochDataset(processed_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings, labels = [], []
    with torch.no_grad():
        for epochs, batch_labels in loader:
            embeddings.append(encoders.embed_combined(epochs.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels), len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-data", type=Path, default=None, help="Dossier des sujets PhysioNet 2018 (.hea/.mat/-arousal.mat). Omis = synthétique.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint SleepFM officiel. Omis = poids aléatoires.")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction de sujets pour le test (split par sujet, jamais par époque)")
    parser.add_argument("--num-synthetic-subjects", type=int, default=6)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda (GPU Colab/Kaggle) ou cpu. Par défaut : cuda si disponible.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    is_synthetic = args.raw_data is None
    is_random_checkpoint = args.checkpoint is None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = args.raw_data
        if is_synthetic:
            raw_dir = tmp_path / "raw"
            print(f"[synthétique] Génération de {args.num_synthetic_subjects} sujets factices...")
            generate_synthetic_raw(raw_dir, args.num_synthetic_subjects, args.seed)

        train_subjects, test_subjects = split_subjects_train_test(raw_dir, args.test_fraction, args.seed)
        if not test_subjects:
            print("ATTENTION : un seul sujet disponible -> pas de split train/test possible.")
            print("Les résultats ci-dessous sont une simple vérification de plomberie, pas une mesure de généralisation.\n")
            train_subjects = test_subjects = train_subjects  # même sujet des deux côtés, assumé et annoncé

        processed_train, processed_test = tmp_path / "train", tmp_path / "test"
        for subset, out_dir in [(train_subjects, processed_train), (test_subjects, processed_test)]:
            subset_raw = tmp_path / f"raw_{out_dir.name}"
            subset_raw.mkdir(parents=True, exist_ok=True)
            for subject in subset:
                target = subset_raw / subject
                if not target.exists():
                    shutil.copytree(raw_dir / subject, target)
            extract(subset_raw, out_dir, args.num_threads)

        checkpoint_path = args.checkpoint
        if is_random_checkpoint:
            checkpoint_path = tmp_path / "random_checkpoint.pt"
            build_random_checkpoint(checkpoint_path)

        encoders = SleepFMEncoders(checkpoint_path, embedding_dim=512).to(args.device)
        print(f"Embeddings sur : {args.device}")
        X_train, y_train, n_train = collect_embeddings(encoders, processed_train, args.batch_size, args.device, args.num_threads)
        X_test, y_test, n_test = collect_embeddings(encoders, processed_test, args.batch_size, args.device, args.num_threads)

        probe = train_linear_probe(X_train, y_train)
        results = evaluate_probe(probe, X_test, y_test)

    print("=" * 60)
    print(f"Données          : {'SYNTHÉTIQUES' if is_synthetic else 'RÉELLES'} "
          f"({len(train_subjects)} sujet(s) train / {len(test_subjects)} test, "
          f"{n_train} / {n_test} époques)")
    print(f"Checkpoint       : {'ALÉATOIRE (pas de poids SleepFM officiels)' if is_random_checkpoint else args.checkpoint}")
    print(f"Accuracy         : {results['accuracy']:.4f}")
    print(f"Cohen's kappa    : {results['cohen_kappa']:.4f}")
    print(f"Macro F1         : {results['macro_f1']:.4f}")
    print("=" * 60)
    if is_synthetic or is_random_checkpoint:
        print("Ces résultats ne sont PAS interprétables médicalement (données et/ou "
              "checkpoint non réels) -- ils valident uniquement que le pipeline tourne.")


if __name__ == "__main__":
    main()
