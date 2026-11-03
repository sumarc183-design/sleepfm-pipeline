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

    # Mode par lots, pour 100+ sujets : télécharge les 200 premiers sujets
    # PhysioNet 2018 au fil de l'eau, ne garde sur disque que les embeddings
    # (un .npz par sujet) et reprend là où il s'est arrêté si on relance
    python run_baseline.py --physionet-subjects 200 --embeddings-dir embeddings/ \\
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
from scripts.download_physionet2018 import download_subject, list_subjects
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL, PhysioNet2018EpochDataset
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
    return split_subject_ids([p.name for p in raw_dir.iterdir() if p.is_dir()], test_fraction, seed)


def split_subject_ids(subjects: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split au niveau sujet (pas époque) pour éviter toute fuite -- un même
    sujet ne doit jamais apparaître à la fois en train et en test."""
    subjects = sorted(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_fraction)) if len(subjects) > 1 else 0
    return subjects[n_test:], subjects[:n_test]


def collect_embeddings(encoders: SleepFMEncoders, processed_dir: Path, batch_size: int = 32,
                       device: str = "cpu", num_workers: int = 0):
    dataset = PhysioNet2018EpochDataset(processed_dir)
    X, y = embed_dataset(encoders, dataset, batch_size, device, num_workers)
    return X, y, len(dataset)


def embed_dataset(encoders: SleepFMEncoders, dataset: PhysioNet2018EpochDataset, batch_size: int = 32,
                  device: str = "cpu", num_workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings, labels = [], []
    with torch.no_grad():
        for epochs, batch_labels in loader:
            embeddings.append(encoders.embed_combined(epochs.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def embed_subjects_in_chunks(subject_ids: list[str], fetch_subject, encoders: SleepFMEncoders,
                             embeddings_dir: Path, chunk_size: int, num_threads: int,
                             batch_size: int, device: str) -> None:
    """Traite les sujets par lots : brut -> époques -> embeddings, un
    <id>.npz par sujet dans embeddings_dir, puis supprime brut et époques.
    Seuls les embeddings (~5 Mo/sujet, contre ~0,85 Go de brut + époques
    float64) restent sur disque, ce qui permet de passer à 100+ sujets sur
    le disque de Colab. Un sujet déjà présent dans embeddings_dir n'est pas
    recalculé : relancer après une déconnexion reprend où on s'est arrêté.

    `fetch_subject(subject_id, dest_dir)` doit déposer les 3 fichiers bruts
    du sujet dans dest_dir/<subject_id>/ (téléchargement ou lien local)."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in subject_ids
            if not (embeddings_dir / f"{s}.npz").exists() and not (embeddings_dir / f"{s}.skipped").exists()]
    print(f"{len(subject_ids) - len(todo)} sujet(s) déjà en cache, {len(todo)} à traiter")

    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        print(f"Lot {start // chunk_size + 1}/{-(-len(todo) // chunk_size)} : {len(chunk)} sujet(s)")
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, processed_dir = Path(tmp) / "raw", Path(tmp) / "processed"
            raw_dir.mkdir()
            for subject in chunk:
                fetch_subject(subject, raw_dir)
            extract(raw_dir, processed_dir, num_threads)

            try:
                dataset = PhysioNet2018EpochDataset(processed_dir)
            except FileNotFoundError:
                dataset = None  # aucun sujet du lot n'a passé l'extraction
            if dataset is not None:
                X, y = embed_dataset(encoders, dataset, batch_size, device, num_threads)
                sample_subjects = np.array([epoch_file.parent.name for epoch_file, _ in dataset.samples])
            for subject in chunk:
                mask = sample_subjects == subject if dataset is not None else np.zeros(0, dtype=bool)
                if not mask.any():
                    # Sujet rejeté par l'extraction (canaux manquants, pas de
                    # stades...) : marqué pour ne pas le retenter à chaque relance.
                    (embeddings_dir / f"{subject}.skipped").touch()
                    continue
                np.savez(embeddings_dir / f"{subject}.npz", X=X[mask].astype(np.float32), y=y[mask])


def load_cached_embeddings(embeddings_dir: Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for subject in subjects:
        with np.load(embeddings_dir / f"{subject}.npz") as data:
            X.append(data["X"])
            y.append(data["y"])
    return np.concatenate(X), np.concatenate(y)


def print_results(results: dict, data_desc: str, checkpoint_desc: str, not_interpretable: bool) -> None:
    label_to_stage = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}
    print("=" * 60)
    print(f"Données          : {data_desc}")
    print(f"Checkpoint       : {checkpoint_desc}")
    print(f"Accuracy         : {results['accuracy']:.4f}")
    print(f"Cohen's kappa    : {results['cohen_kappa']:.4f}")
    print(f"Macro F1         : {results['macro_f1']:.4f}")
    for label, metrics in results["per_class_report"].items():
        if label.isdigit():
            print(f"  F1 {label_to_stage[int(label)]:>10s} : {metrics['f1-score']:.4f}  (n={int(metrics['support'])})")
    print("=" * 60)
    if not_interpretable:
        print("Ces résultats ne sont PAS interprétables médicalement (données et/ou "
              "checkpoint non réels) -- ils valident uniquement que le pipeline tourne.")


def run_in_chunks(args, checkpoint_path: Path, tmp_path: Path) -> None:
    """Mode --embeddings-dir : voir embed_subjects_in_chunks."""
    if args.physionet_subjects:
        subject_ids = list_subjects("training")[:args.physionet_subjects]
        fetch_subject = download_subject
        data_kind = "RÉELLES"
    else:
        raw_dir = args.raw_data
        if raw_dir is None:
            raw_dir = tmp_path / "raw"
            print(f"[synthétique] Génération de {args.num_synthetic_subjects} sujets factices...")
            generate_synthetic_raw(raw_dir, args.num_synthetic_subjects, args.seed)
        subject_ids = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

        def fetch_subject(subject: str, dest_dir: Path) -> None:
            (dest_dir / subject).symlink_to((raw_dir / subject).resolve(), target_is_directory=True)

        data_kind = "SYNTHÉTIQUES" if args.raw_data is None else "RÉELLES"

    encoders = SleepFMEncoders(checkpoint_path, embedding_dim=512).to(args.device)
    print(f"Embeddings sur : {args.device}")
    embed_subjects_in_chunks(subject_ids, fetch_subject, encoders, args.embeddings_dir, args.chunk_size,
                             args.num_threads, args.batch_size, args.device)

    cached = [s for s in subject_ids if (args.embeddings_dir / f"{s}.npz").exists()]
    n_skipped = len(subject_ids) - len(cached)
    if len(cached) < 2:
        raise SystemExit(f"Seulement {len(cached)} sujet(s) exploitable(s) : impossible de faire un split train/test.")
    train_subjects, test_subjects = split_subject_ids(cached, args.test_fraction, args.seed)
    X_train, y_train = load_cached_embeddings(args.embeddings_dir, train_subjects)
    X_test, y_test = load_cached_embeddings(args.embeddings_dir, test_subjects)

    probe = train_linear_probe(X_train, y_train, max_iter=args.max_iter)
    results = evaluate_probe(probe, X_test, y_test)
    print_results(
        results,
        f"{data_kind} ({len(train_subjects)} sujet(s) train / {len(test_subjects)} test, "
        f"{len(y_train)} / {len(y_test)} époques ; {n_skipped} sujet(s) rejeté(s) à l'extraction)",
        "ALÉATOIRE (pas de poids SleepFM officiels)" if args.checkpoint is None else str(args.checkpoint),
        data_kind == "SYNTHÉTIQUES" or args.checkpoint is None,
    )


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
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="Mode par lots : un .npz d'embeddings par sujet dans ce dossier, brut et époques "
                             "supprimés au fur et à mesure, reprise automatique (voir embed_subjects_in_chunks).")
    parser.add_argument("--physionet-subjects", type=int, default=None,
                        help="Avec --embeddings-dir : télécharge au fil de l'eau les N premiers sujets "
                             "d'entraînement PhysioNet 2018 (au lieu de --raw-data).")
    parser.add_argument("--chunk-size", type=int, default=8, help="Sujets par lot en mode --embeddings-dir")
    parser.add_argument("--max-iter", type=int, default=100, help="Itérations max de la régression logistique")
    args = parser.parse_args()

    if args.physionet_subjects and args.embeddings_dir is None:
        parser.error("--physionet-subjects nécessite --embeddings-dir")
    if args.physionet_subjects and args.raw_data is not None:
        parser.error("--physionet-subjects et --raw-data sont exclusifs")

    is_synthetic = args.raw_data is None
    is_random_checkpoint = args.checkpoint is None

    if args.embeddings_dir is not None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = args.checkpoint
            if is_random_checkpoint:
                checkpoint_path = Path(tmp) / "random_checkpoint.pt"
                build_random_checkpoint(checkpoint_path)
            run_in_chunks(args, checkpoint_path, Path(tmp))
        return

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

        probe = train_linear_probe(X_train, y_train, max_iter=args.max_iter)
        results = evaluate_probe(probe, X_test, y_test)

    print_results(
        results,
        f"{'SYNTHÉTIQUES' if is_synthetic else 'RÉELLES'} "
        f"({len(train_subjects)} sujet(s) train / {len(test_subjects)} test, {n_train} / {n_test} époques)",
        "ALÉATOIRE (pas de poids SleepFM officiels)" if is_random_checkpoint else str(args.checkpoint),
        is_synthetic or is_random_checkpoint,
    )


if __name__ == "__main__":
    main()
