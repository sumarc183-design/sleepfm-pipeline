"""Génère un jeu de données SYNTHÉTIQUE au format exact de PhysioNet
Challenge 2018 (.hea + .mat signal + -arousal.mat labels), pour pouvoir
faire tourner tout le pipeline (extraction -> embeddings -> sonde ->
évaluation) sans attendre le vrai téléchargement.

A REMPLACER par de vraies données PhysioNet 2018 dès que le téléchargement
est possible (scripts/download_physionet2018.py) -- ce générateur sert
uniquement à valider la plomberie du pipeline, pas à produire des résultats
médicalement significatifs. Les signaux sont du bruit aléatoire, pas de la
physiologie réelle.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from scipy.io import savemat

ALL_CHANNELS = [
    "F3-M2",
    "F4-M1",
    "C3-M2",
    "C4-M1",
    "O1-M2",
    "O2-M1",
    "E1-M2",
    "Chin1-Chin2",
    "ABD",
    "CHEST",
    "AIRFLOW",
    "SaO2",
    "ECG",
]
STAGE_CYCLE = ["wake", "nonrem1", "nonrem2", "nonrem3", "rem"]


def write_hea(path: str, subject_id: str, sample_rate: int, n_samples: int) -> None:
    lines = [f"{subject_id} {len(ALL_CHANNELS)} {sample_rate} {n_samples}"]
    for ch in ALL_CHANNELS:
        # Format PhysioNet réel : nom_fichier gain unite adc_res zero_val ... nom_canal
        lines.append(f"{subject_id}.mat 16 100/mV 16 0 0 0 0 {ch}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_signal(
    path: str, sample_rate: int, lead_in_sec: int, n_epochs: int, epoch_sec: int, rng: np.random.Generator
) -> int:
    n_samples = (lead_in_sec + n_epochs * epoch_sec) * sample_rate
    signal = (rng.standard_normal((len(ALL_CHANNELS), n_samples)) * 100).astype(np.int16)
    savemat(path, {"val": signal})
    return n_samples


def write_arousal(path: str, sample_rate: int, lead_in_sec: int, n_epochs: int, epoch_sec: int) -> None:
    """Labels one-hot par échantillon. Une colonne "undefined" marque le
    lead-in (comme le fait le pipeline officiel pour couper le début non
    exploitable de l'enregistrement), suivie d'un cycle des 5 stades."""
    lead_in_samples = lead_in_sec * sample_rate
    epoch_samples = epoch_sec * sample_rate
    n_samples = lead_in_samples + n_epochs * epoch_samples

    stage_names = STAGE_CYCLE + ["undefined"]
    labels = np.zeros((n_samples, len(stage_names)), dtype=np.int16)
    labels[:lead_in_samples, stage_names.index("undefined")] = 1

    for epoch_idx in range(n_epochs):
        stage = STAGE_CYCLE[epoch_idx % len(STAGE_CYCLE)]
        start = lead_in_samples + epoch_idx * epoch_samples
        labels[start : start + epoch_samples, stage_names.index(stage)] = 1

    with h5py.File(path, "w") as f:
        grp = f.create_group("data").create_group("sleep_stages")
        for i, name in enumerate(stage_names):
            grp.create_dataset(name, data=labels[:, i : i + 1].T)  # shape (1, n_samples), comme les fichiers réels


def generate_subject(
    out_dir: str,
    subject_id: str,
    sample_rate: int,
    n_epochs: int,
    epoch_sec: int,
    lead_in_sec: int,
    rng: np.random.Generator,
) -> None:
    subject_dir = os.path.join(out_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    n_samples = write_signal(
        os.path.join(subject_dir, f"{subject_id}.mat"), sample_rate, lead_in_sec, n_epochs, epoch_sec, rng
    )
    write_hea(os.path.join(subject_dir, f"{subject_id}.hea"), subject_id, sample_rate, n_samples)
    write_arousal(os.path.join(subject_dir, f"{subject_id}-arousal.mat"), sample_rate, lead_in_sec, n_epochs, epoch_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-subjects", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20, help="époques de 30s par sujet")
    parser.add_argument(
        "--sample-rate", type=int, default=32, help="Hz -- bas pour un test rapide, réhaussé à 256Hz par le pipeline"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.num_subjects):
        subject_id = f"synthetic-{i:04d}"
        generate_subject(args.out, subject_id, args.sample_rate, args.num_epochs, epoch_sec=30, lead_in_sec=2, rng=rng)
        print(f"Généré {subject_id}")
    print(f"{args.num_subjects} sujet(s) synthétique(s) écrits dans {args.out}")


if __name__ == "__main__":
    main()
