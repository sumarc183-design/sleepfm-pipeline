"""Dataset PyTorch sur l'arborescence X/Y produite par extract_physionet2018.py :
époques X/<sujet>/<sujet>_<idx>.npy (13, n_échantillons), et Y/<sujet>.pickle
qui associe chaque fichier à son stade ("wake", "nonrem2", ...).

Sert uniquement à faire passer les époques par lots dans SleepFMEncoders (voir
src/models/embeddings.py) pour produire les embeddings -- les sondes
s'entraînent ensuite sur les tableaux numpy obtenus, pas sur ce Dataset.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_NAME_TO_LABEL = {"wake": 0, "nonrem1": 1, "nonrem2": 2, "nonrem3": 3, "rem": 4}


class PhysioNet2018EpochDataset(Dataset):
    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.x_dir = self.processed_dir / "X"
        self.y_dir = self.processed_dir / "Y"

        self.samples: list[tuple[Path, str]] = []
        for subject_dir in sorted(self.x_dir.iterdir()):
            label_path = self.y_dir / f"{subject_dir.name}.pickle"
            if not label_path.exists():
                continue
            with open(label_path, "rb") as f:
                labels = pickle.load(f)
            for epoch_file in sorted(subject_dir.glob("*.npy")):
                label = labels.get(epoch_file.name)
                if isinstance(label, dict):
                    label = next(iter(label.keys()))
                if label in STAGE_NAME_TO_LABEL:
                    self.samples.append((epoch_file, label))

        if not self.samples:
            raise FileNotFoundError(f"Aucune époque valide sous {self.processed_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        epoch_file, label = self.samples[idx]
        epoch = np.load(epoch_file).astype(np.float32)
        return torch.from_numpy(epoch), STAGE_NAME_TO_LABEL[label]
