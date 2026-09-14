"""Dataset PyTorch pour la classification des stades du sommeil à partir d'époques prétraitées."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_TO_LABEL = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


class SleepStageDataset(Dataset):
    """Attend un répertoire de fichiers .npz avec les clés `signal` (n_epochs, n_samples)
    et `stage` (n_epochs,) contenant des codes de stade parmi STAGE_TO_LABEL.
    """

    def __init__(self, processed_dir: str | Path):
        self.processed_dir = Path(processed_dir)
        self.files = sorted(self.processed_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"Aucun fichier .npz dans {self.processed_dir}")

        self._index: list[tuple[int, int]] = []
        self._cache: dict[int, np.lib.npyio.NpzFile] = {}
        for file_idx, file in enumerate(self.files):
            n_epochs = len(np.load(file)["stage"])
            self._index.extend((file_idx, epoch_idx) for epoch_idx in range(n_epochs))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, epoch_idx = self._index[idx]
        if file_idx not in self._cache:
            self._cache[file_idx] = np.load(self.files[file_idx])
        data = self._cache[file_idx]
        signal = torch.from_numpy(data["signal"][epoch_idx]).float()
        label = int(data["stage"][epoch_idx])
        return signal, torch.tensor(label, dtype=torch.long)
