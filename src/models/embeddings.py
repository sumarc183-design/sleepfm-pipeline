"""Wrapper pour charger l'encodeur préentraîné SleepFM (ou OSF) et générer des embeddings.

Ne réentraîne rien : sert uniquement à figer le backbone et produire des
représentations utilisées ensuite par la sonde linéaire.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PretrainedEncoder(nn.Module):
    def __init__(self, checkpoint_path: str | Path, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = self._load_backbone(checkpoint_path)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _load_backbone(self, checkpoint_path: str | Path) -> nn.Module:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Poids introuvables : {checkpoint_path}. "
                "Suivre les instructions de third_party/sleepfm/README.md pour les récupérer."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # NOTE: la structure exacte du checkpoint dépend de la version du dépôt
        # SleepFM clonée dans third_party/. Adapter l'extraction du state_dict
        # et l'architecture du backbone en conséquence avant le premier run.
        raise NotImplementedError(
            "Brancher ici le chargement du backbone officiel SleepFM/OSF "
            "une fois third_party/sleepfm cloné."
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
