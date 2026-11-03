"""Charge le checkpoint tri-modal officiel de SleepFM (best.pt) et produit des
embeddings figés, normalisés L2, par modalité (respiratoire / stades / ECG),
exactement comme 3_generate_embed_pretraining.py du dépôt officiel.

Le checkpoint contient, pour chaque modalité, l'une de deux conventions de
nommage des clés (les deux existent dans le dépôt officiel
rthapa84/sleepfm-codebase) :
  - "respiratory_state_dict", "sleep_stages_state_dict", "ekg_state_dict"
    (utilisées par 2_pretrain.py / 3_generate_embed_pretraining.py, c'est-à-dire
    ce qu'on obtient en ré-entraînant soi-même) ;
  - "resp_state_dict", "sleep_state_dict", "ekg_state_dict"
    (les clés réellement présentes dans le best.pt publié -- vérifié en le
    chargeant : incohérence du dépôt amont entre le checkpoint livré et son
    propre script de chargement, pas une erreur introduite ici).
Les deux formes sont acceptées pour que le best.pt publié se charge tel quel
(un préfixe "module." issu d'un entraînement DataParallel est retiré).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .effnet import EffNet

CHANNEL_GROUPS = {
    "respiratory": ["CHEST", "SaO2", "ABD"],
    "sleep_stages": ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"],
    "ekg": ["ECG"],
}
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
CHANNEL_INDICES = {group: [ALL_CHANNELS.index(ch) for ch in channels] for group, channels in CHANNEL_GROUPS.items()}


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _get_state_dict(checkpoint: dict, *possible_keys: str) -> dict:
    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(
        f"Aucune des clés {possible_keys} n'est présente dans le checkpoint "
        f"(clés disponibles : {list(checkpoint.keys())})."
    )


def build_encoder(in_channels: int, embedding_dim: int) -> EffNet:
    encoder = EffNet(in_channel=in_channels, stride=2, dilation=1)
    encoder.fc = torch.nn.Linear(encoder.fc.in_features, embedding_dim)
    return encoder


class SleepFMEncoders(torch.nn.Module):
    """Trois encodeurs EffNet figés, un par groupe de modalités."""

    def __init__(self, checkpoint_path: str | Path, embedding_dim: int = 512) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}. "
                "Lancer bash scripts/setup_sleepfm.sh : best.pt est versionné dans le dépôt officiel "
                "(third_party/sleepfm/sleepfm/checkpoint/best.pt)."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.respiratory = build_encoder(len(CHANNEL_INDICES["respiratory"]), embedding_dim)
        self.sleep_stages = build_encoder(len(CHANNEL_INDICES["sleep_stages"]), embedding_dim)
        self.ekg = build_encoder(len(CHANNEL_INDICES["ekg"]), embedding_dim)

        self.respiratory.load_state_dict(
            _strip_module_prefix(_get_state_dict(checkpoint, "respiratory_state_dict", "resp_state_dict"))
        )
        self.sleep_stages.load_state_dict(
            _strip_module_prefix(_get_state_dict(checkpoint, "sleep_stages_state_dict", "sleep_state_dict"))
        )
        self.ekg.load_state_dict(_strip_module_prefix(_get_state_dict(checkpoint, "ekg_state_dict")))

        for module in (self.respiratory, self.sleep_stages, self.ekg):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, epoch: torch.Tensor) -> dict[str, torch.Tensor]:
        """`epoch`: (batch, len(ALL_CHANNELS), n_samples) float tensor."""
        return {
            "respiratory": F.normalize(self.respiratory(epoch[:, CHANNEL_INDICES["respiratory"], :])),
            "sleep_stages": F.normalize(self.sleep_stages(epoch[:, CHANNEL_INDICES["sleep_stages"], :])),
            "ekg": F.normalize(self.ekg(epoch[:, CHANNEL_INDICES["ekg"], :])),
        }

    @torch.no_grad()
    def embed_combined(self, epoch: torch.Tensor) -> torch.Tensor:
        embeddings = self.forward(epoch)
        return torch.cat([embeddings["respiratory"], embeddings["sleep_stages"], embeddings["ekg"]], dim=1)
