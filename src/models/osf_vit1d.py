"""Backbone Vision Transformer 1D d'OSF, repris de yang-ai-lab/OSF-Open-Sleep-FM
(licence MIT) pour produire des embeddings OSF en vue de la comparaison
SleepFM / OSF, sans cloner de dépôt dans third_party/.

IMPORTANT -- OSF n'est PAS directement comparable à SleepFM :
- Convention d'entrée différente : 12 canaux (ECG, EMG_Chin, EMG_LLeg,
  EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1)
  à 64 Hz par époques de 30 s (1920 échantillons), contre le montage 13 canaux
  à 256 Hz de SleepFM. Les canaux ne se correspondent pas 1 pour 1 :
  reconstruire les 12 canaux d'OSF depuis PhysioNet 2018 / MESA demande une
  étape de correspondance dédiée (pas encore écrite).
- Schéma de labels différent pour la stadification : OSF évalue 4 classes
  (Éveil / Sommeil léger / Sommeil profond / REM), pas les 5 de SleepFM
  (Wake / N1 / N2 / N3 / REM). Une comparaison équitable demande de fusionner
  N1+N2 en "Sommeil léger" côté SleepFM (ou d'évaluer les deux sur un schéma
  commun) -- à faire explicitement avant de comparer des kappas.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div_(keep_prob)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 12, dim_head: int = 64, qkv_bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, mlp_dim: int, heads: int, dim_head: int, qkv_bias: bool, drop_path_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, qkv_bias))
        self.droppath1 = DropPath(drop_path_rate)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim))
        self.droppath2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.droppath1(self.attn(x)) + x
        return self.droppath2(self.ff(x)) + x


class OSFViT1D(nn.Module):
    """Réimplémentation minimale fidèle à osf/backbone/vit1d_cls.py::ViT, assez
    proche pour que le state_dict publié par OSF se charge sans renommage."""

    def __init__(
        self,
        num_leads: int,
        seq_len: int,
        patch_size: int,
        width: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert seq_len % patch_size == 0
        num_patches = seq_len // patch_size
        self.width = width
        self.depth = depth

        self.to_patch_embedding = nn.Conv1d(num_leads, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, width))

        for i in range(depth):
            self.add_module(f"block{i}", TransformerBlock(width, mlp_dim, heads, dim_head, qkv_bias))
        self.norm = nn.LayerNorm(width)

    def forward_encoding(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """series: [B, num_leads, seq_len] -> (cls_embedding [B, width], patch_embeddings [B, N, width])."""
        x = self.to_patch_embedding(series)
        x = rearrange(x, "b c n -> b n c")
        cls_tok = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tok, x], dim=1) + self.pos_embedding[:, : x.size(1) + 1, :]
        for i in range(self.depth):
            x = getattr(self, f"block{i}")(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


def vit_base(num_leads: int, seq_len: int = 1920, patch_size: int = 64, **kwargs) -> OSFViT1D:
    return OSFViT1D(
        num_leads=num_leads,
        seq_len=seq_len,
        patch_size=patch_size,
        width=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )


def load_osf_backbone(checkpoint_path: str | Path) -> OSFViT1D:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint OSF introuvable : {checkpoint_path}. "
            "Télécharger osf_backbone.pth depuis https://huggingface.co/yang-ai-lab/OSF-Base"
        )
    payload = torch.load(checkpoint_path, map_location="cpu")
    meta = payload["metadata"]
    if meta.get("lead_wise", 0) != 0:
        raise NotImplementedError("lead_wise=1 (2D patchify) checkpoints not supported by this minimal loader.")

    model = vit_base(num_leads=meta["num_leads"], seq_len=meta["seq_len"], patch_size=meta["patch_size_time"])
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    print(f"OSF backbone loaded -- missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
