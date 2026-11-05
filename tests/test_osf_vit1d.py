"""Vérifie l'architecture OSF (ViT-1D) et le round-trip de checkpoint."""

import torch

from src.models.osf_vit1d import load_osf_backbone, vit_base


def test_osf_vit_base_forward_shapes():
    model = vit_base(num_leads=12, seq_len=1920, patch_size=64)
    cls_emb, patch_emb = model.forward_encoding(torch.randn(2, 12, 1920))
    assert cls_emb.shape == (2, 768)
    assert patch_emb.shape == (2, 1920 // 64, 768)


def test_osf_checkpoint_roundtrip(tmp_path):
    model = vit_base(num_leads=12, seq_len=1920, patch_size=64)
    ckpt_path = tmp_path / "osf_backbone.pth"
    torch.save(
        {
            "metadata": {"num_leads": 12, "seq_len": 1920, "patch_size_time": 64, "lead_wise": 0, "patch_size_ch": 4},
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    loaded = load_osf_backbone(ckpt_path)
    x = torch.randn(1, 12, 1920)
    cls_a, _ = model.forward_encoding(x)
    cls_b, _ = loaded.forward_encoding(x)
    assert torch.allclose(cls_a, cls_b, atol=1e-5)


def test_osf_loading_refuses_missing_or_unexpected_keys(tmp_path):
    import pytest

    model = vit_base(num_leads=12, seq_len=1920, patch_size=64)
    state = model.state_dict()
    state.pop(next(iter(state)))  # une clé manquante
    state["poids_inconnu"] = torch.zeros(1)  # une clé inattendue
    ckpt_path = tmp_path / "osf_backbone.pth"
    meta = {"num_leads": 12, "seq_len": 1920, "patch_size_time": 64, "lead_wise": 0, "patch_size_ch": 4}
    torch.save({"metadata": meta, "state_dict": state}, ckpt_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_osf_backbone(ckpt_path)
    assert load_osf_backbone(ckpt_path, allow_key_mismatch=True) is not None
