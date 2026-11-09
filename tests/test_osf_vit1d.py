"""Vérifie l'architecture OSF (ViT-1D) et le round-trip de checkpoint."""

import pytest
import torch

from src.models.osf_vit1d import OSFViT1D, load_osf_backbone, vit_base


def tiny_model(**kwargs):
    return OSFViT1D(width=8, depth=1, heads=2, dim_head=4, mlp_dim=16, **kwargs)


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


def test_checkpoint_uses_upstream_attention_key_names():
    model = tiny_model(num_leads=4, seq_len=8, patch_size=2)
    keys = model.state_dict().keys()
    assert "block0.attn.fn.to_out.0.weight" in keys
    assert "block0.attn.fn.to_out.0.bias" in keys
    assert "block0.attn.fn.to_out.weight" not in keys


def test_grouped_patch_order_and_deprecated_embedding():
    model = tiny_model(num_leads=4, seq_len=8, patch_size=2, lead_wise=1, patch_size_ch=2).eval()
    # Isoler le découpage, avec des sorties de convolution calculables à la main.
    model.block0 = torch.nn.Identity()
    model.norm = torch.nn.Identity()
    with torch.no_grad():
        model.to_patch_embedding.weight.fill_(1)
    signal = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    cls, patches = model.forward_encoding(signal)
    expected = torch.tensor([18, 26, 34, 42, 82, 90, 98, 106], dtype=torch.float32)
    assert patches.shape == (1, 8, 8)
    torch.testing.assert_close(patches[0, :, 0], expected)
    sequence = model.forward_encoding(signal, return_sequence=True)
    torch.testing.assert_close(sequence[:, 0], cls)
    torch.testing.assert_close(sequence[:, 1:], patches)
    torch.testing.assert_close(model(signal), cls)
    with torch.no_grad():
        model.lead_emb.weight.fill_(1_000_000)
    torch.testing.assert_close(model.forward_encoding(signal)[1], patches)


def test_grouped_checkpoint_loader_honours_metadata_and_freezes_weights(tmp_path, monkeypatch):
    from src.models import osf_vit1d

    model = tiny_model(num_leads=4, seq_len=8, patch_size=2, lead_wise=1, patch_size_ch=2).eval()
    path = tmp_path / "grouped.pth"
    metadata = {"num_leads": 4, "seq_len": 8, "patch_size_time": 2, "lead_wise": 1, "patch_size_ch": 2}
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, path)
    monkeypatch.setattr(osf_vit1d, "vit_base", tiny_model)
    loaded = load_osf_backbone(path)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert loaded.lead_wise == 1
    signal = torch.randn(2, 4, 8)
    torch.testing.assert_close(loaded.forward_encoding(signal)[1], model.forward_encoding(signal)[1])
    metadata.pop("patch_size_ch")
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="patch_size_ch obligatoire"):
        load_osf_backbone(path)


@pytest.mark.parametrize("problem", ["channels", "duration", "empty", "nan", "integer"])
def test_bad_osf_inputs_are_refused(problem):
    model = tiny_model(num_leads=4, seq_len=8, patch_size=2)
    signal = torch.ones(1, 4, 8)
    if problem == "channels":
        signal = signal[:, :3]
    elif problem == "duration":
        signal = signal[:, :, :7]
    elif problem == "empty":
        signal = signal[:0]
    elif problem == "nan":
        signal[0, 0, 0] = float("nan")
    else:
        signal = signal.long()
    with pytest.raises(ValueError, match="Entrée OSF"):
        model.forward_encoding(signal)


@pytest.mark.parametrize("changes", [{"lead_wise": 2}, {"patch_size": 3}, {"num_leads": 3, "lead_wise": 1}])
def test_invalid_osf_geometry_is_refused(changes):
    with pytest.raises(ValueError):
        tiny_model(**({"num_leads": 4, "seq_len": 8, "patch_size": 2} | changes))
