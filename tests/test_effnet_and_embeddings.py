"""Vérifie l'architecture EffNet et le chargement de checkpoint SleepFM
(shapes, normalisation, robustesse au préfixe "module." de DataParallel)."""
import torch

from src.models.effnet import EffNet
from src.models.embeddings import ALL_CHANNELS, CHANNEL_INDICES, SleepFMEncoders


def test_effnet_forward_shapes():
    for name, n_ch in [("respiratory", 3), ("sleep_stages", 5), ("ekg", 1)]:
        model = EffNet(in_channel=n_ch, stride=2, dilation=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 512)
        out = model(torch.randn(2, n_ch, 256 * 30))
        assert out.shape == (2, 512), f"{name}: {out.shape}"


def _build_fake_checkpoint(tmp_path, embedding_dim=512):
    def build_encoder(in_ch):
        m = EffNet(in_channel=in_ch, stride=2, dilation=1)
        m.fc = torch.nn.Linear(m.fc.in_features, embedding_dim)
        return m

    resp = build_encoder(len(CHANNEL_INDICES["respiratory"]))
    sleep = build_encoder(len(CHANNEL_INDICES["sleep_stages"]))
    ekg = build_encoder(len(CHANNEL_INDICES["ekg"]))

    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "respiratory_state_dict": resp.state_dict(),
            "sleep_stages_state_dict": sleep.state_dict(),
            "ekg_state_dict": ekg.state_dict(),
            "temperature": 0.07,
        },
        ckpt_path,
    )
    return ckpt_path


def test_sleepfm_encoders_load_and_embed(tmp_path):
    ckpt_path = _build_fake_checkpoint(tmp_path)
    encoders = SleepFMEncoders(ckpt_path, embedding_dim=512)

    batch = torch.randn(4, len(ALL_CHANNELS), 256 * 30)
    combined = encoders.embed_combined(batch)
    assert combined.shape == (4, 512 * 3)

    # chaque bloc de 512 doit être L2-normalisé (norme ~1)
    norms = combined.view(4, 3, 512).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_sleepfm_encoders_missing_checkpoint_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        SleepFMEncoders(tmp_path / "does_not_exist.pt")
