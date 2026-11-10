"""Conversion PhysioNet 2018 -> entrée OSF et embeddings sur les mêmes fenêtres que SleepFM."""

import numpy as np
import pytest

from src.data.extract_physionet2018 import ALL_CHANNELS
from src.data.osf_input import OSF_CHANNELS
from src.data.osf_physionet2018 import (
    CLIP,
    MISSING_OSF_CHANNELS,
    PHYSIONET_TO_OSF,
    night_to_osf,
    preparation_contract,
)


def _night(n=3, seed=0):
    rng = np.random.default_rng(seed)
    epochs = rng.normal(size=(n, len(ALL_CHANNELS), 7680))
    for i in range(len(ALL_CHANNELS)):
        epochs[:, i] = 100 * epochs[:, i] + 10 * i  # amplitudes et décalages différents par canal
    return epochs


def test_mapping_order_zero_fill_and_shape():
    out, flat = night_to_osf(_night())
    assert out.shape == (3, 12, 1920) and out.dtype == np.float32 and flat == []
    for name in MISSING_OSF_CHANNELS:
        assert not out[:, OSF_CHANNELS.index(name)].any()
    assert sorted(MISSING_OSF_CHANNELS) == ["EMG_LLeg", "EMG_RLeg", "EOG_E2_A1", "SN"]
    for name in PHYSIONET_TO_OSF:
        channel = out[:, OSF_CHANNELS.index(name)]
        assert abs(channel.mean()) < 0.05 and abs(channel.std() - 1) < 0.05  # z-score sur la nuit


def test_each_osf_channel_comes_from_its_source():
    epochs = np.zeros((2, len(ALL_CHANNELS), 7680))
    epochs[:, ALL_CHANNELS.index("CHEST")] = np.sin(np.linspace(0, 40, 7680))
    out, flat = night_to_osf(epochs)
    assert out[:, OSF_CHANNELS.index("THX")].std() > 0.5
    assert set(flat) == set(PHYSIONET_TO_OSF) - {"THX"}  # canaux constants signalés et laissés à zéro
    assert not out[:, OSF_CHANNELS.index("ABD")].any()


def test_values_are_clipped():
    epochs = _night()
    epochs[0, ALL_CHANNELS.index("ECG"), :10] = 1e6
    out, _ = night_to_osf(epochs)
    assert out.max() <= CLIP and out.min() >= -CLIP


@pytest.mark.parametrize("bad", [np.zeros((0, 13, 7680)), np.zeros((1, 12, 7680)), np.full((1, 13, 7680), np.nan)])
def test_invalid_epochs_are_refused(bad):
    with pytest.raises(ValueError):
        night_to_osf(bad)


def test_contract_records_the_choices():
    contract = preparation_contract()
    assert contract["zero_filled"] == MISSING_OSF_CHANNELS and contract["uncertain"] == ["NP"]
    assert contract["mapping"]["THX"] == "CHEST" and "z-score" in contract["normalization"]


@pytest.mark.slow
def test_osf_embeddings_use_the_same_windows_as_sleepfm(tmp_path):
    from run_baseline import epoch_groups, epoch_onsets, extract
    from scripts.embed_osf_physionet2018 import save_chunk
    from scripts.generate_synthetic_physionet2018 import generate_subject
    from src.data.epoch_dataset import PhysioNet2018EpochDataset
    from src.models.osf_vit1d import OSFViT1D

    raw = tmp_path / "raw"
    rng = np.random.default_rng(1)
    for i in range(2):
        generate_subject(str(raw), f"tr00-000{i}", sample_rate=32, n_epochs=12, epoch_sec=30, lead_in_sec=2, rng=rng)
    processed = tmp_path / "processed"
    extract(raw, processed, num_threads=1)
    model = OSFViT1D(num_leads=12, seq_len=1920, patch_size=64, width=768, depth=1, heads=2, dim_head=8, mlp_dim=16)
    out = tmp_path / "osf"
    out.mkdir()
    save_chunk(processed, ["tr00-0000", "tr00-0001", "tr00-0009"], model.eval(), out, batch_size=4, device="cpu")

    dataset = PhysioNet2018EpochDataset(processed)
    subjects, epoch_idx = epoch_groups(dataset)
    onsets = epoch_onsets(dataset, processed)
    for subject in ("tr00-0000", "tr00-0001"):
        with np.load(out / f"{subject}.npz") as data:
            rows = np.flatnonzero(subjects == subject)
            assert data["X"].shape == (len(rows), 768)
            assert sorted(data["epoch_idx"].tolist()) == sorted(epoch_idx[rows].tolist())
            assert list(data["epoch_idx"]) == sorted(data["epoch_idx"])  # ordre chronologique
            np.testing.assert_allclose(np.sort(data["onset_s"]), np.sort(onsets[rows]))
    assert (out / "tr00-0009.skipped").exists()  # sujet absent du lot : rejet motivé
