"""Fait tourner tout le pipeline (génération synthétique -> extraction
PhysioNet 2018 -> embeddings SleepFM -> sonde logistique -> évaluation) de
bout en bout, pour valider la plomberie avant l'arrivée des vraies données.
Marqué "slow" : exclure avec `pytest -m "not slow"` pour n'avoir que les
tests unitaires rapides.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_synthetic_physionet2018 import generate_subject
from src.data.epoch_dataset import PhysioNet2018EpochDataset
from src.data.extract_physionet2018 import main as extract_main
from src.models.effnet import EffNet
from src.models.embeddings import CHANNEL_INDICES, SleepFMEncoders
from src.models.linear_probe import evaluate_probe, train_linear_probe


def _generate_synthetic_raw(raw_dir: Path, n_subjects: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_subjects):
        generate_subject(
            str(raw_dir), f"synthetic-{i:04d}", sample_rate=32,
            n_epochs=10, epoch_sec=30, lead_in_sec=2, rng=rng,
        )


def _build_fake_checkpoint(path: Path, embedding_dim: int = 512) -> None:
    def build_encoder(in_ch):
        m = EffNet(in_channel=in_ch, stride=2, dilation=1)
        m.fc = torch.nn.Linear(m.fc.in_features, embedding_dim)
        return m

    torch.save(
        {
            "respiratory_state_dict": build_encoder(len(CHANNEL_INDICES["respiratory"])).state_dict(),
            "sleep_stages_state_dict": build_encoder(len(CHANNEL_INDICES["sleep_stages"])).state_dict(),
            "ekg_state_dict": build_encoder(len(CHANNEL_INDICES["ekg"])).state_dict(),
        },
        path,
    )


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path):
    raw_train, raw_test = tmp_path / "raw_train", tmp_path / "raw_test"
    processed_train, processed_test = tmp_path / "proc_train", tmp_path / "proc_test"
    _generate_synthetic_raw(raw_train, n_subjects=3, seed=1)
    _generate_synthetic_raw(raw_test, n_subjects=2, seed=2)

    sys_argv_backup = None
    import sys as _sys

    for raw_dir, processed_dir in [(raw_train, processed_train), (raw_test, processed_test)]:
        sys_argv_backup, _sys.argv = _sys.argv, [
            "extract", "--data_path", str(raw_dir), "--save_path", str(processed_dir), "--num_threads", "1",
        ]
        try:
            extract_main()
        finally:
            _sys.argv = sys_argv_backup

    assert len(list((processed_train / "X").rglob("*.npy"))) > 0
    assert len(list((processed_test / "X").rglob("*.npy"))) > 0

    ckpt_path = tmp_path / "best.pt"
    _build_fake_checkpoint(ckpt_path)
    encoders = SleepFMEncoders(ckpt_path, embedding_dim=512)

    def collect(processed_dir):
        loader = torch.utils.data.DataLoader(PhysioNet2018EpochDataset(processed_dir), batch_size=16)
        embeddings, labels = [], []
        with torch.no_grad():
            for epochs, batch_labels in loader:
                embeddings.append(encoders.embed_combined(epochs).numpy())
                labels.append(batch_labels.numpy())
        return np.concatenate(embeddings), np.concatenate(labels)

    X_train, y_train = collect(processed_train)
    X_test, y_test = collect(processed_test)

    probe = train_linear_probe(X_train, y_train)
    results = evaluate_probe(probe, X_test, y_test)

    assert 0.0 <= results["accuracy"] <= 1.0
    assert -1.0 <= results["cohen_kappa"] <= 1.0
