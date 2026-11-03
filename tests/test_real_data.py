"""Valide le pipeline (embeddings SleepFM -> sonde) sur de VRAIES époques
PhysioNet 2018 extraites par src/data/extract_physionet2018.py, en
remplacement du smoke test synthétique de test_pipeline_smoke.py.

Avec un seul sujet réel, un vrai split train/test n'a pas de sens
statistique (généralisation non mesurable) -- ce test fait donc un
ajustement sur l'ensemble des données comme simple vérification de
plomberie (le code tourne sur du vrai signal PSG), pas une évaluation.
Le backbone reste à poids aléatoires (pas de checkpoint SleepFM officiel
téléchargé), donc les métriques n'ont aucune valeur médicale ici.

Sauté si data/processed/train est absent.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.epoch_dataset import PhysioNet2018EpochDataset
from src.models.effnet import EffNet
from src.models.embeddings import CHANNEL_INDICES, SleepFMEncoders
from src.models.linear_probe import evaluate_probe, train_linear_probe

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "train"

requires_real_data = pytest.mark.skipif(
    not PROCESSED_DIR.exists(),
    reason="data/processed/train absent -- voir scripts/download_physionet2018.py et src/data/extract_physionet2018.py",
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
@requires_real_data
def test_real_physionet_data_end_to_end(tmp_path):
    dataset = PhysioNet2018EpochDataset(PROCESSED_DIR)
    print(f"\n{len(dataset)} époques réelles chargées")
    assert len(dataset) > 0

    from collections import Counter

    label_counts = Counter(label for _, label in (dataset[i] for i in range(len(dataset))))
    print(f"Répartition des stades (0=wake,1=n1,2=n2,3=n3,4=rem) : {dict(label_counts)}")

    ckpt_path = tmp_path / "best.pt"
    _build_fake_checkpoint(ckpt_path)
    encoders = SleepFMEncoders(ckpt_path, embedding_dim=512)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    embeddings, labels = [], []
    with torch.no_grad():
        for epochs, batch_labels in loader:
            embeddings.append(encoders.embed_combined(epochs).numpy())
            labels.append(batch_labels.numpy())

    import numpy as np

    X = np.concatenate(embeddings)
    y = np.concatenate(labels)
    assert X.shape == (len(dataset), 512 * 3)
    print(f"Embeddings générés : {X.shape}")

    # Un seul sujet : pas de split train/test significatif, on ajuste sur
    # tout et on lit le score comme vérification de plomberie uniquement
    # (le modèle peut "voir" ces exact échantillons -- ce n'est PAS une
    # mesure de généralisation).
    probe = train_linear_probe(X, y, max_iter=200)
    results = evaluate_probe(probe, X, y)
    print(f"Sonde (ajustée et évaluée sur les mêmes données -- plomberie uniquement) : {results['accuracy']:.3f}")
    assert 0.0 <= results["accuracy"] <= 1.0

    print("\nOK -- pipeline SleepFM validé sur données PhysioNet 2018 réelles.")
