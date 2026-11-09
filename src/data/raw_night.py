"""Fenêtres PhysioNet 2018 sans lecture ni création de labels.

Même traitement du signal que l'extraction annotée : valeurs numériques du
.mat, rééchantillonnage MNE à 256 Hz, puis scipy par époque. Seul le choix des
fenêtres change : grille fixe de 30 s depuis zéro, fin incomplète retirée.
Ces fenêtres ne doivent pas être ajoutées aux caches d'entraînement annotés.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.signal
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from src.data.extract_physionet2018 import ALL_CHANNELS, import_signal_names

EPOCH_DURATION_S = 30
SAMPLE_RATE = 256
EPOCH_SAMPLES = EPOCH_DURATION_S * SAMPLE_RATE


class RawNightDataset(Dataset):
    """Une nuit en mémoire ; les tenseurs d'époques sont créés à la demande.

    Accepte les fichiers .hea/.mat du Challenge 2018, pas les EDF ni les
    montages d'autres appareils. Aucun fichier -arousal.mat n'est consulté.
    """

    def __init__(self, subject_dir: Path) -> None:
        self.subject_dir = Path(subject_dir)
        self.subject = self.subject_dir.name
        self.header_path = self.subject_dir / f"{self.subject}.hea"
        self.signal_path = self.subject_dir / f"{self.subject}.mat"
        names, sample_rate, n_samples = import_signal_names(str(self.header_path))
        if sample_rate <= 0 or n_samples <= 0:
            raise ValueError("Fréquence et nombre d'échantillons doivent être positifs")
        if len(names) != len(set(names)):
            raise ValueError("Noms de canaux en double")
        missing = [name for name in ALL_CHANNELS if name not in names]
        if missing:
            raise ValueError(f"Canaux obligatoires absents : {missing}")
        signal = loadmat(self.signal_path, variable_names=["val"])["val"]
        if signal.shape != (len(names), n_samples):
            raise ValueError(f"Signal de forme {signal.shape}, attendu {(len(names), n_samples)} selon l'en-tête")
        if signal.dtype.kind not in "iuf" or not np.isfinite(signal).all():
            raise ValueError("Signal non numérique réel ou contenant des valeurs non finies")
        # Compter avant rééchantillonnage : son arrondi ne doit pas transformer
        # une fin incomplète en fenêtre complète.
        self.n_epochs = n_samples // (sample_rate * EPOCH_DURATION_S)
        if not self.n_epochs:
            raise ValueError("Enregistrement trop court : aucune fenêtre complète de 30 s")
        info = mne.create_info(names, sample_rate, ch_types="eeg")
        raw = mne.io.RawArray(signal, info, verbose=False)
        raw.resample(sfreq=SAMPLE_RATE, verbose=False)
        self.signal = raw.get_data(picks=ALL_CHANNELS)
        self.duration_s = n_samples / sample_rate
        self.source_sample_rate = sample_rate
        self.epoch_idx = np.arange(self.n_epochs, dtype=np.int64)
        self.onset_s = self.epoch_idx.astype(float) * EPOCH_DURATION_S

    def __len__(self) -> int:
        return self.n_epochs

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.n_epochs:
            raise IndexError(index)
        epoch = self.signal[:, index * EPOCH_SAMPLES : (index + 1) * EPOCH_SAMPLES]
        # Conserve aussi la dernière étape de l'extracteur annoté, même à
        # fréquence inchangée, pour garder le même calcul numérique.
        return torch.from_numpy(scipy.signal.resample(epoch, EPOCH_SAMPLES, axis=1).astype(np.float32))

    def metadata(self) -> dict:
        return {
            "subject": self.subject,
            "epoch_selection": "fixed_30s_from_recording_start",
            "epoch_duration_s": EPOCH_DURATION_S,
            "epoch_count": self.n_epochs,
            "recording_duration_s": self.duration_s,
            "discarded_tail_s": self.duration_s - self.n_epochs * EPOCH_DURATION_S,
            "source_sample_rate": self.source_sample_rate,
            "annotations_used": False,
        }
