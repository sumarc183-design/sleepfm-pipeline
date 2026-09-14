"""Prétraitement des PSG brutes vers des époques de 30s alignées sur les labels de stade."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessConfig:
    epoch_length_sec: int = 30
    target_sample_rate_hz: int = 128


def resample_signal(signal: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    from scipy.signal import resample

    n_samples = int(len(signal) * dst_hz / src_hz)
    return resample(signal, n_samples)


def segment_into_epochs(signal: np.ndarray, sample_rate_hz: int, epoch_length_sec: int) -> np.ndarray:
    """Découpe un signal continu en époques non chevauchantes de `epoch_length_sec`."""
    epoch_len = sample_rate_hz * epoch_length_sec
    n_epochs = len(signal) // epoch_len
    trimmed = signal[: n_epochs * epoch_len]
    return trimmed.reshape(n_epochs, epoch_len)


def zscore_normalize(epochs: np.ndarray) -> np.ndarray:
    mean = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True) + 1e-8
    return (epochs - mean) / std
