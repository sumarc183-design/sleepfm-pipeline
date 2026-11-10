"""Époques PhysioNet 2018 converties en entrée OSF (12 canaux, 64 Hz, 30 s).

Part des époques annotées déjà extraites pour SleepFM (13 canaux, 256 Hz,
`ALL_CHANNELS`) : les deux modèles voient donc exactement les mêmes fenêtres.

Choix, à garder en tête pour toute comparaison (option A, exploratoire) :
- 8 canaux sur 12 seulement. Les 4 absents (EMG des deux jambes, ronflement,
  EOG E2) sont remplis de zéros, comme le fait le code officiel d'OSF pour un
  canal manquant. NP (pression nasale) reçoit AIRFLOW : capteur peut-être différent.
- Rééchantillonnage 256 -> 64 Hz par filtre polyphase (le code officiel interpole
  linéairement depuis ses fichiers prétraités, non publiés).
- Normalisation : z-score par canal sur toute la nuit, puis bornage à ±6. Le code
  officiel borne à ±6 sans normaliser les signaux PSG au chargement, ce qui suppose
  des fichiers déjà standardisés ; le prétraitement exact de SleepBench n'est pas
  publié. C'est une hypothèse, pas une reproduction.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

from src.data.extract_physionet2018 import ALL_CHANNELS
from src.data.osf_input import OSF_CHANNELS, OSF_EPOCH_SAMPLES, OSF_SAMPLE_RATE

PREPARATION_VERSION = 1
SOURCE_SAMPLE_RATE = 256
SOURCE_EPOCH_SAMPLES = 30 * SOURCE_SAMPLE_RATE
CLIP = 6.0
# Canal OSF -> canal PhysioNet 2018. Les canaux OSF absents de ce dictionnaire sont mis à zéro.
PHYSIONET_TO_OSF = {
    "ECG": "ECG",
    "EMG_Chin": "Chin1-Chin2",
    "ABD": "ABD",
    "THX": "CHEST",
    "NP": "AIRFLOW",
    "EOG_E1_A2": "E1-M2",
    "EEG_C3_A2": "C3-M2",
    "EEG_C4_A1": "C4-M1",
}
MISSING_OSF_CHANNELS = [name for name in OSF_CHANNELS if name not in PHYSIONET_TO_OSF]
UNCERTAIN_OSF_CHANNELS = ["NP"]


def preparation_contract() -> dict:
    """Ce qui détermine l'entrée OSF : à enregistrer avec les embeddings."""
    return {
        "preparation_version": PREPARATION_VERSION,
        "osf_channels": list(OSF_CHANNELS),
        "mapping": dict(PHYSIONET_TO_OSF),
        "zero_filled": list(MISSING_OSF_CHANNELS),
        "uncertain": list(UNCERTAIN_OSF_CHANNELS),
        "sample_rate": OSF_SAMPLE_RATE,
        "resampling": "scipy.signal.resample_poly 1/4 par époque",
        "normalization": f"z-score par canal sur la nuit, bornage à ±{CLIP:g}",
    }


def night_to_osf(epochs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(époques OSF float32 (n, 12, 1920), canaux plats mis à zéro).

    `epochs` : toutes les époques d'une nuit, (n, 13, 7680) dans l'ordre `ALL_CHANNELS`.
    La moyenne et l'écart-type viennent de toute la nuit, pas de chaque époque."""
    if epochs.ndim != 3 or epochs.shape[1:] != (len(ALL_CHANNELS), SOURCE_EPOCH_SAMPLES) or not len(epochs):
        raise ValueError(f"Époques attendues : (n > 0, {len(ALL_CHANNELS)}, {SOURCE_EPOCH_SAMPLES})")
    if epochs.dtype.kind != "f" or not np.isfinite(epochs).all():
        raise ValueError("Époques flottantes et finies obligatoires")
    down = scipy.signal.resample_poly(epochs, OSF_SAMPLE_RATE, SOURCE_SAMPLE_RATE, axis=2)
    out = np.zeros((len(epochs), len(OSF_CHANNELS), OSF_EPOCH_SAMPLES), dtype=np.float32)
    flat = []
    for target, source in PHYSIONET_TO_OSF.items():
        signal = down[:, ALL_CHANNELS.index(source), :].astype(np.float64)
        std = signal.std()
        if not std > 0:
            flat.append(target)  # canal constant sur toute la nuit : laissé à zéro, signalé
            continue
        out[:, OSF_CHANNELS.index(target), :] = np.clip((signal - signal.mean()) / std, -CLIP, CLIP)
    return out, flat
