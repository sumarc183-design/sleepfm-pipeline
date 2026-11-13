"""Nuit NSRR (EDF) -> époques au format PhysioNet 2018 vu par SleepFM (ébauche du 13 novembre).

Sortie : (n, 13, 7680) float32, ordre `ALL_CHANNELS`, 256 Hz, échelle des valeurs brutes PhysioNet
(`to_physionet_scale`). Elle sert telle quelle à SleepFM et passe par `night_to_osf` pour OSF
(qui normalise chaque canal sur la nuit). Lecteur EDF minimal pour garder la main sur les
unités et la fréquence propre à chaque canal (MNE suréchantillonne tout à la lecture).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np

from src.data.nsrr_channels import SLEEPFM_CHANNELS, channel_plan, to_physionet_scale

SAMPLE_RATE = 256
EPOCH_S = 30
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_S
RESPIRATORY_INPUTS = ("ABD", "CHEST", "AIRFLOW")
RespiratoryScale = Callable[[str, np.ndarray, str], np.ndarray]


def _field(raw: bytes, start: int, width: int, count: int) -> list[str]:
    return [raw[start + i * width : start + (i + 1) * width].decode("ascii").strip() for i in range(count)]


def read_edf(path: str | Path, wanted: list[int] | None = None) -> dict:
    """En-tête EDF et signaux physiques (float64) des canaux demandés, à leur fréquence propre."""
    data = Path(path).read_bytes()
    if len(data) < 256:
        raise ValueError("EDF tronqué")
    header_bytes, n_records = int(data[184:192]), int(data[236:244])
    record_s, n_signals = float(data[244:252]), int(data[252:256])
    if data[192:197] == b"EDF+D":
        raise ValueError("EDF+ discontinu non pris en charge")
    if n_records <= 0 or record_s <= 0:
        raise ValueError("Nombre ou durée des enregistrements invalide")
    h = data[256:header_bytes]
    offsets = np.cumsum([0, 16, 80, 8, 8, 8, 8, 8, 80, 8]) * n_signals
    labels = _field(h, offsets[0], 16, n_signals)
    units = _field(h, offsets[2], 8, n_signals)
    pmin, pmax, dmin, dmax = (np.array(_field(h, offsets[k], 8, n_signals), float) for k in (3, 4, 5, 6))
    per_record = np.array(_field(h, offsets[8], 8, n_signals), int)
    record_len = int(per_record.sum())
    raw = np.frombuffer(data, dtype="<i2", offset=header_bytes)
    if raw.size < n_records * record_len:
        raise ValueError("Données EDF incomplètes")
    records = raw[: n_records * record_len].reshape(n_records, record_len)
    starts = np.concatenate([[0], np.cumsum(per_record)])
    signals = {}
    for i in range(n_signals) if wanted is None else wanted:
        if dmax[i] == dmin[i]:
            raise ValueError(f"{labels[i]} : plage numérique nulle")
        digital = records[:, starts[i] : starts[i + 1]].reshape(-1).astype(np.float64)
        signals[i] = (digital - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i]
    return {
        "labels": labels,
        "units": units,
        "sample_rates": per_record / record_s,
        "duration_s": n_records * record_s,
        "signals": signals,
    }


def to_256(signal: np.ndarray, rate: float) -> np.ndarray:
    """Rééchantillonnage de la nuit entière vers 256 Hz (même méthode que MNE pour PhysioNet)."""
    if rate == SAMPLE_RATE:
        return signal
    if rate <= 0 or rate != round(rate):
        raise ValueError(f"Fréquence non entière : {rate}")
    return mne.filter.resample(signal, up=SAMPLE_RATE, down=int(rate), npad="auto", verbose=False)


def nsrr_night(path: str | Path, cohort: str, respiratory_scale: RespiratoryScale | None) -> tuple[np.ndarray, dict]:
    """Époques de 30 s depuis le début de l'enregistrement (fin incomplète retirée).

    `respiratory_scale(cible, signal_physique, unité)` met les sangles et le débit à l'échelle
    PhysioNet : règle à décider, donc obligatoire dès qu'un de ces canaux est présent. Pour OSF
    seul, une fonction identité suffit (z-score par nuit) ; la sortie ne sert alors pas à SleepFM.
    """
    header = read_edf(path, wanted=[])
    plan = channel_plan(header["labels"], cohort, "sleepfm")
    present = {target: index for target, index in plan["indices"].items() if index is not None}
    if respiratory_scale is None and any(t in present for t in RESPIRATORY_INPUTS):
        raise ValueError("Règle d'échelle des sangles et du débit à fournir")
    edf = read_edf(path, wanted=sorted(set(present.values())))
    n_epochs = int(edf["duration_s"] // EPOCH_S)
    if n_epochs == 0:
        raise ValueError("Enregistrement de moins de 30 s")
    night = np.zeros((len(SLEEPFM_CHANNELS), n_epochs * EPOCH_SAMPLES), dtype=np.float32)
    for target, index in present.items():
        signal, unit = edf["signals"][index], edf["units"][index]
        scaled = (
            respiratory_scale(target, signal, unit)
            if target in RESPIRATORY_INPUTS
            else to_physionet_scale(signal, unit, target)
        )
        resampled = to_256(np.asarray(scaled, dtype=np.float64), float(edf["sample_rates"][index]))
        if resampled.size < night.shape[1]:
            raise ValueError(f"{target} : signal plus court que l'enregistrement")
        night[SLEEPFM_CHANNELS.index(target)] = resampled[: night.shape[1]]
    if not np.isfinite(night).all():
        raise ValueError("Valeurs non finies après préparation")
    epochs = night.reshape(len(SLEEPFM_CHANNELS), n_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)
    info = {
        "cohort": cohort,
        "n_epochs": n_epochs,
        "source_labels": {t: edf["labels"][i] for t, i in present.items()},
        "source_rates": {t: float(edf["sample_rates"][i]) for t, i in present.items()},
        "zero_filled": plan["zero_filled"],
        "uncertain": plan["uncertain"],
    }
    return np.ascontiguousarray(epochs), info
