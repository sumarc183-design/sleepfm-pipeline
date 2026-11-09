"""Contrôle d'époques OSF déjà préparées ; aucun rééchantillonnage ni imputation.

Convention : README officiel OSF, section Input Format. Le format ne prouve
pas les unités, les références des électrodes ou la conformité à SleepBench.
"""

from __future__ import annotations

import numpy as np

OSF_CHANNELS = (
    "ECG",
    "EMG_Chin",
    "EMG_LLeg",
    "EMG_RLeg",
    "ABD",
    "THX",
    "NP",
    "SN",
    "EOG_E1_A2",
    "EOG_E2_A1",
    "EEG_C3_A2",
    "EEG_C4_A1",
)
OSF_SAMPLE_RATE = 64
OSF_EPOCH_SAMPLES = 1920


def resolve_osf_channels(channel_names: list[str], mapping: dict[str, str] | None = None) -> list[int]:
    """Indices dans l'ordre OSF. Mapping explicite : nom OSF -> nom source.

    Les noms canoniques sont reconnus à l'identique ; aucune équivalence de
    montage, inversion de polarité ou duplication de signal n'est déduite.
    """
    if (
        not isinstance(channel_names, list)
        or not channel_names
        or any(not isinstance(name, str) or not name.strip() for name in channel_names)
    ):
        raise ValueError("channels doit être une liste non vide de noms textuels")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("Noms de canaux source en double")
    mapping = {} if mapping is None else mapping
    if not isinstance(mapping, dict) or any(
        target not in OSF_CHANNELS or not isinstance(source, str) or not source.strip()
        for target, source in mapping.items()
    ):
        raise ValueError("Mapping attendu : noms OSF connus -> noms source textuels")
    sources = [mapping.get(target, target) for target in OSF_CHANNELS]
    if len(set(sources)) != len(sources):
        raise ValueError("Un même canal source ne peut pas remplacer plusieurs canaux OSF")
    missing = [
        f"{target} ({source})"
        for target, source in zip(OSF_CHANNELS, sources, strict=True)
        if source not in channel_names
    ]
    if missing:
        raise ValueError(f"Canaux OSF absents : {', '.join(missing)}")
    return [channel_names.index(source) for source in sources]


def prepare_osf_epochs(
    epochs: np.ndarray,
    channel_names: list[str],
    sample_rate: float,
    mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Vérifie et ordonne les époques, puis renvoie une copie float32 contiguë.

    Le signal doit déjà être flottant, à 64 Hz et découpé en fenêtres de 30 s.
    Cette étape ne normalise pas les amplitudes et ne touche pas aux labels.
    """
    indices = resolve_osf_channels(channel_names, mapping)
    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(sample_rate)
        or sample_rate != OSF_SAMPLE_RATE
    ):
        raise ValueError("Fréquence OSF attendue : 64 Hz ; rééchantillonnage à définir en amont")
    if (
        not isinstance(epochs, np.ndarray)
        or epochs.ndim != 3
        or len(epochs) == 0
        or epochs.shape[1:] != (len(channel_names), OSF_EPOCH_SAMPLES)
    ):
        raise ValueError(f"Époques attendues : (n > 0, {len(channel_names)}, {OSF_EPOCH_SAMPLES})")
    if epochs.dtype.kind != "f" or not np.isfinite(epochs).all():
        raise ValueError("Époques flottantes et finies obligatoires")
    with np.errstate(over="ignore"):
        ordered = np.ascontiguousarray(epochs[:, indices, :], dtype=np.float32)
    if not np.isfinite(ordered).all():
        raise ValueError("Valeurs hors de la plage float32")
    metadata = {
        "preparation_version": 1,
        "channels": list(OSF_CHANNELS),
        "source_channels": channel_names.copy(),
        "source_indices": indices,
        "sample_rate": OSF_SAMPLE_RATE,
        "epoch_samples": OSF_EPOCH_SAMPLES,
        "epoch_count": len(ordered),
        "explicit_mapping": dict(mapping or {}),
        "amplitude_normalization_applied": False,
        "physiological_equivalence_verified": False,
    }
    return ordered, metadata
