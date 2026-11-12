"""Correspondance des canaux MESA / SHHS-1 vers les entrées SleepFM et OSF (ébauche du 13 novembre).

Préparé avant l'accès aux données. Les libellés EDF viennent de la documentation publique NSRR
et restent **à vérifier sur les premiers fichiers** : un libellé attendu introuvable, ou trouvé
deux fois, arrête la préparation au lieu d'être deviné.

Règle (à valider) : la règle IAH est figée, donc l'entrée doit ressembler le plus possible à
celle vue sur PhysioNet 2018. Un canal mis à zéro sur PhysioNet reste à zéro sur NSRR même
s'il existe (jambes, ronflement, EOG droit pour OSF). Un canal présent sur PhysioNet mais absent
sur NSRR est mis à zéro et signalé (dégradation attendue, mesurée par la confirmation).

Échelle : SleepFM reçoit les valeurs brutes du .mat PhysioNet. D'après les en-têtes (.hea),
elles valent 1 unité par µV pour l'EEG, l'EOG, l'EMG, l'ECG, les sangles et le débit, et
655,35 × % − 32768 pour la SaO2. Les sangles et le débit n'ont pas d'unité physique comparable
d'un appareil à l'autre : leur mise à l'échelle reste une décision ouverte.
"""

from __future__ import annotations

import numpy as np

from src.data.osf_input import OSF_CHANNELS
from src.data.osf_physionet2018 import PHYSIONET_TO_OSF

SLEEPFM_CHANNELS = (
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2",
    "Chin1-Chin2", "ABD", "CHEST", "AIRFLOW", "SaO2", "ECG",
)  # fmt: skip

# Libellés EDF possibles par signal (documentation NSRR, à vérifier).
MESA_LABELS = {
    "EEG_C4_M1": ("EEG3",),
    "EOG_L": ("EOG-L",),
    "EMG_CHIN": ("EMG",),
    "ECG": ("EKG",),
    "THORAX": ("Thor",),
    "ABDOMEN": ("Abdo",),
    "THERMISTOR": ("Flow",),
    "NASAL_PRESSURE": ("Pres",),
    "SPO2": ("SpO2",),
}
SHHS1_LABELS = {
    "EEG_C3_M2": ("EEG(sec)", "EEG2", "EEG 2", "EEG sec"),
    "EEG_C4_M1": ("EEG",),
    "EOG_L": ("EOG(L)",),
    "EMG_CHIN": ("EMG",),
    "ECG": ("ECG",),
    "THORAX": ("THOR RES",),
    "ABDOMEN": ("ABDO RES",),
    "THERMISTOR": ("AIRFLOW", "NEW AIR", "new A/F"),
    "SPO2": ("SaO2",),
}
COHORT_LABELS = {"mesa": MESA_LABELS, "shhs1": SHHS1_LABELS}

# Entrée SleepFM -> signal NSRR. Absent = zéros. `AIRFLOW` : capteur PhysioNet non documenté.
SLEEPFM_SOURCES = {
    "mesa": {
        "C4-M1": "EEG_C4_M1", "E1-M2": "EOG_L", "Chin1-Chin2": "EMG_CHIN", "ABD": "ABDOMEN",
        "CHEST": "THORAX", "AIRFLOW": "THERMISTOR", "SaO2": "SPO2", "ECG": "ECG",
    },
    "shhs1": {
        "C3-M2": "EEG_C3_M2", "C4-M1": "EEG_C4_M1", "E1-M2": "EOG_L", "Chin1-Chin2": "EMG_CHIN",
        "ABD": "ABDOMEN", "CHEST": "THORAX", "AIRFLOW": "THERMISTOR", "SaO2": "SPO2", "ECG": "ECG",
    },
}  # fmt: skip
# Entrée OSF -> signal NSRR, limitée aux canaux non nuls sur PhysioNet (PHYSIONET_TO_OSF).
# `NP` : PhysioNet a fourni AIRFLOW ; on garde le même type de capteur (thermistance).
OSF_SOURCES = {
    "mesa": {
        "ECG": "ECG", "EMG_Chin": "EMG_CHIN", "ABD": "ABDOMEN", "THX": "THORAX", "NP": "THERMISTOR",
        "EOG_E1_A2": "EOG_L", "EEG_C4_A1": "EEG_C4_M1",
    },
    "shhs1": {
        "ECG": "ECG", "EMG_Chin": "EMG_CHIN", "ABD": "ABDOMEN", "THX": "THORAX", "NP": "THERMISTOR",
        "EOG_E1_A2": "EOG_L", "EEG_C3_A2": "EEG_C3_M2", "EEG_C4_A1": "EEG_C4_M1",
    },
}  # fmt: skip
UNCERTAIN = {"sleepfm": ("AIRFLOW",), "osf": ("NP",)}
MODEL_CHANNELS = {"sleepfm": SLEEPFM_CHANNELS, "osf": OSF_CHANNELS}
MODEL_SOURCES = {"sleepfm": SLEEPFM_SOURCES, "osf": OSF_SOURCES}

VOLTAGE_TO_UV = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6}
SAO2_GAIN, SAO2_BASELINE = 655.35, -32768.0  # en-tête PhysioNet 2018 : 655.35(-32768)/%


def _norm(label: str) -> str:
    return " ".join(label.strip().lower().split())


def find_signal(edf_labels: list[str], cohort: str, signal: str) -> int:
    """Indice unique du signal dans l'EDF ; erreur s'il manque ou s'il est ambigu."""
    wanted = {_norm(label) for label in COHORT_LABELS[cohort][signal]}
    hits = [i for i, label in enumerate(edf_labels) if _norm(label) in wanted]
    if len(hits) != 1:
        raise ValueError(f"{cohort} : signal {signal} trouvé {len(hits)} fois ({sorted(wanted)})")
    return hits[0]


def channel_plan(edf_labels: list[str], cohort: str, model: str) -> dict:
    """Pour chaque entrée du modèle : indice EDF ou None (zéros), avec le détail à publier."""
    if cohort not in COHORT_LABELS or model not in MODEL_CHANNELS:
        raise ValueError(f"Cohorte ou modèle inconnu : {cohort}, {model}")
    sources = MODEL_SOURCES[model][cohort]
    indices = {
        target: find_signal(edf_labels, cohort, sources[target]) if target in sources else None
        for target in MODEL_CHANNELS[model]
    }
    physionet_zero = set(OSF_CHANNELS) - set(PHYSIONET_TO_OSF) if model == "osf" else set()
    return {
        "indices": indices,
        "zero_filled": [t for t, i in indices.items() if i is None],
        "lost_vs_physionet": [t for t, i in indices.items() if i is None and t not in physionet_zero],
        "uncertain": [t for t in UNCERTAIN[model] if indices.get(t) is not None],
    }


def to_physionet_scale(signal: np.ndarray, unit: str, target: str) -> np.ndarray:
    """Remet un signal NSRR à l'échelle des valeurs brutes PhysioNet vues par SleepFM."""
    unit_key = _norm(unit)
    if target == "SaO2":
        if unit_key != "%":
            raise ValueError(f"SaO2 en % attendue, reçu {unit!r}")
        return SAO2_GAIN * np.asarray(signal, dtype=np.float64) + SAO2_BASELINE
    if target in ("ABD", "CHEST", "AIRFLOW"):
        raise ValueError(f"{target} : échelle non comparable entre appareils, règle à décider")
    if unit_key not in VOLTAGE_TO_UV:
        raise ValueError(f"{target} : unité de tension attendue, reçu {unit!r}")
    return VOLTAGE_TO_UV[unit_key] * np.asarray(signal, dtype=np.float64)
