"""Stade annoté de chaque fenêtre de la grille fixe, pour évaluer seulement.

La prédiction sur PSG brute (`RawNightDataset`) découpe la nuit en fenêtres de
30 s depuis le début de l'enregistrement, sans lire les annotations. Pour mesurer
ses performances, ce module lit à part le fichier -arousal.mat et donne à chaque
fenêtre le stade de l'expert, seulement si la fenêtre entière porte un seul stade
défini. Les fenêtres à cheval sur deux stades ou dans une période « undefined »
reçoivent -1 et sont comptées à part : elles ne sont pas notées.

Ces labels ne servent jamais à choisir les fenêtres ni à entraîner un modèle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.epoch_dataset import STAGE_NAME_TO_LABEL
from src.data.extract_physionet2018 import annotation_problem, extract_labels, import_signal_names

EPOCH_DURATION_S = 30
UNSCORED = -1


def grid_labels(subject_dir: Path, n_epochs: int) -> tuple[np.ndarray, dict]:
    """(label par fenêtre, -1 si non notée ; décompte des fenêtres non notées)."""
    subject_dir = Path(subject_dir)
    subject = subject_dir.name
    names, sample_rate, n_samples = import_signal_names(str(subject_dir / f"{subject}.hea"))
    labels, stage_names = extract_labels(str(subject_dir / f"{subject}-arousal.mat"))
    # Le signal n'est pas relu ici : l'en-tête suffit pour la longueur attendue.
    problem = annotation_problem(labels, stage_names, (len(names), n_samples), (len(names), n_samples))
    if problem:
        raise ValueError(f"{subject} : annotations invalides ({problem})")
    window = sample_rate * EPOCH_DURATION_S
    if n_epochs * window > n_samples:
        raise ValueError(f"{subject} : {n_epochs} fenêtres dépassent l'enregistrement")
    stage_of_column = np.array([STAGE_NAME_TO_LABEL.get(name, UNSCORED) for name in stage_names])
    per_sample = stage_of_column[labels[: n_epochs * window].argmax(axis=1)]
    per_sample[labels[: n_epochs * window, stage_names.index("undefined")] == 1] = UNSCORED
    windows = per_sample.reshape(n_epochs, window)
    uniform = (windows == windows[:, :1]).all(axis=1)
    y = np.where(uniform, windows[:, 0], UNSCORED).astype(np.int64)
    counts = {
        "scored": int((y != UNSCORED).sum()),
        "mixed_stages": int((~uniform).sum()),
        "undefined": int((uniform & (windows[:, 0] == UNSCORED)).sum()),
    }
    return y, counts
