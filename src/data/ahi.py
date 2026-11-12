"""IAH de référence, indice prédit par nuit et statistiques de classement IAH ≥ 15."""

from __future__ import annotations

import numpy as np

from src.data.respiratory_events import TARGET_EVENTS

SLEEP_STAGES = ("N1", "N2", "N3", "R")
STAGE_NOTES = frozenset(("W", *SLEEP_STAGES))
SEVERITY_EDGES = (5.0, 15.0, 30.0)
AHI_CUTOFF = 15.0
MIN_SLEEP_H = 1.0
WINDOW_S = 30.0


def reference_ahi(samples, notes, sample_rate: float) -> dict:
    """Même règle que l'étude de faisabilité (`compter.py`).

    Sommeil : durée entre deux annotations de stade successives, si la première est
    N1, N2, N3 ou R (la dernière annotation de stade ne compte pas). Événements :
    débuts « (resp_… » des quatre types, comptés séparément, sans fusion.
    """
    if sample_rate <= 0 or len(samples) != len(notes):
        raise ValueError("Annotations ou fréquence invalides")
    samples = np.asarray(samples, dtype=np.int64)
    stages = [(int(s), n) for s, n in zip(samples, notes, strict=True) if n in STAGE_NOTES]
    sleep_s = sum((s1 - s0) / sample_rate for (s0, n0), (s1, _) in zip(stages, stages[1:]) if n0 != "W")
    counts = {kind: 0 for kind in TARGET_EVENTS}
    for note in notes:
        if note.startswith("(") and note[1:] in counts:
            counts[note[1:]] += 1
    sleep_h = sleep_s / 3600
    total = sum(counts.values())
    return {
        "sleep_h": sleep_h,
        "event_counts": counts,
        "events": total,
        "ahi": total / sleep_h if sleep_h > 0 else float("nan"),
    }


def positive_rate(probability, threshold: float) -> tuple[float, float, int]:
    """Fenêtres positives par heure de fenêtres, somme des probabilités par heure, effectif."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or not len(probability) or not np.isfinite(probability).all():
        raise ValueError("Probabilités invalides")
    hours = len(probability) * WINDOW_S / 3600
    return float((probability >= threshold).sum() / hours), float(probability.sum() / hours), len(probability)


def fit_line(x, y) -> tuple[float, float]:
    """Moindres carrés ordinaires y = a + b·x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3 or np.ptp(x) == 0:
        raise ValueError("Calibration impossible")
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def severity(values) -> np.ndarray:
    """0 : < 5, 1 : 5–15, 2 : 15–30, 3 : ≥ 30."""
    return np.searchsorted(SEVERITY_EDGES, np.asarray(values, dtype=float), side="right")


def sensitivity_specificity(predicted, truth) -> tuple[float, float]:
    predicted, truth = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    if predicted.shape != truth.shape or truth.all() or not truth.any():
        raise ValueError("Deux classes de référence requises")
    return float((predicted & truth).sum() / truth.sum()), float((~predicted & ~truth).sum() / (~truth).sum())


def auc(score, truth) -> float:
    """AUC de Mann-Whitney, ex æquo comptés 0,5."""
    score, truth = np.asarray(score, dtype=float), np.asarray(truth, dtype=bool)
    if score.shape != truth.shape or truth.all() or not truth.any():
        raise ValueError("Deux classes de référence requises")
    positive, negative = score[truth], score[~truth]
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def linear_weighted_kappa(a, b, n_classes: int = 4) -> float:
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.shape != b.shape or not len(a) or a.min() < 0 or b.min() < 0 or max(a.max(), b.max()) >= n_classes:
        raise ValueError("Classes invalides")
    observed = np.zeros((n_classes, n_classes))
    np.add.at(observed, (a, b), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    grid = np.arange(n_classes)
    weights = np.abs(grid[:, None] - grid[None, :]) / (n_classes - 1)
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator else float("nan")


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def bland_altman(estimated, reference) -> dict:
    difference = np.asarray(estimated, dtype=float) - np.asarray(reference, dtype=float)
    bias, sd = float(difference.mean()), float(difference.std(ddof=1))
    return {"bias": bias, "lower": bias - 1.96 * sd, "upper": bias + 1.96 * sd}


def percentile_ci(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_draws(n_items: int, n_boot: int, seed: int) -> np.ndarray:
    """Indices tirés avec remise, mêmes tirages pour tous les modèles."""
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))
