import numpy as np
import pytest

from src.data.ahi import (
    auc,
    bland_altman,
    bootstrap_draws,
    fit_line,
    linear_weighted_kappa,
    positive_rate,
    reference_ahi,
    sensitivity_specificity,
    severity,
)


def test_reference_ahi_follows_feasibility_rule():
    fs = 200
    hour = 3600 * fs
    samples = [0, hour, hour + 10, hour + 20, 2 * hour, 2 * hour + 5, 3 * hour, 4 * hour]
    notes = ["W", "N2", "(resp_hypopnea", "resp_hypopnea)", "R", "(resp_obstructiveapnea", "W", "N1"]
    result = reference_ahi(samples, notes, fs)
    # Sommeil : N2 puis R (2 h) ; la dernière annotation de stade (N1) ne compte pas.
    assert result["sleep_h"] == pytest.approx(2.0)
    assert result["events"] == 2
    assert result["event_counts"]["resp_hypopnea"] == 1
    assert result["ahi"] == pytest.approx(1.0)


def test_reference_ahi_ignores_other_events_and_counts_without_merging():
    notes = ["N2", "(arousal_rera", "(resp_centralapnea", "(resp_centralapnea", "W"]
    result = reference_ahi([0, 1, 2, 3, 3600], notes, 1)
    assert result["events"] == 2
    assert result["sleep_h"] == pytest.approx(1.0)


def test_reference_ahi_without_sleep_is_nan():
    assert np.isnan(reference_ahi([0, 10], ["W", "W"], 1)["ahi"])


def test_positive_rate_per_window_hour():
    probability = np.array([0.1, 0.9, 0.8, 0.2] * 30)  # 120 fenêtres = 1 h
    rate, total, n = positive_rate(probability, 0.8)
    assert (rate, n) == (60.0, 120)
    assert total == pytest.approx(probability.sum())


def test_fit_line_recovers_exact_relation():
    x = np.array([0.0, 1.0, 2.0, 5.0])
    assert fit_line(x, 3 + 2 * x) == pytest.approx((3.0, 2.0))
    with pytest.raises(ValueError):
        fit_line(np.ones(4), x)


def test_severity_edges():
    assert severity([0, 4.99, 5, 14.99, 15, 29.99, 30, -2]).tolist() == [0, 0, 1, 1, 2, 2, 3, 0]


def test_sensitivity_specificity_and_auc():
    truth = np.array([True, True, False, False])
    assert sensitivity_specificity(np.array([True, False, False, True]), truth) == (0.5, 0.5)
    assert auc(np.array([3, 2, 2, 1]), truth) == pytest.approx(0.875)
    with pytest.raises(ValueError):
        auc(np.ones(2), np.array([True, True]))


def test_linear_weighted_kappa():
    a = np.array([0, 1, 2, 3, 0, 1])
    assert linear_weighted_kappa(a, a) == pytest.approx(1.0)
    shifted = linear_weighted_kappa(a, np.minimum(a + 1, 3))
    far = linear_weighted_kappa(a, 3 - a)
    assert far < shifted < 1


def test_bland_altman_and_draws():
    result = bland_altman([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert result["bias"] == pytest.approx(2.0)
    assert result["upper"] - result["bias"] == pytest.approx(1.96)
    draws = bootstrap_draws(5, 3, 42)
    assert draws.shape == (3, 5) and np.array_equal(draws, bootstrap_draws(5, 3, 42))
