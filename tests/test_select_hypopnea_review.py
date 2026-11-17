"""Tirage des fenêtres à faire relire : effectifs, une fenêtre par nuit et par groupe, reproductible."""

import numpy as np
import pytest

from scripts import select_hypopnea_review as sel


def synthetic(n_nights=30, per_night=40, seed=0):
    rng = np.random.default_rng(seed)
    subject = np.repeat([f"tr00-{i:04d}" for i in range(n_nights)], per_night)
    event_group = rng.choice(["aucun", "resp_hypopnea", "resp_obstructiveapnea"], size=len(subject))
    flagged = rng.random(len(subject)) < 0.4
    return subject, event_group, flagged


def test_draw_respects_groups_sizes_and_one_window_per_night():
    subject, event_group, flagged = synthetic()
    picks = sel.draw(subject, event_group, flagged)
    assert [p["order"] for p in picks] == list(range(1, 41))
    for group, size in sel.SIZES.items():
        rows = [p["row"] for p in picks if p["group"] == group]
        assert len(rows) == size and len(set(subject[rows])) == size
        assert set(rows) <= set(sel.eligible(group, event_group, flagged).tolist())
    rows_a = [p["row"] for p in picks if p["group"] == "A"]
    assert (event_group[rows_a] == "resp_hypopnea").all() and not flagged[rows_a].any()


def test_draw_is_reproducible_and_shuffled():
    subject, event_group, flagged = synthetic()
    first, second = sel.draw(subject, event_group, flagged), sel.draw(subject, event_group, flagged)
    assert first == second
    assert [p["group"] for p in first] != sorted(p["group"] for p in first)


def test_draw_refuses_too_few_nights():
    subject, event_group, flagged = synthetic(n_nights=15)
    with pytest.raises(ValueError, match="nuits éligibles"):
        sel.draw(subject, event_group, flagged)
