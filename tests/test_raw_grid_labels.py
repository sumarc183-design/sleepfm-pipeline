"""Stade de l'expert par fenêtre de la grille fixe : seulement les fenêtres
d'un seul stade défini sont notées, les autres sont comptées à part."""

import numpy as np
import pytest

from scripts.generate_synthetic_physionet2018 import STAGE_CYCLE, generate_subject
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL
from src.data.raw_grid_labels import UNSCORED, grid_labels

CYCLE = [STAGE_NAME_TO_LABEL[s] for s in STAGE_CYCLE]


def _night(tmp_path, lead_in_sec, n_epochs=6):
    generate_subject(str(tmp_path), "tr00-0001", 32, n_epochs, 30, lead_in_sec, np.random.default_rng(0))
    return tmp_path / "tr00-0001"


def test_aligned_grid_gets_every_stage(tmp_path):
    y, counts = grid_labels(_night(tmp_path, 0), 6)
    assert y.tolist() == [CYCLE[i % len(CYCLE)] for i in range(6)]
    assert counts == {"scored": 6, "mixed_stages": 0, "undefined": 0}


def test_undefined_start_is_counted_not_scored(tmp_path):
    y, counts = grid_labels(_night(tmp_path, 30), 7)
    assert y[0] == UNSCORED and y[1:].tolist() == [CYCLE[i % len(CYCLE)] for i in range(6)]
    assert counts == {"scored": 6, "mixed_stages": 0, "undefined": 1}


def test_windows_across_two_stages_are_not_scored(tmp_path):
    y, counts = grid_labels(_night(tmp_path, 2), 6)
    assert (y == UNSCORED).all() and counts == {"scored": 0, "mixed_stages": 6, "undefined": 0}


def test_too_many_windows_is_refused(tmp_path):
    with pytest.raises(ValueError, match="dépassent"):
        grid_labels(_night(tmp_path, 0), 7)
