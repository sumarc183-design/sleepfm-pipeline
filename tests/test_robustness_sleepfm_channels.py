import numpy as np

from scripts.robustness_sleepfm_channels import CONDITIONS, degrade, through_rate
from src.data.extract_physionet2018 import ALL_CHANNELS


def test_through_rate_keeps_slow_signal_and_removes_fast():
    t = np.arange(256 * 60) / 256
    slow, fast = np.sin(2 * np.pi * 0.25 * t), np.sin(2 * np.pi * 20 * t)
    assert np.allclose(through_rate(slow, 8)[512:-512], slow[512:-512], atol=0.02)
    assert np.abs(through_rate(fast, 8)[512:-512]).max() < 0.05


def test_degrade_zeroes_and_copies():
    night = np.ones((len(ALL_CHANNELS), 256 * 60))
    out = degrade(night, CONDITIONS["ucddb_like"])
    assert out[ALL_CHANNELS.index("F3-M2")].max() == 0 and out[ALL_CHANNELS.index("C3-M2")].min() == 1
    assert night.min() == 1  # l'original n'est pas modifié
