import numpy as np
import pytest

from src.data.nsrr_channels import SLEEPFM_CHANNELS, channel_plan, find_signal, to_physionet_scale
from src.models.embeddings import ALL_CHANNELS

MESA = ["EKG", "EOG-L", "EOG-R", "EMG", "EEG1", "EEG2", "EEG3", "Pres", "Flow", "Snore",
        "Thor", "Abdo", "Leg", "SpO2", "Pleth", "Pos"]  # fmt: skip
SHHS1 = ["SaO2", "H.R.", "EEG(sec)", "ECG", "EMG", "EOG(L)", "EOG(R)", "EEG", "THOR RES",
         "ABDO RES", "POSITION", "LIGHT", "AIRFLOW", "OX stat"]  # fmt: skip


def test_sleepfm_order_matches_encoder():
    assert list(SLEEPFM_CHANNELS) == ALL_CHANNELS


def test_mesa_sleepfm_plan():
    plan = channel_plan(MESA, "mesa", "sleepfm")
    assert plan["indices"]["C4-M1"] == MESA.index("EEG3")
    assert plan["indices"]["SaO2"] == MESA.index("SpO2")
    assert plan["zero_filled"] == ["F3-M2", "F4-M1", "C3-M2", "O1-M2", "O2-M1"]
    assert plan["uncertain"] == ["AIRFLOW"]


def test_osf_keeps_physionet_zeros():
    plan = channel_plan(MESA, "mesa", "osf")
    assert plan["indices"]["EMG_LLeg"] is None and plan["indices"]["SN"] is None
    assert plan["indices"]["EOG_E2_A1"] is None  # nul sur PhysioNet, même si EOG-R existe
    assert plan["lost_vs_physionet"] == ["EEG_C3_A2"]
    assert channel_plan(SHHS1, "shhs1", "osf")["lost_vs_physionet"] == []


def test_missing_or_duplicate_label_rejected():
    with pytest.raises(ValueError):
        find_signal(["EKG"], "mesa", "SPO2")
    with pytest.raises(ValueError):
        find_signal(["EEG(sec)", "EEG2"], "shhs1", "EEG_C3_M2")
    assert find_signal([" eeg(SEC) "], "shhs1", "EEG_C3_M2") == 0


def test_physionet_scale():
    assert to_physionet_scale(np.array([1e-6]), "V", "C4-M1")[0] == pytest.approx(1.0)
    assert to_physionet_scale(np.array([0.5]), "mV", "ECG")[0] == pytest.approx(500.0)
    assert to_physionet_scale(np.array([100.0]), "%", "SaO2")[0] == pytest.approx(32767.0)
    with pytest.raises(ValueError):
        to_physionet_scale(np.array([1.0]), "", "ABD")
    with pytest.raises(ValueError):
        to_physionet_scale(np.array([1.0]), "uV", "SaO2")


def test_physionet_median_scale_matches_archived_result():
    import json
    from pathlib import Path

    from src.data.nsrr_channels import PHYSIONET_MEDIAN_SCALE

    root = Path(__file__).resolve().parents[1] / "results"
    summary = json.loads((root / "respiratory_amplitude_physionet2018.json").read_text())["summary"]
    summary |= json.loads((root / "eeg_amplitude_physionet2018.json").read_text())["summary"]
    assert len(PHYSIONET_MEDIAN_SCALE) == 8
    for target, value in PHYSIONET_MEDIAN_SCALE.items():
        assert summary[target]["robust_scale"]["median"] == value


def test_physionet_median_scale_rule():
    from src.data.nsrr_channels import PHYSIONET_MEDIAN_SCALE, physionet_median_scale, robust_scale

    signal = np.sin(np.linspace(0, 60, 3000)) * 0.003
    scaled = physionet_median_scale("CHEST", signal, "")
    assert robust_scale(scaled) == pytest.approx(PHYSIONET_MEDIAN_SCALE["CHEST"])
    assert np.corrcoef(scaled, signal)[0, 1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        physionet_median_scale("ABD", np.zeros(100), "")
    with pytest.raises(ValueError):
        physionet_median_scale("SaO2", signal, "%")
