import numpy as np
import pytest

from src.data.nsrr_channels import OSF_SOURCES, SLEEPFM_SOURCES
from src.data.nsrr_signals import EPOCH_SAMPLES, nsrr_night, read_edf
from src.data.osf_physionet2018 import night_to_osf
from src.models.embeddings import ALL_CHANNELS

DURATION_S = 90  # 3 époques
SHHS = [  # libellé, unité, fréquence, plage physique
    ("SaO2", "%", 1, (0.0, 100.0)),
    ("EEG(sec)", "uV", 125, (-250.0, 250.0)),
    ("ECG", "mV", 125, (-2.0, 2.0)),
    ("EMG", "uV", 125, (-100.0, 100.0)),
    ("EOG(L)", "uV", 50, (-500.0, 500.0)),
    ("EEG", "uV", 125, (-250.0, 250.0)),
    ("THOR RES", "", 10, (-1.0, 1.0)),
    ("ABDO RES", "", 10, (-1.0, 1.0)),
    ("AIRFLOW", "", 10, (-1.0, 1.0)),
]


def write_edf(path, channels, signals, record_s=1):
    n_records = DURATION_S // record_s
    head = b"0".ljust(8) + b"".ljust(160) + b"01.01.01".ljust(8) + b"00.00.00".ljust(8)
    head += str(256 * (len(channels) + 1)).encode().ljust(8) + b"".ljust(44)
    head += str(n_records).encode().ljust(8) + str(record_s).encode().ljust(8) + str(len(channels)).encode().ljust(4)
    cols = [
        [c[0].encode().ljust(16) for c in channels],
        [b"".ljust(80)] * len(channels),
        [c[1].encode().ljust(8) for c in channels],
        [str(c[3][0]).encode().ljust(8) for c in channels],
        [str(c[3][1]).encode().ljust(8) for c in channels],
        [b"-32768".ljust(8)] * len(channels),
        [b"32767".ljust(8)] * len(channels),
        [b"".ljust(80)] * len(channels),
        [str(c[2] * record_s).encode().ljust(8) for c in channels],
        [b"".ljust(32)] * len(channels),
    ]
    head += b"".join(b"".join(col) for col in cols)
    digital = []
    for (_, _, rate, (lo, hi)), signal in zip(channels, signals, strict=True):
        d = np.round((signal - lo) * 65535 / (hi - lo) - 32768).astype("<i2")
        digital.append(d.reshape(n_records, rate * record_s))
    path.write_bytes(head + np.concatenate(digital, axis=1).tobytes())


def synthetic(rate, kind):
    t = np.arange(DURATION_S * rate) / rate
    if kind == "%":
        return np.full(t.size, 95.0)
    return 0.4 * (np.sin(2 * np.pi * 0.25 * t) if rate <= 10 else np.sin(2 * np.pi * 2 * t)) * 100


@pytest.fixture
def shhs_edf(tmp_path):
    path = tmp_path / "shhs1-200001.edf"
    signals = [synthetic(c[2], c[1]) / (100 if c[1] in ("", "mV") else 1) for c in SHHS]
    write_edf(path, SHHS, signals)
    return path


def test_read_edf_physical_values(shhs_edf):
    edf = read_edf(shhs_edf)
    assert edf["labels"][0] == "SaO2" and edf["units"][2] == "mV"
    assert edf["sample_rates"].tolist() == [c[2] for c in SHHS] and edf["duration_s"] == DURATION_S
    assert edf["signals"][0] == pytest.approx(np.full(DURATION_S, 95.0), abs=0.01)


def test_nsrr_night_physionet_format(shhs_edf):
    epochs, info = nsrr_night(shhs_edf, "shhs1", lambda target, signal, unit: signal)
    assert epochs.shape == (3, 13, EPOCH_SAMPLES) and epochs.dtype == np.float32
    assert info["zero_filled"] == ["F3-M2", "F4-M1", "O1-M2", "O2-M1"]
    assert not epochs[:, ALL_CHANNELS.index("F3-M2")].any()
    assert epochs[:, ALL_CHANNELS.index("SaO2")].mean() == pytest.approx(655.35 * 95 - 32768, abs=10)
    ecg = epochs[:, ALL_CHANNELS.index("ECG")].reshape(-1)
    assert np.abs(ecg).max() == pytest.approx(40.0 * 1000 / 100, rel=0.05)  # 0,4 mV -> 400 µV
    osf, flat = night_to_osf(epochs)
    assert osf.shape == (3, 12, 1920) and flat == []


def test_respiratory_rule_required(shhs_edf):
    with pytest.raises(ValueError):
        nsrr_night(shhs_edf, "shhs1", None)


def test_osf_sources_subset_of_sleepfm():
    for cohort, sources in OSF_SOURCES.items():
        assert set(sources.values()) <= set(SLEEPFM_SOURCES[cohort].values())


def test_truncated_edf_rejected(shhs_edf):
    shhs_edf.write_bytes(shhs_edf.read_bytes()[:-10])
    with pytest.raises(ValueError):
        read_edf(shhs_edf)


@pytest.mark.parametrize("rate", [200, 125, 10, 1])
def test_resampling_matches_physionet_path(rate):
    import mne

    from src.data.nsrr_signals import to_256

    x = np.random.default_rng(0).normal(size=(1, rate * 120))
    raw = mne.io.RawArray(x, mne.create_info(1, float(rate), "eeg"), verbose=False).resample(256, verbose=False)
    assert np.array_equal(raw.get_data()[0], to_256(x[0], rate))
