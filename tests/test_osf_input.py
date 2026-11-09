"""Format OSF : ordonner sans inventer les canaux ni transformer la physiologie."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.check_osf_input import check_archive, main
from src.data.osf_input import OSF_CHANNELS, prepare_osf_epochs, resolve_osf_channels
from src.models.osf_vit1d import OSFViT1D


@pytest.fixture
def epochs():
    return np.broadcast_to(np.arange(12, dtype=np.float64)[None, :, None], (2, 12, 1920)).copy()


def test_reorders_by_name_and_preserves_signal_without_mutating_input(epochs):
    shuffled = epochs[:, ::-1, :]
    shuffled.setflags(write=False)
    ordered, metadata = prepare_osf_epochs(shuffled, list(OSF_CHANNELS)[::-1], 64)
    np.testing.assert_array_equal(ordered[0, :, 0], np.arange(12))
    assert ordered.dtype == np.float32 and ordered.flags.c_contiguous
    assert not np.shares_memory(ordered, epochs)
    assert metadata["source_indices"] == list(range(11, -1, -1))
    assert metadata["physiological_equivalence_verified"] is False
    assert metadata["amplitude_normalization_applied"] is False
    ordered.fill(100)
    assert epochs[0, 0, 0] == 0


def test_aliases_require_an_explicit_mapping(epochs):
    names = list(OSF_CHANNELS)
    names[-2] = "C3-M2"
    with pytest.raises(ValueError, match="EEG_C3_A2"):
        prepare_osf_epochs(epochs, names, 64)
    ordered, metadata = prepare_osf_epochs(epochs, names, 64, {"EEG_C3_A2": "C3-M2"})
    np.testing.assert_array_equal(ordered[0, :, 0], np.arange(12))
    assert metadata["explicit_mapping"] == {"EEG_C3_A2": "C3-M2"}
    assert metadata["physiological_equivalence_verified"] is False


def test_missing_channels_are_never_filled_with_zeros(epochs):
    with pytest.raises(ValueError, match="EMG_LLeg"):
        prepare_osf_epochs(
            epochs[:, [i for i in range(12) if i != 2]], [c for c in OSF_CHANNELS if c != "EMG_LLeg"], 64
        )


def test_extra_channels_are_excluded_with_their_origin_recorded(epochs):
    signal = np.concatenate([np.full((2, 1, 1920), 999.0), epochs], axis=1)
    ordered, metadata = prepare_osf_epochs(signal, ["EXTRA", *OSF_CHANNELS], 64)
    np.testing.assert_array_equal(ordered[0, :, 0], np.arange(12))
    assert metadata["source_indices"] == list(range(1, 13))
    assert metadata["source_channels"][0] == "EXTRA"


@pytest.mark.parametrize("problem", ["names", "empty_names", "duplicate", "unknown_target", "reuse", "mapping_type"])
def test_ambiguous_channel_lists_and_mappings_are_refused(problem):
    names, mapping = list(OSF_CHANNELS), None
    if problem == "names":
        names[0] = 42
    elif problem == "empty_names":
        names[0] = " "
    elif problem == "duplicate":
        names[0] = names[1]
    elif problem == "unknown_target":
        mapping = {"AUTRE": "ECG"}
    elif problem == "reuse":
        mapping = {"EMG_Chin": "ECG"}
    else:
        mapping = ["ECG"]
    with pytest.raises(ValueError):
        resolve_osf_channels(names, mapping)


@pytest.mark.parametrize("rate", [256, 32, 0, float("nan"), float("inf"), "64", True])
def test_wrong_sampling_rate_is_not_silently_converted(epochs, rate):
    with pytest.raises(ValueError, match="64 Hz"):
        prepare_osf_epochs(epochs, list(OSF_CHANNELS), rate)


@pytest.mark.parametrize("problem", ["short", "channels", "empty", "integer", "nan", "overflow"])
def test_bad_signals_are_refused(epochs, problem):
    if problem == "short":
        epochs = epochs[:, :, :1919]
    elif problem == "channels":
        epochs = epochs[:, :11, :]
    elif problem == "empty":
        epochs = epochs[:0]
    elif problem == "integer":
        epochs = epochs.astype(np.int16)
    elif problem == "nan":
        epochs[0, 0, 0] = np.nan
    else:
        epochs[0, 0, 0] = 1e100
    with pytest.raises(ValueError):
        prepare_osf_epochs(epochs, list(OSF_CHANNELS), 64)


def test_prepared_epochs_can_enter_the_backbone(epochs):
    ordered, _ = prepare_osf_epochs(epochs, list(OSF_CHANNELS), 64)
    model = OSFViT1D(12, 1920, 64, width=8, depth=1, heads=2, dim_head=4, mlp_dim=16).eval()
    with torch.no_grad():
        embedding = model(torch.from_numpy(ordered))
    assert embedding.shape == (2, 8) and torch.isfinite(embedding).all()


def test_cli_checks_without_importing_torch_or_writing_signals(tmp_path, epochs):
    archive, report = tmp_path / "epochs.npz", tmp_path / "report.json"
    np.savez(archive, epochs=epochs, channels=np.array(OSF_CHANNELS), sample_rate=64)
    script = Path(__file__).resolve().parents[1] / "scripts/check_osf_input.py"
    code = (
        "import sys,runpy; sys.modules['torch']=None; sys.argv=sys.argv[1:]; "
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script), str(archive), "--report", str(report)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(report.read_text())
    assert metadata["format_valid"] is True and metadata["epoch_count"] == 2
    assert metadata["channels"] == list(OSF_CHANNELS)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["epochs.npz", "report.json"]


def test_invalid_archive_preserves_existing_report_and_inputs(tmp_path, epochs, monkeypatch):
    archive, report = tmp_path / "bad.npz", tmp_path / "report.json"
    np.savez(archive, epochs=epochs, channels=np.array(OSF_CHANNELS), sample_rate=256)
    before = archive.read_bytes()
    report.write_text("ancien rapport")
    monkeypatch.setattr(sys, "argv", ["check", str(archive), "--report", str(report)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert report.read_text() == "ancien rapport" and archive.read_bytes() == before
    monkeypatch.setattr(sys, "argv", ["check", str(archive), "--report", str(archive)])
    with pytest.raises(SystemExit):
        main()
    assert archive.read_bytes() == before


@pytest.mark.parametrize("problem", ["npy", "missing", "object", "rate_vector"])
def test_bad_archive_schemas_are_refused(tmp_path, epochs, problem):
    path = tmp_path / "data.npz"
    arrays = {"epochs": epochs, "channels": np.array(OSF_CHANNELS), "sample_rate": np.array(64)}
    if problem == "npy":
        path = tmp_path / "data.npy"
        np.save(path, epochs)
    else:
        if problem == "missing":
            arrays.pop("epochs")
        elif problem == "object":
            arrays["channels"] = np.array(OSF_CHANNELS, dtype=object)
        else:
            arrays["sample_rate"] = np.array([64])
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        check_archive(path)
