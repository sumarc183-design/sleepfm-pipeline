"""La démo doit présenter une vraie nuit de test et refuser les sources incohérentes."""

import json

import numpy as np
import pytest

from scripts.demo_night import night_data, render
from src.data.cache_contract import file_sha256


def archive(path):
    proba = np.eye(5)[[0, 1, 2]]
    np.savez(
        path,
        subject=["night", "night", "night"],
        y_true=[0, 1, 2],
        epoch_idx=[0, 1, 2],
        onset_s=[0.0, 30.0, 60.0],
        pred_logistic=[0, 1, 2],
        pred_sequence=[0, 1, 2],
        proba_logistic=proba,
        proba_sequence=proba,
        classes_logistic=np.arange(5),
        classes_sequence=np.arange(5),
    )


def test_demo_defaults_to_first_test_identifier_and_escapes_embedded_json(tmp_path):
    path = tmp_path / "stages.npz"
    archive(path)
    data = night_data(path, None, None, None)
    assert data["subject"] == "night" and data["respiratory"] is None
    assert data["probes"]["sequence"]["pred"] == [0, 1, 2]
    data["sources"]["injection"] = "</script><script>bad</script>"
    page = render(data)
    payload = page.split('<script id="night-data" type="application/json">')[1].split("</script>")[0]
    assert "</script>" not in payload
    assert json.loads(payload)["sources"]["injection"] == data["sources"]["injection"]
    assert "https://" not in page and "<iframe" not in page
    with pytest.raises(ValueError, match="absente"):
        night_data(path, "unknown", None, None)


def test_demo_refuses_nonmatching_respiratory_archive(tmp_path):
    path = tmp_path / "stages.npz"
    archive(path)
    report = tmp_path / "resp.json"
    predictions = tmp_path / "resp.predictions.npz"
    np.savez(predictions, subject=["night"], epoch_idx=[0], onset_s=[1.0], y_true=[1], probability=[0.8])
    payload = {
        "status": "terminé",
        "partitions": {"test": ["night"]},
        "predictions_sha256": file_sha256(predictions),
        "threshold": {"value": 0.5},
    }
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="alignées"):
        night_data(path, "night", report, None)
    payload["predictions_sha256"] = "different"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="empreinte"):
        night_data(path, "night", report, None)


def test_demo_marks_excluded_windows_as_missing_not_negative(tmp_path):
    path = tmp_path / "stages.npz"
    archive(path)
    report = tmp_path / "resp.json"
    predictions = tmp_path / "resp.predictions.npz"
    np.savez(
        predictions,
        subject=["night", "night"],
        epoch_idx=[0, 2],
        onset_s=[0.0, 60.0],
        y_true=[1, 0],
        probability=[0.8, 0.1],
    )
    report.write_text(
        json.dumps(
            {
                "status": "terminé",
                "partitions": {"test": ["night"]},
                "predictions_sha256": file_sha256(predictions),
                "threshold": {"value": 0.5},
            }
        )
    )
    data = night_data(path, None, report, None)
    assert data["respiratory"]["target"] == [1, None, 0]
    assert data["respiratory"]["probability"] == [0.8, None, 0.1]


def test_demo_adds_osf_only_for_the_same_reference_and_windows(tmp_path):
    stages = tmp_path / "stages.npz"
    archive(stages)
    respiratory = tmp_path / "resp.json"
    predictions = tmp_path / "resp.predictions.npz"
    arrays = {
        "subject": np.array(["night", "night"]),
        "epoch_idx": np.array([0, 2]),
        "onset_s": np.array([0.0, 60.0]),
        "y_true": np.array([1, 0]),
    }
    np.savez(predictions, **arrays, probability=[0.8, 0.1])
    respiratory.write_text(
        json.dumps(
            {
                "status": "terminé",
                "partitions": {"test": ["night"]},
                "threshold": {"value": 0.5},
                "predictions_sha256": file_sha256(predictions),
            }
        )
    )
    comparison = tmp_path / "comparison.json"
    comparative_predictions = tmp_path / "comparison.predictions.npz"
    np.savez(comparative_predictions, **arrays, probability_sleepfm=[0.8, 0.1], probability_osf=[0.9, 0.2])
    report = {
        "status": "terminé",
        "partitions": {"test": ["night"]},
        "baseline_sha256": file_sha256(respiratory),
        "thresholds": {"osf": 0.6},
        "artifacts": {"predictions.npz": {"sha256": file_sha256(comparative_predictions)}},
    }
    comparison.write_text(json.dumps(report))
    data = night_data(stages, None, respiratory, None, comparison)
    assert data["respiratory"]["osf_probability"] == [0.9, None, 0.2]
    assert data["respiratory"]["osf_threshold"] == 0.6
    report["baseline_sha256"] = "different"
    comparison.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="baseline différente"):
        night_data(stages, None, respiratory, None, comparison)
