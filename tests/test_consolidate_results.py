"""Consolidation fidèle des scores existants, sans inférence ni remplacement des sources."""

import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.consolidate_results import consolidate, main, render_csv, render_markdown


@pytest.fixture
def baseline():
    return {
        "train_subjects": ["a", "b"],
        "test_subjects": ["c"],
        "seed": 42,
        "probes": {
            "logistic": {
                "test": {
                    "accuracy": 0.75,
                    "cohen_kappa": 0.5,
                    "macro_f1": 0.7,
                    "confusion_matrix": [[2, 1], [0, 1]],
                    "ci95": {"cohen_kappa": [0.2, 0.8], "n_subjects": 1},
                }
            }
        },
    }


def save(tmp_path, payload, name="run.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_keeps_exact_scores_and_unknown_historical_fields(tmp_path, baseline):
    path = save(tmp_path, baseline)
    before = path.read_bytes()
    rows = consolidate([path])
    assert rows[0]["cohen_kappa"] == 0.5
    assert rows[0]["test_epochs"] == 4
    assert rows[0]["roc_auc_macro"] is None
    assert rows[0]["preprocessing_version"] is None
    assert rows[0]["source_sha256"] == hashlib.sha256(before).hexdigest()
    csv_text = render_csv(rows)
    assert "\r" not in csv_text
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert csv_rows[0]["roc_auc_macro"] == ""
    assert csv_rows[0]["cohen_kappa"] == "0.5"
    assert "non renseigné" in render_markdown(rows)
    assert path.read_bytes() == before


def test_confirmation_and_rehearsal_stay_distinct(tmp_path, baseline):
    metrics = baseline["probes"]["logistic"]["test"]
    confirmation = {
        "status": "terminé",
        "kind": "confirmation sur les sujets réservés",
        "n_evaluated": 1,
        "n_requested": 1,
        "n_epochs": 4,
        "secondary": {"metrics": {"logistic": metrics}},
    }
    first = save(tmp_path, confirmation, "confirmation.json")
    rehearsal = save(tmp_path, confirmation | {"kind": "répétition sur des sujets déjà ouverts"}, "rehearsal.json")
    rows = consolidate([first, rehearsal])
    assert rows[0]["kind"] == "confirmation sur les sujets réservés"
    assert rows[1]["kind"] == "répétition sur des sujets déjà ouverts"
    assert rows[0]["train_nights"] is None
    assert rows[0]["seed"] is None


@pytest.mark.parametrize("problem", ["nan", "inf", "range", "bool", "ci", "overlap", "duplicates", "epochs", "status"])
def test_rejects_inconsistent_sources(tmp_path, baseline, problem):
    metrics = baseline["probes"]["logistic"]["test"]
    if problem in ("nan", "inf", "range", "bool"):
        metrics["macro_f1"] = {"nan": float("nan"), "inf": float("inf"), "range": 1.1, "bool": True}[problem]
    elif problem == "ci":
        metrics["ci95"]["cohen_kappa"] = [0.8, 0.2]
    elif problem == "overlap":
        baseline["test_subjects"] = ["a"]
    elif problem == "duplicates":
        baseline["test_subjects"] = ["c", "c"]
    elif problem == "epochs":
        second = copy.deepcopy(metrics)
        second["confusion_matrix"][0][0] += 1
        baseline["probes"]["sequence"] = {"test": second}
    else:
        baseline["status"] = "en cours"
    path = save(tmp_path, baseline)
    with pytest.raises(ValueError, match="run.json"):
        consolidate([path])


def test_incomplete_confirmation_never_looks_completed(tmp_path, baseline):
    payload = {
        "kind": "confirmation sur les sujets réservés",
        "n_evaluated": 1,
        "n_requested": 1,
        "n_epochs": 4,
        "secondary": {"metrics": {"logistic": baseline["probes"]["logistic"]["test"]}},
    }
    with pytest.raises(ValueError, match="Statut terminé"):
        consolidate([save(tmp_path, payload)])


def test_cli_failure_preserves_previous_outputs(tmp_path, baseline, monkeypatch):
    source = save(tmp_path, baseline)
    bad = save(tmp_path, {"status": "en cours"}, "bad.json")
    output = tmp_path / "result.csv"
    output.write_text("ancien tableau")
    monkeypatch.setattr(sys, "argv", ["consolidate", str(source), str(bad), "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == "ancien tableau"


def test_cli_never_overwrites_input_or_merges_outputs(tmp_path, baseline, monkeypatch):
    source = save(tmp_path, baseline)
    before = source.read_bytes()
    for outputs in (["--out", str(source)], ["--out", str(tmp_path / "x"), "--markdown", str(tmp_path / "x")]):
        monkeypatch.setattr(sys, "argv", ["consolidate", str(source), *outputs])
        with pytest.raises(SystemExit):
            main()
    assert source.read_bytes() == before
    assert not (tmp_path / "x").exists()
    with pytest.raises(ValueError, match="plusieurs fois"):
        consolidate([source, source])


def test_cli_runs_without_model_dependencies(tmp_path, baseline):
    source = save(tmp_path, baseline)
    csv_path, md_path = tmp_path / "table.csv", tmp_path / "table.md"
    script = Path(__file__).resolve().parents[1] / "scripts/consolidate_results.py"
    # -S retire site-packages : ni torch, ni sklearn, ni numpy ne sont nécessaires.
    result = subprocess.run(
        [sys.executable, "-S", str(script), str(source), "--out", str(csv_path), "--markdown", str(md_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(csv.DictReader(csv_path.open()))) == 1
    assert "0.500 [0.200 ; 0.800]" in md_path.read_text()
