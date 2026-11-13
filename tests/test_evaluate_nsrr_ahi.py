import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_nsrr_ahi import FROZEN_AHI_SHA256, PLAN, criterion, load_frozen, run
from tests.test_nsrr_annotations import document, scored

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/respiratory_ahi_physionet2018.json"
BASELINE = ROOT / "results/respiratory_baseline_physionet2018.json"
COMPARISON = ROOT / "results/respiratory_osf_vs_sleepfm_physionet2018.json"
needs_results = pytest.mark.skipif(not FROZEN.exists(), reason="résultats PhysioNet absents")


def night_xml(n_events: int, sleep_s: int = 7200) -> bytes:
    events = [scored("Stages|Stages", "Wake|0", 0, 600), scored("Stages|Stages", "Stage 2 sleep|2", 600, sleep_s)]
    events += [scored("Respiratory|Respiratory", "Hypopnea|Hypopnea", 700 + 60 * i, 20) for i in range(n_events)]
    return document(*events)


@pytest.fixture
def frozen():
    return load_frozen(FROZEN, BASELINE, COMPARISON)


@needs_results
def test_frozen_rule_is_checked(tmp_path, frozen):
    assert frozen["thresholds"]["sleepfm"] == pytest.approx(0.6934, abs=1e-4)
    with pytest.raises(ValueError):
        load_frozen(FROZEN, BASELINE, COMPARISON, expected_sha="0" * 64)


@needs_results
def test_run_end_to_end(tmp_path, frozen):
    emb, ann = tmp_path / "emb", tmp_path / "ann"
    emb.mkdir(), ann.mkdir()
    rng = np.random.default_rng(0)
    subjects = [f"mesa-sleep-{i:04d}" for i in range(12)]
    for i, subject in enumerate(subjects):
        n = 260
        np.savez(
            emb / f"{subject}.npz",
            X_sleepfm=rng.normal(size=(n, 1536)).astype(np.float32),
            X_osf=rng.normal(size=(n, 768)).astype(np.float32),
            epoch_idx=np.arange(n),
            onset_s=30.0 * np.arange(n),
        )
        (ann / f"{subject}-nsrr.xml").write_bytes(night_xml(n_events=5 + 10 * i))
    (ann / "mesa-sleep-0001-nsrr.xml").write_bytes(night_xml(n_events=3, sleep_s=1800))  # < 1 h : exclue
    (emb / "manifest.json").write_text(
        json.dumps(
            {
                "cohort": "mesa",
                "subjects_sha256": "x",
                "models": ["osf", "sleepfm"],
                "respiratory_rule": "physionet_median_v1",
                "sleepfm_checkpoint_sha256": frozen["checkpoints"]["sleepfm"],
                "osf_checkpoint_sha256": frozen["checkpoints"]["osf"],
                "preparation_version": 1,
            }
        )
    )
    (emb / "run_status.json").write_text(
        json.dumps({"done": subjects, "rejected": {"mesa-sleep-9999": "EDF tronqué"}, "missing": [], "complete": True})
    )
    official = tmp_path / "official.csv"
    official.write_text("nsrrid,ahi_a0h3\n" + "\n".join(f"{s},{3 * i}" for i, s in enumerate(subjects)) + "\n")
    args = Namespace(
        cohort="mesa",
        embeddings=emb,
        annotations_dir=ann,
        xml_pattern="{subject}-nsrr.xml",
        out=tmp_path / "out" / "nsrr_ahi_mesa",
        frozen_ahi=FROZEN,
        baseline=BASELINE,
        comparison=COMPARISON,
        plan=PLAN,
        official_csv=official,
        official_id_column="nsrrid",
        official_column="ahi_a0h3",
        expected_sha=FROZEN_AHI_SHA256,
    )
    report = run(args)
    assert report["nights"]["evaluated"] == 11
    assert set(report["nights"]["excluded"]) == {"mesa-sleep-0001", "mesa-sleep-9999"}
    assert set(report["success_criterion"]["models"]) == {"sleepfm", "osf"}
    assert report["official_reference"]["nights"] == 11
    with pytest.raises(ValueError):
        run(args)  # un seul passage


@needs_results
def test_incomplete_or_wrong_embeddings_refused(tmp_path, frozen):
    from scripts.evaluate_nsrr_ahi import check_embeddings

    (tmp_path / "manifest.json").write_text(json.dumps({"cohort": "mesa", "models": ["osf"]}))
    (tmp_path / "run_status.json").write_text(json.dumps({"complete": True, "missing": []}))
    with pytest.raises(ValueError):
        check_embeddings(tmp_path, "mesa", frozen)


def test_criterion():
    ci = {
        "ci95": {
            "sleepfm": {"sensitivity": [0.61, 0.9], "specificity": [0.7, 0.9]},
            "osf": {"sensitivity": [0.59, 0.9], "specificity": [0.7, 0.9]},
        }
    }
    verdict = criterion(ci)
    assert verdict["sleepfm"]["success"] and not verdict["osf"]["success"]
