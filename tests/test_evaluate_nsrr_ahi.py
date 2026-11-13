import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate_nsrr_ahi as ev
from scripts.embed_nsrr import expected_contract, save_night
from scripts.estimate_respiratory_ahi import MODELS
from tests.test_nsrr_annotations import document, scored

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/respiratory_ahi_physionet2018.json"
BASELINE = ROOT / "results/respiratory_baseline_physionet2018.json"
COMPARISON = ROOT / "results/respiratory_osf_vs_sleepfm_physionet2018.json"
needs_results = pytest.mark.skipif(not FROZEN.exists(), reason="résultats PhysioNet absents")
N_EPOCHS = 260
SUBJECTS = [f"mesa-sleep-{i:04d}" for i in range(13)]  # 0012 : rejeté à l'extraction


def night_xml(n_events: int, sleep_s: int = 7200, extra: str = "") -> bytes:
    events = [scored("Stages|Stages", "Wake|0", 0, 600), scored("Stages|Stages", "Stage 2 sleep|2", 600, sleep_s)]
    events += [scored("Respiratory|Respiratory", "Hypopnea|Hypopnea", 700 + 60 * i, 20) for i in range(n_events)]
    return document(*events, extra)


@pytest.fixture(scope="module")
def frozen():
    return ev.load_frozen(FROZEN, BASELINE, COMPARISON)


@pytest.fixture
def setup(tmp_path, frozen, monkeypatch):
    """Extraction et annotations synthétiques conformes ; renvoie les arguments du passage."""
    monkeypatch.setattr(ev, "uncommitted", lambda paths: [])
    emb, ann = tmp_path / "emb", tmp_path / "ann"
    emb.mkdir(), ann.mkdir()
    subjects_path = tmp_path / "list.json"
    subjects_path.write_text(json.dumps({"cohort": "mesa", "subjects": SUBJECTS}))
    list_sha = hashlib.sha256(subjects_path.read_bytes()).hexdigest()
    edf = tmp_path / "fake.edf"
    edf.write_bytes(b" " * 176 + b"22.00.00" + b" " * 72)  # heure de début dans l'en-tête
    rng = np.random.default_rng(0)
    for i, subject in enumerate(SUBJECTS[:-1]):
        result = {
            "X_sleepfm": rng.normal(size=(N_EPOCHS, 1536)).astype(np.float32),
            "X_osf": rng.normal(size=(N_EPOCHS, 768)).astype(np.float32),
            "epoch_idx": np.arange(N_EPOCHS),
            "onset_s": 30.0 * np.arange(N_EPOCHS),
            "info": {"n_epochs": N_EPOCHS},
        }
        save_night(emb, subject, result, edf)
        (ann / f"{subject}-nsrr.xml").write_bytes(night_xml(n_events=5 + 10 * i))
    (ann / "mesa-sleep-0001-nsrr.xml").write_bytes(night_xml(n_events=3, sleep_s=1800))  # < 1 h : exclue
    (emb / "mesa-sleep-0012.skipped").write_text("EDF tronqué\n")
    contract = expected_contract(
        "mesa", list_sha, MODELS, frozen["checkpoints"]["sleepfm"], frozen["checkpoints"]["osf"], "physionet_median_v1"
    )
    (emb / "manifest.json").write_text(json.dumps(contract))
    status = {"done": SUBJECTS[:-1], "rejected": {"mesa-sleep-0012": "EDF tronqué"}, "missing": [], "complete": True}
    (emb / "run_status.json").write_text(json.dumps(status))
    official = tmp_path / "official.csv"
    lines = [f"{i},{3 * i},{2 * i}" for i in range(12)] + ["5000,1,1"]
    lines[3] = "3,,6"  # IAH 3 % manquant pour ce sujet
    official.write_text("mesaid,ahi_a0h3,ahi_a0h4\n" + "\n".join(lines) + "\n")
    return Namespace(
        cohort="mesa",
        embeddings=emb,
        annotations_dir=ann,
        xml_pattern="{subject}-nsrr.xml",
        out=tmp_path / "out" / "nsrr_ahi_mesa",
        frozen_ahi=FROZEN,
        baseline=BASELINE,
        comparison=COMPARISON,
        plan=ev.PLAN,
        official_csv=official,
        expected_sha=ev.FROZEN_AHI_SHA256,
        approved_subjects=(subjects_path, list_sha),
        marker_dir=tmp_path / "markers",
    )


def edit_json(path: Path, **changes) -> None:
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def refused_before_targets(args, monkeypatch, match: str) -> None:
    """Refus au précontrôle : aucune prédiction, aucun marqueur de passage."""

    def forbidden(*a, **k):
        raise AssertionError("prédiction ou lecture des cibles avant la fin du précontrôle")

    monkeypatch.setattr(ev, "evaluate", forbidden)
    monkeypatch.setattr(ev, "night_row", forbidden)
    with pytest.raises(ValueError, match=match):
        ev.run(args)
    assert not args.marker_dir.exists() or not any(args.marker_dir.iterdir())


@needs_results
def test_frozen_rule_is_checked(frozen):
    assert frozen["thresholds"]["sleepfm"] == pytest.approx(0.6934, abs=1e-4)
    with pytest.raises(ValueError):
        ev.load_frozen(FROZEN, BASELINE, COMPARISON, expected_sha="0" * 64)


@needs_results
def test_run_end_to_end_and_single_pass(setup):
    report = ev.run(setup)
    assert report["nights"]["evaluated"] == 11
    assert set(report["nights"]["excluded"]) == {"mesa-sleep-0001", "mesa-sleep-0012"}
    assert set(report["success_criterion"]["models"]) == {"sleepfm", "osf"}
    official = report["official_reference"]
    assert official["id_column"] == "mesaid"
    assert official["ahi_a0h3"]["nights"] == 10 and official["ahi_a0h3"]["missing"] == ["mesa-sleep-0003"]
    assert official["ahi_a0h4"]["nights"] == 11
    for column in ("ahi_a0h3", "ahi_a0h4"):
        result = official[column]["results"]["sleepfm"]
        assert {"auc_rate", "bland_altman", "weighted_kappa_linear"} <= set(result)
        assert "ci95" in official[column]["bootstrap"]
    marker = json.loads(next(setup.marker_dir.iterdir()).read_text())
    assert marker["state"] == "completed"
    setup.out = setup.out.with_name("autre_prefixe")
    with pytest.raises(ValueError, match="déjà lancée"):
        ev.run(setup)  # un autre préfixe ne contourne pas le passage unique


@needs_results
def test_failed_pass_cannot_be_rerun(setup, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("panne pendant l'évaluation")

    monkeypatch.setattr(ev, "evaluate", broken)
    with pytest.raises(RuntimeError):
        ev.run(setup)
    marker = json.loads(next(setup.marker_dir.iterdir()).read_text())
    assert marker["state"] == "failed" and "panne" in marker["error"]
    monkeypatch.undo()
    monkeypatch.setattr(ev, "uncommitted", lambda paths: [])
    with pytest.raises(ValueError, match="déjà lancée"):
        ev.run(setup)


EXPECTED_REFUSAL = {
    "wrong_list": "approuvée",
    "truncated": "Couverture",
    "foreign": "Couverture",
    "duplicate": "double",
    "empty_rejection": "sans motif",
    "version": "contrat",
    "sha_field": "contrat",
}


@needs_results
@pytest.mark.parametrize("case", sorted(EXPECTED_REFUSAL))
def test_population_and_contract_refused(setup, monkeypatch, case):
    emb, status = setup.embeddings, setup.embeddings / "run_status.json"
    done = SUBJECTS[:-1]
    if case == "wrong_list":
        setup.approved_subjects = (setup.approved_subjects[0], "0" * 64)
    elif case == "truncated":
        edit_json(status, done=done[:-1])
        (emb / f"{done[-1]}.npz").unlink()
    elif case == "foreign":
        edit_json(status, rejected={"mesa-sleep-0012": "EDF tronqué", "mesa-sleep-9999": "x"})
    elif case == "duplicate":
        edit_json(status, done=done + [done[0]])
    elif case == "empty_rejection":
        edit_json(status, rejected={"mesa-sleep-0012": " "})
    elif case == "version":
        edit_json(emb / "manifest.json", preparation_version=999)
    elif case == "sha_field":
        edit_json(emb / "manifest.json", subjects_sha256="not-the-frozen-list")
    refused_before_targets(setup, monkeypatch, EXPECTED_REFUSAL[case])


@needs_results
def test_inconsistent_or_modified_cache_refused(setup, monkeypatch):
    npz = setup.embeddings / "mesa-sleep-0002.npz"
    np.savez(npz, X_sleepfm=np.zeros((4, 1536)), X_osf=np.zeros((4, 768)), epoch_idx=[1, 1], onset_s=[60.0, 0.0])
    refused_before_targets(setup, monkeypatch, "modifié")


def test_cache_arrays_validated():
    from src.data.nsrr_cache import validate_night_arrays

    bad = {"X_sleepfm": np.zeros((4, 1536)), "X_osf": np.zeros((4, 768)), "epoch_idx": [1, 1], "onset_s": [60, 0]}
    with pytest.raises(ValueError):
        validate_night_arrays({k: np.asarray(v) for k, v in bad.items()}, MODELS)
    good = {"X_sleepfm": np.zeros((2, 1536)), "X_osf": np.zeros((2, 768)), "epoch_idx": np.arange(2)}
    assert validate_night_arrays(good | {"onset_s": np.array([0.0, 30.0])}, MODELS) == 2
    with pytest.raises(ValueError, match="époques"):
        validate_night_arrays(good | {"onset_s": np.array([0.0, 30.0])}, MODELS, n_epochs=3)


@needs_results
def test_missing_plan_or_dirty_code_refused(setup, monkeypatch):
    setup.plan = setup.embeddings / "absent.md"
    refused_before_targets(setup, monkeypatch, "Plan absent")
    setup.plan = ev.PLAN
    monkeypatch.setattr(ev, "uncommitted", lambda paths: ["scripts/evaluate_nsrr_ahi.py"])
    refused_before_targets(setup, monkeypatch, "non commités")


@needs_results
@pytest.mark.parametrize(
    ("csv_text", "match"),
    [
        ("nsrrid,ahi_a0h3,ahi_a0h4\n1,2,3\n", "colonnes"),
        ("mesaid,ahi_a0h3,ahi_a0h4\n1,-5,3\n", "invalide"),
        ("mesaid,ahi_a0h3,ahi_a0h4\n1.5,2,3\n", "non entier"),
        ("mesaid,ahi_a0h3,ahi_a0h4\n1,2,3\n1,2,3\n", "double"),
        ("mesaid,ahi_a0h3,ahi_a0h4\n7777,2,3\n", "aucun sujet"),
    ],
)
def test_official_csv_checked_before_pass(setup, monkeypatch, csv_text, match):
    setup.official_csv.write_text(csv_text)
    refused_before_targets(setup, monkeypatch, match)


@needs_results
def test_unknown_concept_refused_and_truncated_xml_excluded(setup, monkeypatch):
    ann = setup.annotations_dir
    good = (ann / "mesa-sleep-0004-nsrr.xml").read_bytes()
    unknown = scored("Respiratory|Respiratory", "Unrecognized apnea|variant", 900, 20)
    (ann / "mesa-sleep-0004-nsrr.xml").write_bytes(night_xml(5, extra=unknown))
    refused_before_targets(setup, monkeypatch, "concepts inconnus")
    monkeypatch.undo()
    monkeypatch.setattr(ev, "uncommitted", lambda paths: [])
    (ann / "mesa-sleep-0004-nsrr.xml").write_bytes(good[:-30])  # XML tronqué : exclusion motivée
    (ann / "mesa-sleep-0005-nsrr.xml").write_bytes(night_xml(5, sleep_s=9000))  # au-delà du signal
    report = ev.run(setup)
    assert report["nights"]["excluded"]["mesa-sleep-0004"].startswith("annotations : XML illisible")
    assert "pour 7800 s de signal" in report["nights"]["excluded"]["mesa-sleep-0005"]


def test_approved_lists_pass_metadata_checks():
    from scripts.embed_nsrr import subject_list

    for cohort, (path, sha) in ev.APPROVED_SUBJECTS.items():
        subjects, digest = subject_list(path, cohort)
        assert digest == sha and len(subjects) == {"mesa": 2056, "shhs1": 5793, "ucddb": 25}[cohort]
        if cohort in ev.OFFICIAL_ID_COLUMN:
            assert all(ev.subject_number(s) > 0 for s in subjects)


def test_criterion():
    ci = {
        "ci95": {
            "sleepfm": {"sensitivity": [0.61, 0.9], "specificity": [0.7, 0.9]},
            "osf": {"sensitivity": [0.59, 0.9], "specificity": [0.7, 0.9]},
        }
    }
    verdict = ev.criterion(ci)
    assert verdict["sleepfm"]["success"] and not verdict["osf"]["success"]


def test_subject_number_matches_nsrr_ids():
    assert ev.subject_number("mesa-sleep-0001") == 1 and ev.subject_number("shhs1-200001") == 200001
    with pytest.raises(ValueError):
        ev.subject_number("shhs2-200001")


@needs_results
def test_ucddb_descriptive_run(setup, monkeypatch):
    """UCDDB : annotations texte recalées sur l'heure de début de l'EDF, pas de verdict."""
    emb, ann = setup.embeddings, setup.annotations_dir
    subjects = [f"ucddb{i:03d}" for i in range(13)]
    for old, new in zip(SUBJECTS[:-1], subjects[:-1], strict=True):
        for ext in (".npz", ".json"):
            (emb / f"{old}{ext}").rename(emb / f"{new}{ext}")
        record = json.loads((emb / f"{new}.json").read_text())
        (emb / f"{new}.json").write_text(json.dumps(record | {"subject": new}))
    (emb / "mesa-sleep-0012.skipped").rename(emb / "ucddb012.skipped")
    for i, subject in enumerate(subjects[:-1]):
        (ann / f"{subject}_stage.txt").write_text("\n".join(["0"] * 20 + ["3"] * 240) + "\n")
        events = "".join(f"22:{12 + i // 2:02d}:{(i % 2) * 30:02d}  HYP-O   20  90  3\n" for i in range(5 + 6 * i))
        (ann / f"{subject}_respevt.txt").write_text(" Time Type\n" + events)
    subjects_path = setup.approved_subjects[0]
    subjects_path.write_text(json.dumps({"cohort": "ucddb", "subjects": subjects}))
    list_sha = hashlib.sha256(subjects_path.read_bytes()).hexdigest()
    manifest = json.loads((emb / "manifest.json").read_text()) | {"cohort": "ucddb", "subjects_sha256": list_sha}
    (emb / "manifest.json").write_text(json.dumps(manifest))
    status = {"done": subjects[:-1], "rejected": {"ucddb012": "EDF tronqué"}, "missing": [], "complete": True}
    (emb / "run_status.json").write_text(json.dumps(status))
    setup.cohort, setup.official_csv, setup.approved_subjects = "ucddb", None, (subjects_path, list_sha)
    report = ev.run(setup)
    assert report["success_criterion"]["applies"] is False
    assert report["nights"]["evaluated"] == 12 and report["results"]["sleepfm"]["confusion_15"]
