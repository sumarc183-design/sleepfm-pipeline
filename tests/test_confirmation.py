"""Le passage de confirmation n'entraîne rien et refuse tout écart au protocole."""

import json

import numpy as np
import pytest

from scripts import confirm_frozen_models as confirm
from src.data.cache_contract import file_sha256
from src.models.linear_probe import train_probe
from src.models.probe_export import save_probes
from src.models.sequence_probe import SequenceProbe

CONTRACT = {"checkpoint_sha256": "aaa", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
RESERVED = ["tr09-0001", "tr09-0002", "tr10-0001", "tr10-0002"]
REJECTED = "tr11-0001"


def _night(seed: int, n: int = 40):
    rng = np.random.default_rng(seed)
    y = np.resize(np.arange(5), n)
    X = rng.normal(size=(n, 1536)).astype(np.float32)
    X[np.arange(n), y] += 3
    return X, y


def _write_cache(folder, subject, seed):
    X, y = _night(seed)
    np.savez(folder / f"{subject}.npz", X=X, y=y, epoch_idx=np.arange(len(y)), onset_s=30.0 * np.arange(len(y)))


@pytest.fixture
def setup(tmp_path):
    X = np.concatenate([_night(i)[0] for i in range(4)])
    y = np.concatenate([_night(i)[1] for i in range(4)])
    subjects = np.repeat([f"tr03-{i}" for i in range(4)], 40)
    groups = (subjects, np.tile(np.arange(40), 4))
    logistic = train_probe("logistic", X, y, seed=0, max_iter=50)
    sequence = SequenceProbe(epochs=1, hidden_dim=8, window=20, seed=0).fit(X, y, groups)
    models = tmp_path / "models"
    save_probes(models, {"logistic": logistic, "sequence": sequence}, 1536, CONTRACT)

    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "manifest.json").write_text(json.dumps(CONTRACT))
    for i, subject in enumerate(RESERVED):
        _write_cache(embeddings, subject, 10 + i)
    (embeddings / f"{REJECTED}.skipped").write_text("canaux manquants\n")

    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"subjects": [*RESERVED, REJECTED]}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["tr03-0", "tr03-1"], "test": ["tr03-2", "tr03-3"]}))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol_version": 2,
                "models": {k: file_sha256(models / f"{k}.joblib") for k in ("logistic", "sequence")},
                "reserved_sha256": file_sha256(reserved),
                "split_sha256": file_sha256(split),
                "contract": CONTRACT,
                "bootstrap": {"n_boot": 20, "seed": 42},
            }
        )
    )
    return {
        "config_path": config,
        "models_dir": models,
        "embeddings_dir": embeddings,
        "reserved_file": reserved,
        "split_file": split,
        "out": tmp_path / "sorties" / "confirmation",
        "require_clean": False,
    }


def test_confirmation_reports_every_subject_without_training(setup, monkeypatch):
    def no_training(*args, **kwargs):
        raise AssertionError("aucun entraînement n'est permis")

    monkeypatch.setattr("src.models.linear_probe.train_probe", no_training)
    monkeypatch.setattr(SequenceProbe, "fit", no_training)

    report = confirm.run_confirmation(**setup)

    assert report["status"] == "terminé" and report["n_requested"] == 5 and report["n_evaluated"] == 4
    assert report["rejected"] == {REJECTED: "canaux manquants"}
    written = json.loads((setup["out"].parent / "confirmation.json").read_text())
    assert written["status"] == "terminé"
    assert "gain_confirmed" in written["primary"] and "tr09" in written["secondary"]["kappa_by_prefix"]
    saved = np.load(setup["out"].parent / "confirmation.predictions.npz")
    assert set(saved["subject"]) == set(RESERVED) and saved["proba_sequence"].shape[1] == 5

    with pytest.raises(SystemExit, match="un seul passage"):
        confirm.run_confirmation(**setup)


@pytest.mark.parametrize(
    "alter, message",
    [
        (lambda s: (s["models_dir"] / "sequence.joblib").write_bytes(b"autre"), "modèle sequence"),
        (lambda s: s["reserved_file"].write_text(json.dumps({"subjects": RESERVED})), "liste"),
        (lambda s: s["split_file"].write_text(json.dumps({"train": ["x"], "test": ["y"]})), "découpage"),
        (lambda s: (s["embeddings_dir"] / f"{RESERVED[0]}.npz").unlink(), "sans résultat"),
        (lambda s: (s["embeddings_dir"] / "tr03-0.npz").write_bytes(b""), "étranger"),
        (lambda s: (s["embeddings_dir"] / f"{RESERVED[0]}.skipped").write_text("x"), "à la fois"),
        (lambda s: (s["embeddings_dir"] / f"{RESERVED[0]}.npz").write_bytes(b"abime"), "illisible"),
        (
            lambda s: (s["embeddings_dir"] / "manifest.json").write_text(json.dumps(CONTRACT | {"sample_rate": 128})),
            "contrat",
        ),
    ],
    ids=["modele", "liste", "decoupage", "manquant", "etranger", "double", "cache_abime", "contrat"],
)
def test_any_gap_with_the_protocol_stops_before_any_score(setup, alter, message):
    alter(setup)
    with pytest.raises(SystemExit, match=message):
        confirm.run_confirmation(**setup)
    assert not list(setup["out"].parent.glob("*")) if setup["out"].parent.exists() else True


def test_reserved_subjects_inside_the_split_are_refused(setup):
    setup["split_file"].write_text(json.dumps({"train": [RESERVED[0]], "test": ["tr03-3"]}))
    config = json.loads(setup["config_path"].read_text())
    config["split_sha256"] = file_sha256(setup["split_file"])
    setup["config_path"].write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="découpage"):
        confirm.run_confirmation(**setup)


def test_rehearsal_refuses_reserved_subjects_and_is_labelled(setup):
    with pytest.raises(SystemExit, match="aucun sujet réservé"):
        confirm.run_confirmation(**setup, rehearsal=[RESERVED[0]])
    with pytest.raises(SystemExit, match="déjà ouverts"):
        confirm.run_confirmation(**setup, rehearsal=["tr99-0000"])

    for i, subject in enumerate(["tr03-2", "tr03-3"]):
        _write_cache(setup["embeddings_dir"], subject, 20 + i)
    report = confirm.run_confirmation(**setup, rehearsal=["tr03-2", "tr03-3"])
    assert report["kind"].startswith("répétition") and report["n_evaluated"] == 2
