"""Fait tourner tout le pipeline (génération synthétique -> extraction
PhysioNet 2018 -> embeddings SleepFM -> sonde logistique -> évaluation) de
bout en bout, pour valider la plomberie avant l'arrivée des vraies données.
Marqué "slow" : exclure avec `pytest -m "not slow"` pour n'avoir que les
tests unitaires rapides.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_synthetic_physionet2018 import generate_subject
from src.data.epoch_dataset import PhysioNet2018EpochDataset
from src.data.extract_physionet2018 import main as extract_main
from src.models.effnet import EffNet
from src.models.embeddings import CHANNEL_INDICES, SleepFMEncoders
from src.models.linear_probe import evaluate_probe, train_linear_probe


def _generate_synthetic_raw(raw_dir: Path, n_subjects: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_subjects):
        generate_subject(
            str(raw_dir),
            f"synthetic-{i:04d}",
            sample_rate=32,
            n_epochs=10,
            epoch_sec=30,
            lead_in_sec=2,
            rng=rng,
        )


def _build_fake_checkpoint(path: Path, embedding_dim: int = 512) -> None:
    def build_encoder(in_ch):
        m = EffNet(in_channel=in_ch, stride=2, dilation=1)
        m.fc = torch.nn.Linear(m.fc.in_features, embedding_dim)
        return m

    torch.save(
        {
            "respiratory_state_dict": build_encoder(len(CHANNEL_INDICES["respiratory"])).state_dict(),
            "sleep_stages_state_dict": build_encoder(len(CHANNEL_INDICES["sleep_stages"])).state_dict(),
            "ekg_state_dict": build_encoder(len(CHANNEL_INDICES["ekg"])).state_dict(),
        },
        path,
    )


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path):
    raw_train, raw_test = tmp_path / "raw_train", tmp_path / "raw_test"
    processed_train, processed_test = tmp_path / "proc_train", tmp_path / "proc_test"
    _generate_synthetic_raw(raw_train, n_subjects=3, seed=1)
    _generate_synthetic_raw(raw_test, n_subjects=2, seed=2)

    sys_argv_backup = None
    import sys as _sys

    for raw_dir, processed_dir in [(raw_train, processed_train), (raw_test, processed_test)]:
        sys_argv_backup, _sys.argv = (
            _sys.argv,
            [
                "extract",
                "--data_path",
                str(raw_dir),
                "--save_path",
                str(processed_dir),
                "--num_threads",
                "1",
            ],
        )
        try:
            extract_main()
        finally:
            _sys.argv = sys_argv_backup

    assert len(list((processed_train / "X").rglob("*.npy"))) > 0
    assert len(list((processed_test / "X").rglob("*.npy"))) > 0

    ckpt_path = tmp_path / "best.pt"
    _build_fake_checkpoint(ckpt_path)
    encoders = SleepFMEncoders(ckpt_path, embedding_dim=512)

    def collect(processed_dir):
        loader = torch.utils.data.DataLoader(PhysioNet2018EpochDataset(processed_dir), batch_size=16)
        embeddings, labels = [], []
        with torch.no_grad():
            for epochs, batch_labels in loader:
                embeddings.append(encoders.embed_combined(epochs).numpy())
                labels.append(batch_labels.numpy())
        return np.concatenate(embeddings), np.concatenate(labels)

    X_train, y_train = collect(processed_train)
    X_test, y_test = collect(processed_test)

    probe = train_linear_probe(X_train, y_train)
    results = evaluate_probe(probe, X_test, y_test)

    assert 0.0 <= results["accuracy"] <= 1.0
    assert -1.0 <= results["cohen_kappa"] <= 1.0


@pytest.mark.slow
def test_chunked_embeddings_cache_and_resume(tmp_path):
    """Mode par lots de run_baseline.py : un .npz par sujet, mêmes époques
    que l'extraction classique, et une relance ne recalcule rien."""
    from run_baseline import embed_subjects_in_chunks, extract, load_cached_embeddings

    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(3)
    for i in range(3):  # 15 époques : "_10".."_14" se trient avant "_2", l'ordre doit être reconstitué
        generate_subject(
            str(raw_dir), f"synthetic-{i:04d}", sample_rate=32, n_epochs=15, epoch_sec=30, lead_in_sec=2, rng=rng
        )
    subjects = sorted(p.name for p in raw_dir.iterdir())

    ckpt_path = tmp_path / "best.pt"
    _build_fake_checkpoint(ckpt_path)
    encoders = SleepFMEncoders(ckpt_path, embedding_dim=512)

    fetched = []

    def fetch_subject(subject, dest_dir):
        fetched.append(subject)
        (dest_dir / subject).symlink_to(raw_dir / subject, target_is_directory=True)

    embeddings_dir = tmp_path / "embeddings"
    embed_subjects_in_chunks(
        subjects, fetch_subject, encoders, embeddings_dir, chunk_size=2, num_threads=1, batch_size=16, device="cpu"
    )
    assert sorted(fetched) == subjects
    assert sorted(p.stem for p in embeddings_dir.glob("*.npz")) == subjects
    # Début de chaque époque dans l'enregistrement d'origine : après la
    # période « undefined » de 2 s, toutes les 30 s.
    with np.load(embeddings_dir / f"{subjects[0]}.npz") as data:
        onsets = np.sort(data["onset_s"])
    assert onsets[0] == pytest.approx(2.0, abs=0.1)
    np.testing.assert_allclose(np.diff(onsets), 30.0, atol=0.1)

    # Même nombre d'époques et mêmes labels que l'extraction classique en un bloc
    processed = tmp_path / "processed"
    extract(raw_dir, processed, num_threads=1)
    reference = PhysioNet2018EpochDataset(processed)
    X, y, (group_subjects, epoch_idx, _) = load_cached_embeddings(embeddings_dir, subjects)
    assert X.shape == (len(reference), 3 * 512)
    reference_idx = [int(f.stem.rsplit("_", 1)[1]) for f, _ in reference.samples]
    assert list(epoch_idx) == reference_idx and list(group_subjects) == [f.parent.name for f, _ in reference.samples]
    assert reference_idx != sorted(reference_idx)  # le piège du tri par nom est bien présent

    # Cache antérieur à epoch_idx : même numérotation reconstruite
    for npz in embeddings_dir.glob("*.npz"):
        with np.load(npz) as data:
            legacy = {"X": data["X"], "y": data["y"]}
        np.savez(npz, **legacy)
    _, _, (_, legacy_idx, legacy_onsets) = load_cached_embeddings(embeddings_dir, subjects)
    assert np.isnan(legacy_onsets).all()  # ancien cache : pas de temps d'origine
    assert list(legacy_idx) == reference_idx
    assert X.dtype == np.float32
    assert list(y) == [reference[i][1] for i in range(len(reference))]

    # Reprise : rien n'est retéléchargé ni recalculé
    fetched.clear()
    embed_subjects_in_chunks(
        subjects, fetch_subject, encoders, embeddings_dir, chunk_size=2, num_threads=1, batch_size=16, device="cpu"
    )
    assert fetched == []

    # Interruption en pleine écriture : le .npz tronqué est recalculé, le
    # fichier temporaire abandonné est supprimé, les autres sujets restent en cache
    truncated = embeddings_dir / f"{subjects[0]}.npz"
    truncated.write_bytes(truncated.read_bytes()[:100])
    (embeddings_dir / f"{subjects[1]}.tmp.npz").write_bytes(b"partiel")
    embed_subjects_in_chunks(
        subjects, fetch_subject, encoders, embeddings_dir, chunk_size=2, num_threads=1, batch_size=16, device="cpu"
    )
    assert fetched == [subjects[0]]
    assert not list(embeddings_dir.glob("*.tmp.npz"))
    X_again, _, _ = load_cached_embeddings(embeddings_dir, subjects)
    np.testing.assert_allclose(X_again, X, atol=1e-5)


def test_results_json_records_subjects_settings_and_convergence(tmp_path):
    """Le JSON suffit à savoir si deux runs sont comparables : mêmes sujets
    de test, mêmes réglages, baseline convergée ou non."""
    import json
    from types import SimpleNamespace

    from run_baseline import save_results_json
    from src.models.linear_probe import compare_probes

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    X = rng.standard_normal((200, 8))
    X[np.arange(200), y] += 3.0
    results = compare_probes(["logistic"], X[:150], y[:150], X[150:], y[150:])
    args = SimpleNamespace(
        test_fraction=0.2, seed=42, probes=["logistic"], max_iter=1000, cv_folds=5, min_gain=0.02, max_gap=0.15
    )
    path = tmp_path / "run.json"
    groups = (np.repeat(["s3", "s4"], 25), np.tile(np.arange(25), 2))
    save_results_json(path, results, None, "desc", "ckpt", args, ["s2", "s1"], ["s3"], y[150:], groups)

    payload = json.loads(path.read_text())
    assert payload["train_subjects"] == ["s1", "s2"] and payload["test_subjects"] == ["s3"]
    assert payload["settings"]["max_gap"] == 0.15
    assert payload["probes"]["logistic"]["converged"] is True
    assert payload["probes"]["logistic"]["n_iter"] > 0
    # Prédictions du test époque par époque, à côté du JSON.
    with np.load(tmp_path / "run.predictions.npz") as preds:
        assert np.array_equal(preds["y_true"], y[150:])
        assert preds["proba_logistic"].shape == (50, 5) and len(preds["pred_logistic"]) == 50
        assert list(preds["subject"][:2]) == ["s3", "s3"] and preds["epoch_idx"][25] == 0


@pytest.mark.slow
def test_loser_of_mlp_vs_xgboost_is_never_evaluated_on_test():
    """La sélection MLP / XGBoost se fait par validation croisée sur le
    train ; le jeu de test ne voit que les sondes retenues."""
    from types import SimpleNamespace

    from run_baseline import select_and_evaluate

    rng = np.random.default_rng(0)

    def data(n_subjects):
        y = rng.integers(0, 5, n_subjects * 40)
        X = rng.standard_normal((len(y), 16)) * 0.5
        X[np.arange(len(y)), y] += 3.0
        groups = (np.repeat([f"s{i}" for i in range(n_subjects)], 40), np.tile(np.arange(40), n_subjects))
        return X, y, groups

    X_train, y_train, g_train = data(10)
    X_test, y_test, g_test = data(3)
    args = SimpleNamespace(
        probes=["logistic", "mlp", "xgboost"],
        cv_folds=5,
        min_gain=0.02,
        max_gap=0.15,
        max_iter=200,
        seed=42,
        device="cpu",
    )
    results, selection = select_and_evaluate(args, X_train, y_train, g_train, X_test, y_test, g_test)

    assert selection["chosen"] in ("mlp", "xgboost")
    loser = {"mlp", "xgboost"} - {selection["chosen"]}
    assert set(results) == {"logistic", selection["chosen"]} and loser.isdisjoint(results)


def test_frozen_split_keeps_roles_when_subjects_are_added(tmp_path):
    from run_baseline import load_or_extend_split

    path = tmp_path / "split.json"
    first = [f"s{i:03d}" for i in range(50)]
    train1, test1 = load_or_extend_split(path, first, 0.2, seed=42)
    assert len(test1) == 10 and not set(train1) & set(test1)
    # Ajout de 50 sujets : les 50 premiers gardent leur rôle.
    more = first + [f"s{i:03d}" for i in range(50, 100)]
    train2, test2 = load_or_extend_split(path, more, 0.2, seed=42)
    assert set(test1) <= set(test2) and set(train1) <= set(train2)
    assert len(test2) == 20 and len(train2) == 80
    # Relecture : identique, et un sujet indisponible est ignoré.
    assert load_or_extend_split(path, more, 0.2, seed=42) == (train2, test2)
    train3, test3 = load_or_extend_split(path, more[1:], 0.2, seed=42)
    assert more[0] not in train3 + test3


def test_embeddings_manifest_refuses_mixing_checkpoints_or_preprocessing(tmp_path):
    from run_baseline import check_embeddings_manifest

    expected = {"checkpoint_sha256": "aaa", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
    # Caches antérieurs au manifeste : déclarés en version 1, signalés.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    np.savez(legacy_dir / "s1.npz", X=np.zeros((1, 2)), y=np.zeros(1))
    manifest = check_embeddings_manifest(legacy_dir, expected, n_to_compute=0)
    assert manifest["preprocessing_version"] == 1 and "legacy" in manifest
    # Rien à calculer : la version différente est seulement signalée...
    check_embeddings_manifest(legacy_dir, expected, n_to_compute=0)
    # ... mais on refuse d'y ajouter des sujets calculés avec la version 2.
    with pytest.raises(SystemExit):
        check_embeddings_manifest(legacy_dir, expected, n_to_compute=3)
    # Autre checkpoint : toujours refusé.
    with pytest.raises(SystemExit):
        check_embeddings_manifest(legacy_dir, expected | {"checkpoint_sha256": "bbb"}, n_to_compute=0)
    # Dossier neuf : manifeste écrit tel quel, puis accepté.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert check_embeddings_manifest(fresh, expected, n_to_compute=5) == expected
    assert check_embeddings_manifest(fresh, expected, n_to_compute=5) == expected


def test_split_file_rejects_overlap_duplicates_and_keeps_test_fraction(tmp_path):
    import json

    from run_baseline import load_or_extend_split

    bad = tmp_path / "bad.json"
    for content in ({"train": ["s1"], "test": ["s1"]}, {"train": ["s1", "s1"], "test": ["s2"]}, {"train": []}):
        bad.write_text(json.dumps(content))
        with pytest.raises(SystemExit):
            load_or_extend_split(bad, ["s1", "s2"], 0.2, seed=42)
    with pytest.raises(SystemExit):
        load_or_extend_split(None, ["s1", "s2"], 1.5, seed=42)
    # Sujets ajoutés un par un : la part du test reste proche de 20 %.
    path = tmp_path / "split.json"
    load_or_extend_split(path, [f"s{i:02d}" for i in range(10)], 0.2, seed=42)
    for n in range(11, 31):
        train, test = load_or_extend_split(path, [f"s{i:02d}" for i in range(n)], 0.2, seed=42)
    assert len(test) == 6 and len(train) == 24
    assert not list(tmp_path.glob("*.tmp.json"))  # écriture atomique


def test_cache_validation_rejects_corrupt_contents(tmp_path):
    from run_baseline import cache_problem

    good = {"X": np.zeros((3, 1536), np.float32), "y": np.array([0, 1, 4]), "epoch_idx": np.arange(3)}
    np.savez(tmp_path / "ok.npz", **good, onset_s=np.array([0.0, 30.0, 60.0]))
    assert cache_problem(tmp_path / "ok.npz") is None
    corrupt = {
        "dim": good | {"X": np.zeros((3, 8), np.float32)},
        "nan": good | {"X": np.full((3, 1536), np.nan, np.float32)},
        "label": good | {"y": np.array([0, 99, 1])},
        "dup": good | {"epoch_idx": np.array([0, 0, 1])},
        "onset": good | {"onset_s": np.array([0.0, 60.0, 30.0])},
    }
    for name, arrays in corrupt.items():
        np.savez(tmp_path / f"{name}.npz", **arrays)
        assert cache_problem(tmp_path / f"{name}.npz") is not None, name


def test_extraction_is_redone_when_labels_file_is_empty(tmp_path):
    from src.data.extract_physionet2018 import is_extracted

    x_dir, y_path, t_path = tmp_path / "X" / "s1", tmp_path / "Y" / "s1.pickle", tmp_path / "T" / "s1.json"
    y_path.parent.mkdir()
    y_path.write_bytes(b"")  # pickle vide : extraction interrompue
    assert not is_extracted(str(x_dir), str(y_path), str(t_path))


def test_embed_only_computes_embeddings_without_training_probes(tmp_path, monkeypatch, capsys):
    """--embed-only : le GPU ne sert qu'aux embeddings, aucune sonde n'est
    entraînée et aucun résultat n'est écrit."""
    import run_baseline

    embeddings_dir, results = tmp_path / "embeddings", tmp_path / "run.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--embeddings-dir",
            str(embeddings_dir),
            "--embed-only",
            "--num-synthetic-subjects",
            "3",
            "--num-threads",
            "1",
            "--device",
            "cpu",
            "--results-json",
            str(results),
        ],
    )
    monkeypatch.setattr(run_baseline, "select_and_evaluate", lambda *a, **k: pytest.fail("sondes entraînées"))
    run_baseline.main()
    assert len(list(embeddings_dir.glob("*.npz"))) == 3 and (embeddings_dir / "manifest.json").exists()
    assert not results.exists()
    assert "Embeddings seulement" in capsys.readouterr().out
