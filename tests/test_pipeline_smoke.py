"""Fait tourner tout le pipeline (génération synthétique -> extraction
PhysioNet 2018 -> embeddings SleepFM -> sonde logistique -> évaluation) de
bout en bout, pour valider la plomberie avant l'arrivée des vraies données.
Marqué "slow" : exclure avec `pytest -m "not slow"` pour n'avoir que les
tests unitaires rapides.
"""

import json
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
@pytest.mark.xgboost
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
            "--num-workers",
            "0",
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


def test_exported_probes_reload_and_predict_identically(tmp_path):
    """--save-models : les sondes rechargées prédisent comme avant l'export,
    et des embeddings d'un autre checkpoint sont refusés."""
    from src.models.linear_probe import compare_probes, predict_probabilities
    from src.models.probe_export import load_probes, save_probes

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 160)
    X = rng.standard_normal((160, 8)).astype(np.float32)
    X[np.arange(160), y] += 3.0
    groups = (np.repeat([f"s{i}" for i in range(8)], 20), np.tile(np.arange(20), 8))
    tr, te = slice(0, 120), slice(120, 160)
    results = compare_probes(
        ["logistic", "mlp", "sequence"],  # mlp : a aussi un attribut model_ (régression)
        X[tr],
        y[tr],
        X[te],
        y[te],
        train_groups=(groups[0][tr], groups[1][tr]),
        test_groups=(groups[0][te], groups[1][te]),
    )
    manifest = {"checkpoint_sha256": "aaa", "preprocessing_version": 2, "channels": ["C3"], "sample_rate": 256}
    save_probes(tmp_path, {k: r["model"] for k, r in results.items()}, 8, manifest)
    models, exported = load_probes(tmp_path, manifest)
    assert exported["probes"]["sequence"]["needs_groups"] and exported["input"]["embedding_dim"] == 8
    test_groups = (groups[0][te], groups[1][te])
    for kind, r in results.items():
        np.testing.assert_allclose(predict_probabilities(models[kind], X[te], test_groups), r["test_outputs"]["proba"])
    with pytest.raises(ValueError, match="incompatibles"):
        load_probes(tmp_path, manifest | {"checkpoint_sha256": "bbb"})


def _cache(path, n=4, **overrides):
    arrays = {
        "X": np.zeros((n, 1536), dtype=np.float32),
        "y": np.zeros(n, dtype=np.int64),
        "epoch_idx": np.arange(n),
        "onset_s": 30.0 * np.arange(n),
    }
    arrays.update(overrides)
    np.savez(path, **{key: value for key, value in arrays.items() if value is not None})


@pytest.mark.parametrize(
    "overrides",
    [
        {"onset_s": np.array([0, 30, np.nan, 1800.0])},
        {"onset_s": np.array([0, 30, np.inf, 1800.0])},
        {"epoch_idx": np.array([0.0, 1.5, 2.0, 3.0])},
        {"epoch_idx": np.array([-1, 0, 1, 2])},
        {"y": np.zeros((4, 1), dtype=np.int64)},
        {"X": np.zeros((4, 1536, 1), dtype=np.float32)},
        {"epoch_idx": None},  # onset_s sans epoch_idx
    ],
)
def test_cache_validation_rejects_inconsistent_shapes_and_times(tmp_path, overrides):
    from run_baseline import cache_problem

    path = tmp_path / "s.npz"
    _cache(path, **overrides)
    assert cache_problem(path) is not None


def test_legacy_cache_without_times_stays_valid_and_labels_are_optional_to_predict(tmp_path):
    from run_baseline import cache_problem, load_cached_embeddings

    legacy = tmp_path / "legacy.npz"
    _cache(legacy, onset_s=np.full(4, np.nan))
    assert cache_problem(legacy) is None

    unlabeled = tmp_path / "unlabeled.npz"
    _cache(unlabeled, y=None)
    assert "stades" in cache_problem(unlabeled)
    assert cache_problem(unlabeled, require_labels=False) is None
    X, y, groups = load_cached_embeddings(tmp_path, ["unlabeled"], require_labels=False)
    assert y is None and len(X) == 4 and len(groups[1]) == 4


def test_split_file_requires_lists(tmp_path):
    from run_baseline import load_or_extend_split

    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": "abc", "test": ["d"]}))
    with pytest.raises(SystemExit, match="liste"):
        load_or_extend_split(path, ["a", "b"], 0.5, 0)


CONTRACT = {"checkpoint_sha256": "aaa", "preprocessing_version": 2, "channels": ["C3", "C4"], "sample_rate": 256}


def _source(directory, manifest, subject="s1", content=b"contenu"):
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / f"{subject}.npz").write_bytes(content)
    return directory / "manifest.json"


def test_import_rejects_other_contract_before_copy_and_flags_conflicts(tmp_path):
    from src.data.cache_contract import import_embeddings

    good = _source(tmp_path / "good", CONTRACT, "s1")
    good2 = _source(tmp_path / "good2", CONTRACT, "s2")
    bad_checkpoint = _source(tmp_path / "bad1", CONTRACT | {"checkpoint_sha256": "bbb"}, "s3")
    bad_channels = _source(tmp_path / "bad2", CONTRACT | {"channels": ["C4", "C3"]}, "s4")
    bad_rate = _source(tmp_path / "bad3", CONTRACT | {"sample_rate": 128}, "s5")
    incomplete = _source(tmp_path / "bad4", {"preprocessing_version": 2}, "s6")
    destination = tmp_path / "dest"

    report = import_embeddings([good, good2, bad_checkpoint, bad_channels, bad_rate, incomplete], destination, CONTRACT)

    assert report["copied"] == 2 and len(report["ignored"]) == 4
    assert sorted(p.name for p in destination.glob("*.npz")) == ["s1.npz", "s2.npz"]
    assert json.loads((destination / "manifest.json").read_text()) == CONTRACT
    assert "s1.npz" in json.loads((destination / "provenance.json").read_text())

    assert import_embeddings([good], destination, CONTRACT)["copied"] == 0  # déjà là, identique
    conflict = _source(tmp_path / "conflict", CONTRACT, "s1", b"autre contenu")
    with pytest.raises(ValueError, match="Conflit"):
        import_embeddings([conflict], destination, CONTRACT)
    with pytest.raises(ValueError, match="contrat"):
        import_embeddings([good], destination, CONTRACT | {"checkpoint_sha256": "bbb"})


def test_loading_probes_requires_a_complete_and_identical_contract(tmp_path):
    from src.models.linear_probe import train_probe
    from src.models.probe_export import load_probes, save_probes

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 100)
    X = rng.standard_normal((100, 8))
    X[np.arange(100), y] += 3.0
    groups = (np.repeat([f"s{i}" for i in range(5)], 20), np.tile(np.arange(20), 5))
    models = {"logistic": train_probe("logistic", X, y), "mlp": train_probe("mlp", X, y, groups=groups)}
    save_probes(tmp_path, models, 8, CONTRACT)

    models_back, export = load_probes(tmp_path, CONTRACT)
    assert set(models_back) == {"logistic", "mlp"}
    mlp = export["probes"]["mlp"]
    assert mlp["hyperparameters"]["hidden_layer_sizes"] == "(256,)" and mlp["hyperparameters"]["patience"] == 10
    assert mlp["fit"]["validation_subjects"] and mlp["fit"]["best_pass"] >= 1
    assert export["environment"]["code_commit"]

    for bad in (
        None,
        {"checkpoint_sha256": "aaa"},
        CONTRACT | {"channels": ["C4", "C3"]},
        CONTRACT | {"sample_rate": 128},
        CONTRACT | {"checkpoint_sha256": "bbb"},
    ):
        with pytest.raises(ValueError, match="incompatibles"):
            load_probes(tmp_path, bad)
    assert set(load_probes(tmp_path, None, allow_unverified=True)[0]) == {"logistic", "mlp"}


def test_interrupted_download_is_not_taken_for_a_complete_file(tmp_path, monkeypatch):
    import io

    from scripts import download_physionet2018 as download

    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise KeyboardInterrupt

    def interrupted(url, timeout):
        return Interrupted(b"")

    monkeypatch.setattr(download.urllib.request, "urlopen", interrupted)
    with pytest.raises(KeyboardInterrupt):
        download.download_subject("tr00-0001", tmp_path)
    folder = tmp_path / "tr00-0001"
    assert not (folder / "tr00-0001.hea").exists()

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"complet"))
    download.download_subject("tr00-0001", tmp_path)
    assert all((folder / name).read_bytes() == b"complet" for name in ("tr00-0001.hea", "tr00-0001.mat"))
    assert not list(folder.glob("*.part"))


def test_band_shares_cover_the_whole_range_including_3_hz():
    from scripts.eeg_spectra_by_group import relative_band_power

    sr = 256
    t = np.arange(30 * sr) / sr
    shares = relative_band_power(np.sin(2 * np.pi * 3 * t)[None].copy(), sr)
    assert shares["delta"][0] > 0.99 and shares["lent"][0] > 0.99
    noise = np.random.default_rng(0).standard_normal((2, 30 * sr))
    shares = relative_band_power(noise, sr)
    total = sum(shares[band] for band in ("delta", "thêta", "alpha", "sigma", "bêta"))
    np.testing.assert_allclose(total, 1.0)


def _npz_source(folder, manifest, subject, content=b"cache"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / f"{subject}.npz").write_bytes(content)
    return folder / "manifest.json"


def test_import_refuses_a_destination_with_files_of_unknown_provenance(tmp_path):
    from src.data.cache_contract import import_embeddings

    destination = tmp_path / "dest"
    destination.mkdir()
    (destination / "inconnu.npz").write_bytes(b"origine inconnue")
    with pytest.raises(ValueError, match="sans manifeste"):
        import_embeddings([_npz_source(tmp_path / "ok", CONTRACT, "s1")], destination, CONTRACT)
    assert not (destination / "manifest.json").exists() and not (destination / "s1.npz").exists()


def test_import_checks_every_conflict_before_copying_anything(tmp_path):
    from src.data.cache_contract import file_sha256, import_embeddings

    first = _npz_source(tmp_path / "a", CONTRACT, "s1")
    (first.parent / "s2.npz").write_bytes(b"version a")
    second = _npz_source(tmp_path / "b", CONTRACT, "s2", b"version b")
    destination = tmp_path / "dest"
    with pytest.raises(ValueError, match="Conflit"):
        import_embeddings([first, second], destination, CONTRACT)
    assert not list(destination.iterdir())

    report = import_embeddings([first], destination, CONTRACT)
    assert report["copied"] == 2 and not (destination / ".import_en_cours").exists()
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["s1.npz"]["sha256"] == file_sha256(destination / "s1.npz")


def test_import_interrupted_before_publication_leaves_only_verified_files(tmp_path, monkeypatch):
    from src.data import cache_contract

    source = _npz_source(tmp_path / "a", CONTRACT, "s1")
    (source.parent / "s2.npz").write_bytes(b"deux")
    destination = tmp_path / "dest"
    real_replace, calls = cache_contract.os.replace, []

    def stop_after_first_publication(src, dst):
        real_replace(src, dst)
        if str(dst).endswith(".npz"):
            calls.append(dst)
            raise KeyboardInterrupt

    monkeypatch.setattr(cache_contract.os, "replace", stop_after_first_publication)
    with pytest.raises(KeyboardInterrupt):
        cache_contract.import_embeddings([source], destination, CONTRACT)
    monkeypatch.setattr(cache_contract.os, "replace", real_replace)
    assert json.loads((destination / "manifest.json").read_text()) == CONTRACT
    assert len(list(destination.glob("*.npz"))) == 1
    assert cache_contract.import_embeddings([source], destination, CONTRACT)["copied"] == 1
    assert len(list(destination.glob("*.npz"))) == 2


@pytest.mark.parametrize(
    "onsets",
    [np.full(4, np.nan), np.array([-30.0, 0.0, 30.0, 60.0]), np.array([0.0, 1.0, 2.0, 3.0])],
    ids=["tous_manquants", "negatif", "chevauchement"],
)
def test_strict_times_reject_unusable_chronology(tmp_path, onsets):
    from run_baseline import cache_problem

    path = tmp_path / "c.npz"
    base = {"X": np.zeros((4, 1536)), "y": np.zeros(4, dtype=int), "epoch_idx": np.arange(4)}
    np.savez(path, **base, onset_s=onsets)
    assert cache_problem(path, strict_times=True) is not None

    np.savez(path, **base, onset_s=np.array([0.0, 30.0, 90.0, 120.0]))  # un trou reste permis
    assert cache_problem(path, strict_times=True) is None
    np.savez(path, **base)
    assert cache_problem(path) is None and cache_problem(path, strict_times=True) is not None


def test_truncated_download_is_retried_and_never_kept(tmp_path, monkeypatch):
    import io

    from scripts import download_physionet2018 as download

    class Response(io.BytesIO):
        headers = {"Content-Length": "1000"}

    answers = []

    def urlopen(url, timeout):
        answers.append(url)
        if len(answers) == 1:
            return Response(b"abc")  # annonce 1000 octets, en donne 3
        response = Response(b"x" * 1000)
        return response

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    download.download_subject("tr00-0001", tmp_path)
    folder = tmp_path / "tr00-0001"
    assert (folder / "tr00-0001.hea").read_bytes() == b"x" * 1000 and not list(folder.glob("*.part"))

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda url, timeout: Response(b"abc"))
    with pytest.raises(OSError, match="incomplet"):
        download.download_subject("tr00-0002", tmp_path, retries=2)
    assert not list((tmp_path / "tr00-0002").iterdir())


def test_same_epochs_require_identical_times(tmp_path):
    from scripts.compare_test_nights import same_epochs_problem

    run = {
        "subject": np.array(["a", "a"]),
        "epoch_idx": np.array([0, 1]),
        "y_true": np.array([0, 1]),
        "onset_s": np.array([0.0, 30.0]),
    }

    def archive(name, **changes):
        np.savez(tmp_path / name, **(run | changes))
        return np.load(tmp_path / name)

    rows = np.array([0, 1])
    old = archive("old.npz")
    assert same_epochs_problem(old, archive("same.npz"), rows, rows) is None
    shifted = archive("shift.npz", onset_s=run["onset_s"] + 3600)
    assert "temps" in same_epochs_problem(old, shifted, rows, rows)
    no_times = {k: v for k, v in run.items() if k != "onset_s"}
    np.savez(tmp_path / "none.npz", **no_times)
    assert "absents" in same_epochs_problem(old, np.load(tmp_path / "none.npz"), rows, rows)
    double = archive("double.npz", epoch_idx=np.array([0, 0]))
    assert "double" in same_epochs_problem(double, double, rows, rows)


def test_run_status_covers_exactly_the_requested_subjects(tmp_path):
    from run_baseline import write_run_status

    (tmp_path / "a.npz").write_bytes(b"")
    (tmp_path / "etranger.npz").write_bytes(b"")
    (tmp_path / "b.skipped").write_text("canaux manquants\n")
    status = write_run_status(tmp_path, ["a", "b", "c"], 0.0)
    assert not status["complete"] and status["missing"] == ["c"] and status["not_requested"] == ["etranger"]
    assert status["rejected"] == {"b": "canaux manquants"}
    (tmp_path / "c.npz").write_bytes(b"")
    assert write_run_status(tmp_path, ["a", "b", "c"], 0.0)["complete"]
    (tmp_path / "a.skipped").write_text("x")
    assert not write_run_status(tmp_path, ["a", "b", "c"], 0.0)["complete"]
