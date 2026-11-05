#!/usr/bin/env python3
"""Point d'entrée unique pour la baseline SleepFM sur PhysioNet 2018.

Enchaîne extraction -> embeddings -> sonde logistique -> évaluation en
réutilisant exclusivement le code déjà testé (src/data/extract_physionet2018.py,
src/models/embeddings.py, src/models/linear_probe.py) : pas de réimplémentation
parallèle.

Sans --raw-data : génère des données synthétiques (comme
tests/test_pipeline_smoke.py) pour un smoke test rapide.
Sans --checkpoint : utilise un checkpoint à poids aléatoires (résultats non
interprétables médicalement, plomberie uniquement -- clairement indiqué en
sortie).

Exemples :
    # Smoke test, tout synthétique
    python run_baseline.py

    # Vos vraies données, checkpoint aléatoire (validation de plomberie)
    python run_baseline.py --raw-data data/raw/physionet2018/training

    # Vos vraies données + vrai checkpoint SleepFM (premier vrai chiffre)
    python run_baseline.py --raw-data data/raw/physionet2018/training \\
        --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt

    # Mode par lots, pour 100+ sujets : télécharge les 200 premiers sujets
    # PhysioNet 2018 au fil de l'eau, ne garde sur disque que les embeddings
    # (un .npz par sujet) et reprend là où il s'est arrêté si on relance
    python run_baseline.py --physionet-subjects 200 --embeddings-dir embeddings/ \\
        --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.download_physionet2018 import download_subject, list_subjects
from scripts.generate_synthetic_physionet2018 import generate_subject
from src.data.cache_contract import file_sha256
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL, PhysioNet2018EpochDataset
from src.data.extract_physionet2018 import ALL_CHANNELS, PREPROCESSING_VERSION
from src.data.extract_physionet2018 import main as extract_main
from src.models.effnet import EffNet
from src.models.embeddings import CHANNEL_INDICES, SleepFMEncoders
from src.models.linear_probe import (
    CV_FOLDS,
    LOGISTIC_MAX_ITER,
    MAX_F1_GAP,
    MIN_XGBOOST_GAIN,
    PROBE_KINDS,
    choose_mlp_or_xgboost,
    compare_probes,
    cross_validate_probes,
)
from src.models.probe_export import run_environment, save_probes


def build_random_checkpoint(path: Path, embedding_dim: int = 512) -> None:
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


def generate_synthetic_raw(raw_dir: Path, n_subjects: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_subjects):
        generate_subject(
            str(raw_dir),
            f"synthetic-{i:04d}",
            sample_rate=32,
            n_epochs=20,
            epoch_sec=30,
            lead_in_sec=2,
            rng=rng,
        )


def extract(raw_dir: Path, processed_dir: Path, num_threads: int) -> None:
    argv_backup, sys.argv = (
        sys.argv,
        [
            "extract",
            "--data_path",
            str(raw_dir),
            "--save_path",
            str(processed_dir),
            "--num_threads",
            str(num_threads),
        ],
    )
    try:
        extract_main()
    finally:
        sys.argv = argv_backup


def split_subjects_train_test(raw_dir: Path, test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    return split_subject_ids([p.name for p in raw_dir.iterdir() if p.is_dir()], test_fraction, seed)


def validate_split(frozen: dict, source: str) -> None:
    """Refuse un découpage incohérent : clés manquantes, doublons, ou sujet
    présent à la fois en train et en test."""
    if not isinstance(frozen, dict) or set(frozen) != {"train", "test"}:
        raise SystemExit(f'{source} : attendu {{"train": [...], "test": [...]}}')
    for role in ("train", "test"):
        if not isinstance(frozen[role], list):
            raise SystemExit(f"{source} : {role!r} doit être une liste de sujets")
        if not all(isinstance(s, str) for s in frozen[role]):
            raise SystemExit(f"{source} : identifiants de sujets non textuels dans {role!r}")
        duplicates = sorted({s for s in frozen[role] if frozen[role].count(s) > 1})
        if duplicates:
            raise SystemExit(f"{source} : sujet(s) en double dans {role!r} : {duplicates}")
    overlap = sorted(set(frozen["train"]) & set(frozen["test"]))
    if overlap:
        raise SystemExit(f"{source} : sujet(s) à la fois en train et en test : {overlap}")


def load_or_extend_split(
    path: Path | None, subjects: list[str], test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Découpage figé dans un fichier JSON ({"train": [...], "test": [...]}).
    Un sujet déjà listé garde toujours son rôle. Les nouveaux sujets sont
    répartis de façon à ramener la part du test à `test_fraction` sur
    l'ensemble (tirés au hasard avec la graine), puis ajoutés au fichier,
    écrit d'un bloc (fichier temporaire renommé). Passer de 100 à 200 sujets
    ne change donc pas le rôle des 100 premiers. Sans fichier (`path=None`),
    découpage aléatoire habituel."""
    if not 0 < test_fraction < 1:
        raise SystemExit(f"--test-fraction doit être entre 0 et 1 (reçu {test_fraction})")
    if path is None:
        return split_subject_ids(subjects, test_fraction, seed)
    frozen = json.loads(path.read_text()) if path.exists() else {"train": [], "test": []}
    validate_split(frozen, str(path))
    known = set(frozen["train"]) | set(frozen["test"])
    new = sorted(set(subjects) - known)
    if new:
        total = len(known) + len(new)
        n_test = min(len(new), max(0, round(total * test_fraction) - len(frozen["test"])))
        rng = np.random.default_rng(seed)
        new_test = set(rng.permutation(new)[:n_test].tolist())
        frozen = {
            "train": sorted(frozen["train"] + [s for s in new if s not in new_test]),
            "test": sorted(frozen["test"] + sorted(new_test)),
        }
        validate_split(frozen, str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(frozen, indent=2) + "\n")
        os.replace(tmp, path)
        print(f"Découpage figé : {len(new)} nouveau(x) sujet(s) ajouté(s) à {path} ({len(new_test)} en test)")
    available = set(subjects)
    missing = sorted(known - available)
    if missing:
        print(f"ATTENTION : {len(missing)} sujet(s) du découpage figé indisponible(s), ignoré(s) : {missing[:5]}...")
    train = [s for s in frozen["train"] if s in available]
    test = [s for s in frozen["test"] if s in available]
    if not train or not test:
        raise SystemExit(f"{path} : aucun sujet disponible en {'train' if not train else 'test'}")
    return train, test


def split_subject_ids(subjects: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split au niveau sujet (pas époque) pour éviter toute fuite -- un même
    sujet ne doit jamais apparaître à la fois en train et en test."""
    subjects = sorted(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_fraction)) if len(subjects) > 1 else 0
    return subjects[n_test:], subjects[:n_test]


def collect_embeddings(
    encoders: SleepFMEncoders, processed_dir: Path, batch_size: int = 32, device: str = "cpu", num_workers: int = 0
):
    """Renvoie X, y et groups = (sujet, numéro d'époque, début en s) de chaque ligne."""
    dataset = PhysioNet2018EpochDataset(processed_dir)
    X, y = embed_dataset(encoders, dataset, batch_size, device, num_workers)
    return X, y, (*epoch_groups(dataset), epoch_onsets(dataset, processed_dir))


def epoch_groups(dataset: PhysioNet2018EpochDataset) -> tuple[np.ndarray, np.ndarray]:
    """(sujet, numéro d'époque) de chaque échantillon, d'après les noms de
    fichiers <sujet>_<idx>.npy. Le dataset trie les fichiers par nom
    ("_10" avant "_2") : le numéro est indispensable pour remettre une nuit
    dans l'ordre chronologique (sonde "sequence")."""
    subjects = np.array([epoch_file.parent.name for epoch_file, _ in dataset.samples])
    epoch_idx = np.array([int(epoch_file.stem.rsplit("_", 1)[1]) for epoch_file, _ in dataset.samples])
    return subjects, epoch_idx


def epoch_onsets(dataset: PhysioNet2018EpochDataset, processed_dir: Path) -> np.ndarray:
    """Début de chaque époque (s depuis le début de l'enregistrement
    d'origine), lu dans T/<sujet>.json ; NaN si absent."""
    onsets_by_subject: dict[str, dict] = {}
    out = np.full(len(dataset.samples), np.nan)
    for i, (epoch_file, _) in enumerate(dataset.samples):
        subject = epoch_file.parent.name
        if subject not in onsets_by_subject:
            path = processed_dir / "T" / f"{subject}.json"
            onsets_by_subject[subject] = json.loads(path.read_text()) if path.exists() else {}
        out[i] = onsets_by_subject[subject].get(epoch_file.name, np.nan)
    return out


def embed_dataset(
    encoders: SleepFMEncoders,
    dataset: PhysioNet2018EpochDataset,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings, labels = [], []
    with torch.no_grad():
        for epochs, batch_labels in loader:
            embeddings.append(encoders.embed_combined(epochs.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def embeddings_manifest(checkpoint_path: Path | None) -> dict:
    """Ce qui détermine les embeddings : poids, prétraitement, montage.
    `checkpoint_path=None` : poids aléatoires (tests, données synthétiques)."""
    return {
        "checkpoint_sha256": "aléatoire" if checkpoint_path is None else file_sha256(checkpoint_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "channels": list(ALL_CHANNELS),
        "sample_rate": 256,
    }


def check_embeddings_manifest(embeddings_dir: Path, expected: dict, n_to_compute: int) -> dict:
    """Le dossier de cache garde un manifest.json décrivant ses embeddings.
    - Checkpoint différent : arrêt, les résultats seraient attribués au
      mauvais modèle.
    - Prétraitement ou montage différent : arrêt s'il reste des sujets à
      calculer (on mélangerait deux versions) ; sinon simple avertissement.
    - Pas de manifeste mais des caches (créés avant son introduction) : on
      l'écrit avec le checkpoint fourni et la version 1 du prétraitement, en
      le signalant (provenance déclarée, pas vérifiable)."""
    path = embeddings_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        legacy = sorted(p.name for p in embeddings_dir.glob("*.npz"))
        manifest = dict(expected)
        if legacy:
            manifest["preprocessing_version"] = 1
            manifest["legacy"] = f"{len(legacy)} cache(s) antérieurs au manifeste : provenance déclarée, non vérifiée"
            print(f"ATTENTION : {manifest['legacy']} ; manifeste écrit avec le checkpoint fourni")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["checkpoint_sha256"] != expected["checkpoint_sha256"]:
        raise SystemExit(
            f"{path} : les embeddings en cache viennent d'un autre checkpoint "
            f"({manifest['checkpoint_sha256'][:12]}… contre {expected['checkpoint_sha256'][:12]}…). "
            "Utiliser un autre --embeddings-dir."
        )
    differs = [k for k in ("preprocessing_version", "channels", "sample_rate") if manifest.get(k) != expected[k]]
    if differs and n_to_compute:
        raise SystemExit(
            f"{path} : cache fait avec un autre {', '.join(differs)} ({ {k: manifest.get(k) for k in differs} }) ; "
            f"{n_to_compute} sujet(s) à calculer mélangeraient deux versions. Utiliser un autre --embeddings-dir."
        )
    if differs:
        print(
            f"ATTENTION : cache fait avec {', '.join(f'{k}={manifest.get(k)}' for k in differs)} (code actuel : "
            f"{', '.join(f'{k}={expected[k]}' for k in differs)}) ; résultats valables pour cette version du cache"
        )
    return manifest


def embed_subjects_in_chunks(
    subject_ids: list[str],
    fetch_subject,
    encoders: SleepFMEncoders,
    embeddings_dir: Path,
    chunk_size: int,
    num_threads: int,
    batch_size: int,
    device: str,
    manifest: dict | None = None,
    retry_skipped: bool = False,
    num_workers: int | None = None,
) -> dict | None:
    """Traite les sujets par lots : brut -> époques -> embeddings, un
    <id>.npz par sujet dans embeddings_dir, puis supprime brut et époques.
    Seuls les embeddings (~5 Mo/sujet, contre ~0,85 Go de brut + époques
    float64) restent sur disque, ce qui permet de passer à 100+ sujets sur
    le disque de Colab. Un sujet déjà présent dans embeddings_dir n'est pas
    recalculé : relancer après une déconnexion reprend où on s'est arrêté.
    Chaque .npz est écrit sous un nom temporaire puis renommé : une
    interruption en pleine écriture ne laisse pas de fichier tronqué pris
    pour un sujet traité. Un .npz illisible (cache écrit avant ce correctif)
    est supprimé et recalculé.

    `fetch_subject(subject_id, dest_dir)` doit déposer les 3 fichiers bruts
    du sujet dans dest_dir/<subject_id>/ (téléchargement ou lien local)."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for leftover in embeddings_dir.glob("*.tmp.npz"):
        leftover.unlink()  # écriture interrompue lors d'un run précédent
    for subject in subject_ids:
        cached = embeddings_dir / f"{subject}.npz"
        problem = cache_problem(cached) if cached.exists() else None
        if problem:
            print(f"[{subject}] cache inutilisable ({problem}) : supprimé, sera recalculé")
            cached.unlink()
    if retry_skipped:
        for marker in embeddings_dir.glob("*.skipped"):
            marker.unlink()  # sujets rejetés retentés (après une correction de l'extraction)
    todo = [
        s
        for s in subject_ids
        if not (embeddings_dir / f"{s}.npz").exists() and not (embeddings_dir / f"{s}.skipped").exists()
    ]
    print(f"{len(subject_ids) - len(todo)} sujet(s) déjà en cache, {len(todo)} à traiter")
    if manifest is not None:
        manifest = check_embeddings_manifest(embeddings_dir, manifest, len(todo))

    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        print(f"Lot {start // chunk_size + 1}/{-(-len(todo) // chunk_size)} : {len(chunk)} sujet(s)")
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir, processed_dir = Path(tmp) / "raw", Path(tmp) / "processed"
            raw_dir.mkdir()
            for subject in chunk:
                fetch_subject(subject, raw_dir)
            extract(raw_dir, processed_dir, num_threads)

            try:
                dataset = PhysioNet2018EpochDataset(processed_dir)
            except FileNotFoundError:
                dataset = None  # aucun sujet du lot n'a passé l'extraction
            if dataset is not None:
                X, y = embed_dataset(
                    encoders, dataset, batch_size, device, num_threads if num_workers is None else num_workers
                )
                sample_subjects, sample_epoch_idx = epoch_groups(dataset)
                sample_onsets = epoch_onsets(dataset, processed_dir)
            for subject in chunk:
                mask = sample_subjects == subject if dataset is not None else np.zeros(0, dtype=bool)
                if not mask.any():
                    # Sujet rejeté par l'extraction (canaux manquants, pas de
                    # stades...) : marqué, avec la raison, pour ne pas le
                    # retenter à chaque relance.
                    reason_file = processed_dir / "rejected" / f"{subject}.txt"
                    reason = reason_file.read_text() if reason_file.exists() else "aucune époque valide\n"
                    (embeddings_dir / f"{subject}.skipped").write_text(reason)
                    continue
                tmp_file = embeddings_dir / f"{subject}.tmp.npz"
                np.savez(
                    tmp_file,
                    X=X[mask].astype(np.float32),
                    y=y[mask],
                    epoch_idx=sample_epoch_idx[mask],
                    onset_s=sample_onsets[mask],
                )
                os.replace(tmp_file, embeddings_dir / f"{subject}.npz")
    return manifest


EMBEDDING_DIM = 3 * 512  # respiratoire + stades + ECG


def cache_problem(path: Path, require_labels: bool = True) -> str | None:
    """Raison pour laquelle un cache d'embeddings est inutilisable, ou None.
    X : matrice (n, 1536) finie. y : vecteur de stades connus (facultatif avec
    require_labels=False, pour prédire sans annotation). epoch_idx : vecteur
    d'entiers positifs ou nuls, sans doublon. onset_s, qui demande epoch_idx :
    vecteur tout entier fini et croissant, ou tout entier NaN (cache historique
    sans temps) ; un temps manquant isolé est refusé."""
    try:
        with np.load(path) as data:
            X = data["X"]
            if X.ndim != 2 or X.shape[1] != EMBEDDING_DIM:
                return f"X de forme {X.shape}, attendu (n, {EMBEDDING_DIM})"
            n = len(X)
            if n == 0:
                return "aucune époque"
            if not np.isfinite(X).all():
                return "X contient des valeurs non finies"
            if "y" in data:
                y = data["y"]
                if y.shape != (n,):
                    return f"y de forme {y.shape}, attendu ({n},)"
                if not np.isin(y, list(STAGE_NAME_TO_LABEL.values())).all():
                    return f"stades inconnus : {sorted(set(y.tolist()) - set(STAGE_NAME_TO_LABEL.values()))}"
            elif require_labels:
                return "pas de stades (y), obligatoires pour évaluer"
            if "onset_s" in data and "epoch_idx" not in data:
                return "onset_s sans epoch_idx"
            if "epoch_idx" in data:
                idx = data["epoch_idx"]
                if idx.shape != (n,) or not np.issubdtype(idx.dtype, np.integer):
                    return f"epoch_idx de forme {idx.shape} et de type {idx.dtype}, attendu ({n},) entiers"
                if (idx < 0).any() or len(np.unique(idx)) != n:
                    return "epoch_idx négatif ou avec doublons"
                if "onset_s" in data:
                    onsets = data["onset_s"]
                    if onsets.shape != (n,):
                        return f"onset_s de forme {onsets.shape}, attendu ({n},)"
                    if not np.isnan(onsets).all():
                        if not np.isfinite(onsets).all():
                            return "onset_s partiellement manquant ou infini"
                        if (np.diff(onsets[np.argsort(idx)]) <= 0).any():
                            return "onset_s ne croît pas avec epoch_idx"
            elif not require_labels:
                return "epoch_idx obligatoire pour prédire sans stades"
    except Exception as exc:
        return f"illisible ({exc.__class__.__name__})"
    return None


def is_valid_cache(path: Path) -> bool:
    return cache_problem(path) is None


def load_cached_embeddings(embeddings_dir: Path, subjects: list[str], require_labels: bool = True):
    """Renvoie X, y et groups = (sujet, numéro d'époque, début en s) de chaque
    ligne. Le début (onset_s) vaut NaN pour un cache qui ne l'a pas : la nuit
    est alors supposée continue. Un cache invalide arrête le run. Avec
    require_labels=False, y vaut None si un cache n'a pas de stades."""
    X, y, groups_subject, groups_epoch, groups_onset = [], [], [], [], []
    without_onsets = 0
    for subject in subjects:
        path = embeddings_dir / f"{subject}.npz"
        problem = cache_problem(path, require_labels)
        if problem:
            raise SystemExit(f"{path} : cache inutilisable ({problem})")
        with np.load(path) as data:
            X.append(data["X"])
            y.append(data["y"] if "y" in data else None)
            if "epoch_idx" in data:
                epoch_idx = data["epoch_idx"]
            else:
                # Cache antérieur à epoch_idx : lignes dans l'ordre des noms de
                # fichiers <sujet>_0..n-1.npy (numérotation sans trou à
                # l'extraction), on retrouve donc le numéro de chaque ligne.
                n = len(data["X"])
                epoch_idx = np.array(sorted(range(n), key=lambda i: f"{subject}_{i}.npy"))
            onsets = data["onset_s"] if "onset_s" in data else np.full(len(epoch_idx), np.nan)
        without_onsets += int(np.isnan(onsets).all())
        groups_subject.append(np.full(len(epoch_idx), subject))
        groups_epoch.append(epoch_idx)
        groups_onset.append(onsets.astype(float))
    if without_onsets:
        print(
            f"ATTENTION : {without_onsets} cache(s) sans temps d'origine (onset_s) : "
            "nuits supposées continues, sans trou d'enregistrement"
        )
    groups = (np.concatenate(groups_subject), np.concatenate(groups_epoch), np.concatenate(groups_onset))
    labels = None if any(part is None for part in y) else np.concatenate(y)
    return np.concatenate(X), labels, groups


def load_verified_split(
    embeddings_dir: Path,
    split_file: Path | None,
    checkpoint: Path | None,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    """Chargement commun aux scripts d'analyse : contrôle du manifeste du cache
    (même checkpoint), découpage figé validé, caches vérifiés. Renvoie
    (X_train, y_train, train_groups), (X_test, y_test, test_groups)."""
    check_embeddings_manifest(embeddings_dir, embeddings_manifest(checkpoint), n_to_compute=0)
    subjects = sorted(p.name.removesuffix(".npz") for p in embeddings_dir.glob("*.npz") if ".tmp." not in p.name)
    train, test = load_or_extend_split(split_file, subjects, test_fraction, seed)
    return load_cached_embeddings(embeddings_dir, train), load_cached_embeddings(embeddings_dir, test)


def select_and_evaluate(args, X_train, y_train, train_groups, X_test, y_test, test_groups) -> tuple[dict, dict | None]:
    """1. Si MLP et XGBoost sont demandés, en garde un seul par validation
    croisée par sujet sur le TRAIN (choose_mlp_or_xgboost) ;
    2. évalue sur le test, une seule fois, les sondes retenues -- la sonde
    écartée n'est jamais évaluée sur le test.
    Renvoie (résultats finaux, sélection ou None)."""
    probes, selection = list(args.probes), None
    if "mlp" in probes and "xgboost" in probes:
        print(
            f"Sélection MLP / XGBoost : validation croisée {args.cv_folds} plis par sujet, sur le train uniquement..."
        )
        cv = cross_validate_probes(
            ["mlp", "xgboost"],
            X_train,
            y_train,
            train_groups[0],
            n_folds=args.cv_folds,
            max_iter=args.max_iter,
            seed=args.seed,
            device=args.device,
        )
        chosen, reason = choose_mlp_or_xgboost(cv, args.min_gain, args.max_gap)
        selection = {"cv": cv, "chosen": chosen, "reason": reason}
        probes = [k for k in probes if k not in ("mlp", "xgboost") or k == chosen]
    results = compare_probes(
        probes,
        X_train,
        y_train,
        X_test,
        y_test,
        max_iter=args.max_iter,
        seed=args.seed,
        device=args.device,
        train_groups=train_groups,
        test_groups=test_groups,
    )
    return results, selection


PROBE_LABELS = {
    "logistic": "Log. officielle",
    "logistic_unweighted": "Log. non pond.",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "sequence": "BiGRU (nuit)",
}


def print_results(
    results: dict, data_desc: str, checkpoint_desc: str, not_interpretable: bool, selection: dict | None = None
) -> None:
    """`results` : sortie de compare_probes, {sonde: {"train": ..., "test": ..., "fit_seconds": ...}}.
    `selection` : sortie de select_and_evaluate (validation croisée MLP / XGBoost).
    Gap = score train - score test : un gap élevé signale du surapprentissage."""
    label_to_stage = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}
    width = 96
    print("=" * width)
    print(f"Données    : {data_desc}")
    print(f"Checkpoint : {checkpoint_desc}")
    if selection is not None:
        cv = selection["cv"]
        n_folds = len(next(iter(cv.values()))["val_f1"])
        print("-" * width)
        print(f"1. Sélection MLP / XGBoost -- validation croisée {n_folds} plis par sujet, train uniquement")
        print(f"{'':<16}{'F1 val moyen':>14}{'écart-type':>12}{'Gap F1':>9}   F1 val par pli")
        for kind in ("mlp", "xgboost"):
            val, gap = np.array(cv[kind]["val_f1"]), np.array(cv[kind]["train_f1"]) - np.array(cv[kind]["val_f1"])
            print(
                f"{PROBE_LABELS[kind]:<16}{val.mean():>14.4f}{val.std():>12.4f}{gap.mean():>+9.4f}   "
                + " ".join(f"{v:.3f}" for v in val)
            )
        print(f"Verdict : {selection['reason']}")
    print("-" * width)
    print("2. Évaluation finale sur le test (jamais utilisé pour la sélection)")
    print(
        f"{'Modèle (test)':<16}{'Accuracy':>9}{'Kappa':>8}{'F1 macro':>10}{'ROC-AUC':>9}{'AUPRC':>8}"
        f" |{'F1 train':>9}{'Gap F1':>8}{'AUC train':>10}{'Gap AUC':>9}{'Durée':>8}"
    )
    for kind, r in results.items():
        test, train = r["test"], r["train"]
        print(
            f"{PROBE_LABELS.get(kind, kind):<16}{test['accuracy']:>9.4f}{test['cohen_kappa']:>8.4f}"
            f"{test['macro_f1']:>10.4f}{test['roc_auc_macro']:>9.4f}{test['auprc_macro']:>8.4f}"
            f" |{train['macro_f1']:>9.4f}{train['macro_f1'] - test['macro_f1']:>+8.4f}"
            f"{train['roc_auc_macro']:>10.4f}{train['roc_auc_macro'] - test['roc_auc_macro']:>+9.4f}"
            f"{r['fit_seconds']:>7.0f}s"
        )
    print("-" * width)
    print("Intervalles de confiance à 95 % (bootstrap sur les sujets de test)")
    for kind, r in results.items():
        ci = r["test"].get("ci95")
        if ci:
            print(
                f"  {PROBE_LABELS.get(kind, kind):<20}kappa [{ci['cohen_kappa'][0]:.3f} ; {ci['cohen_kappa'][1]:.3f}]"
                f"   F1 macro [{ci['macro_f1'][0]:.3f} ; {ci['macro_f1'][1]:.3f}]   ({ci['n_subjects']} sujets)"
            )
    paired = {kind: r["vs_baseline"] for kind, r in results.items() if "vs_baseline" in r}
    if paired:
        print("-" * width)
        print("Comparaison appariée à la baseline (mêmes nuits de test) : écart, IC 95 %")
        for kind, d in paired.items():
            k, f = d["cohen_kappa"], d["macro_f1"]
            verdict = "significatif" if k["ci95"][0] > 0 or k["ci95"][1] < 0 else "non significatif"
            print(
                f"  {PROBE_LABELS.get(kind, kind):<20}kappa {k['diff']:+.3f} "
                f"[{k['ci95'][0]:+.3f} ; {k['ci95'][1]:+.3f}]   F1 macro {f['diff']:+.3f} "
                f"[{f['ci95'][0]:+.3f} ; {f['ci95'][1]:+.3f}]   ({verdict} sur le kappa)"
            )
    for kind, r in results.items():
        if r.get("converged") is False:
            print(
                f"  ATTENTION : {PROBE_LABELS.get(kind, kind)} n'a PAS convergé en {r['n_iter']} itérations "
                "(augmenter --max-iter) : baseline non fiable"
            )
        elif r.get("converged"):
            print(f"  {PROBE_LABELS.get(kind, kind)} : convergé en {r['n_iter']} itérations")
    print("-" * width)
    first_report = next(iter(results.values()))["test"]["per_class_report"]
    stages = [label for label in first_report if label.isdigit()]
    print("F1 par stade (test) " + "".join(f"{PROBE_LABELS.get(k, k):>16}" for k in results) + f"{'n':>8}")
    for label in stages:
        scores = "".join(
            f"{r['test']['per_class_report'].get(label, {}).get('f1-score', float('nan')):>16.4f}"
            for r in results.values()
        )
        print(f"  {label_to_stage[int(label)]:<18}{scores}{int(first_report[label]['support']):>8}")
    baseline = results.get("logistic") or next(iter(results.values()))
    if "logistic" in results:
        print_published_comparison(results["logistic"]["test"], label_to_stage)
    classes = [label_to_stage[c] for c in baseline["test"]["classes"]]
    name = PROBE_LABELS.get("logistic" if "logistic" in results else next(iter(results)), "")
    print("-" * width)
    print(f"Matrice de confusion (test, {name}) : lignes = stade réel, colonnes = stade prédit")
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in classes))
    for stage, row in zip(classes, baseline["test"]["confusion_matrix"], strict=True):
        print(f"  {stage:<10}" + "".join(f"{v:>10d}" for v in row))
    print("=" * width)
    if not_interpretable:
        print(
            "Ces résultats ne sont PAS interprétables médicalement (données et/ou "
            "checkpoint non réels) -- ils valident uniquement que le pipeline tourne."
        )


# SleepFM 2024 (Thapa et al., ICML 2024, arXiv 2405.17766), section 4.6,
# tableau 8 : validation externe sur PhysioNet/CinC 2018, 100 participants
# en test. Pas de kappa publié sur cette base.
PUBLISHED_PHYSIONET2018 = {
    "wake": {"roc_auc": 0.966, "auprc": 0.867, "f1": 0.790},
    "nonrem1": {"roc_auc": 0.830, "auprc": 0.471, "f1": 0.439},
    "nonrem2": {"roc_auc": 0.902, "auprc": 0.857, "f1": 0.793},
    "nonrem3": {"roc_auc": 0.971, "auprc": 0.821, "f1": 0.743},
    "rem": {"roc_auc": 0.950, "auprc": 0.778, "f1": 0.717},
    "macro": {"roc_auc": 0.924, "auprc": 0.759, "f1": 0.700},
}


def print_published_comparison(test: dict, label_to_stage: dict) -> None:
    """Baseline officielle contre le tableau 8 de l'article, stade par stade.
    Protocole différent (nombre de nuits de test, découpage) : à lire comme un
    ordre de grandeur, pas comme une reproduction stricte."""
    rows = {
        label_to_stage[c]: {
            **test["per_class_auc"][c],
            "f1": test["per_class_report"].get(str(c), {}).get("f1-score", float("nan")),
        }
        for c in test["classes"]
        if c in test["per_class_auc"]
    }
    rows["macro"] = {"roc_auc": test["roc_auc_macro"], "auprc": test["auprc_macro"], "f1": test["macro_f1"]}
    print("-" * 96)
    print("Baseline officielle contre l'article SleepFM 2024 (tableau 8, PhysioNet 2018) : nous / article")
    print(f"  {'Stade':<10}{'ROC-AUC':>18}{'AUPRC':>18}{'F1':>18}")
    for stage, ours in rows.items():
        ref = PUBLISHED_PHYSIONET2018.get(stage)
        if ref:
            print(f"  {stage:<10}" + "".join(f"{ours[m]:>10.3f} / {ref[m]:.3f}" for m in ("roc_auc", "auprc", "f1")))


def save_results_json(
    path: Path,
    results: dict,
    selection: dict | None,
    data_desc: str,
    checkpoint_desc: str,
    args,
    train_subjects: list[str],
    test_subjects: list[str],
    y_test: np.ndarray | None = None,
    test_groups: tuple[np.ndarray, np.ndarray] | None = None,
    embeddings_manifest: dict | None = None,
) -> None:
    """Métriques au format JSON, pour comparer des runs sans relire les logs.
    `per_class_report` est réduit au F1 et à l'effectif de chaque stade.
    Les sujets et les réglages sont enregistrés : deux runs ne se comparent
    que s'ils ont été évalués sur les mêmes nuits, avec les mêmes règles."""
    label_to_stage = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}

    def summarize(metrics: dict) -> dict:
        aucs = metrics.get("per_class_auc", {})
        per_stage = {
            label_to_stage[int(label)]: {"f1": m["f1-score"], "support": int(m["support"])} | aucs.get(int(label), {})
            for label, m in metrics["per_class_report"].items()
            if label.isdigit()
        }
        hidden = ("per_class_report", "per_class_auc")
        return {k: v for k, v in metrics.items() if k not in hidden} | {"per_stage": per_stage}

    payload = {
        "data": data_desc,
        "checkpoint": checkpoint_desc,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "settings": {
            "probes": list(args.probes),
            "max_iter": args.max_iter,
            "cv_folds": args.cv_folds,
            "min_gain": args.min_gain,
            "max_gap": args.max_gap,
        },
        "split_file": None if getattr(args, "split_file", None) is None else str(args.split_file),
        "split_sha256": None
        if getattr(args, "split_file", None) is None or not args.split_file.exists()
        else file_sha256(args.split_file),
        "embeddings_manifest": embeddings_manifest,
        "environment": run_environment(),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "probes": {
            kind: {
                "test": summarize(r["test"]),
                "train": summarize(r["train"]),
                "fit_seconds": r["fit_seconds"],
                "converged": r.get("converged"),
                "n_iter": r.get("n_iter"),
                "vs_baseline": r.get("vs_baseline"),
            }
            for kind, r in results.items()
        },
        "selection": None
        if selection is None
        else {"chosen": selection["chosen"], "reason": selection["reason"], "cv": selection["cv"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Résultats JSON : {path}")
    if y_test is not None and test_groups is not None:
        save_test_predictions(path.with_suffix(".predictions.npz"), results, y_test, test_groups)
    if getattr(args, "save_models", None) is not None:
        models = {kind: r["model"] for kind, r in results.items() if "model" in r}
        save_probes(args.save_models, models, EMBEDDING_DIM, embeddings_manifest)


def save_test_predictions(path: Path, results: dict, y_test: np.ndarray, test_groups) -> None:
    """Prédictions et probabilités de chaque sonde sur le test, époque par
    époque, avec le sujet et le numéro d'époque : IC, comparaisons appariées
    et analyse d'erreurs se recalculent sans réentraîner (environ 1 Mo)."""
    arrays = {"y_true": y_test, "subject": np.asarray(test_groups[0]), "epoch_idx": np.asarray(test_groups[1])}
    if len(test_groups) > 2:
        arrays["onset_s"] = np.asarray(test_groups[2])
    for kind, r in results.items():
        out = r.get("test_outputs")
        if out is not None:
            arrays[f"pred_{kind}"] = out["pred"]
            arrays[f"proba_{kind}"] = out["proba"].astype(np.float32)
            arrays[f"classes_{kind}"] = np.asarray(out["classes"])
    np.savez_compressed(path, **arrays)
    print(f"Prédictions du test : {path}")


def run_in_chunks(args, checkpoint_path: Path, tmp_path: Path) -> None:
    """Mode --embeddings-dir : voir embed_subjects_in_chunks."""
    if args.physionet_subjects:
        subject_ids = list_subjects("training")[: args.physionet_subjects]
        fetch_subject = download_subject
        data_kind = "RÉELLES"
    else:
        raw_dir = args.raw_data
        if raw_dir is None:
            raw_dir = tmp_path / "raw"
            print(f"[synthétique] Génération de {args.num_synthetic_subjects} sujets factices...")
            generate_synthetic_raw(raw_dir, args.num_synthetic_subjects, args.seed)
        subject_ids = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

        def fetch_subject(subject: str, dest_dir: Path) -> None:
            (dest_dir / subject).symlink_to((raw_dir / subject).resolve(), target_is_directory=True)

        data_kind = "SYNTHÉTIQUES" if args.raw_data is None else "RÉELLES"

    encoders = SleepFMEncoders(checkpoint_path, embedding_dim=512).to(args.device)
    print(f"Embeddings sur : {args.device}")
    manifest = embed_subjects_in_chunks(
        subject_ids,
        fetch_subject,
        encoders,
        args.embeddings_dir,
        args.chunk_size,
        args.num_threads,
        args.batch_size,
        args.device,
        manifest=embeddings_manifest(args.checkpoint),
        retry_skipped=args.retry_skipped,
        num_workers=args.num_workers,
    )
    if args.embed_only:
        n_done = sum((args.embeddings_dir / f"{s}.npz").exists() for s in subject_ids)
        n_rejected = sum((args.embeddings_dir / f"{s}.skipped").exists() for s in subject_ids)
        print(
            f"Embeddings seulement : {n_done} sujet(s) en cache, {n_rejected} rejeté(s), dans {args.embeddings_dir}. "
            "Sondes à lancer ensuite, sans GPU, sur ce dossier."
        )
        return

    cached = [s for s in subject_ids if (args.embeddings_dir / f"{s}.npz").exists()]
    n_skipped = len(subject_ids) - len(cached)
    if len(cached) < 2:
        raise SystemExit(f"Seulement {len(cached)} sujet(s) exploitable(s) : impossible de faire un split train/test.")
    train_subjects, test_subjects = load_or_extend_split(args.split_file, cached, args.test_fraction, args.seed)
    X_train, y_train, train_groups = load_cached_embeddings(args.embeddings_dir, train_subjects)
    X_test, y_test, test_groups = load_cached_embeddings(args.embeddings_dir, test_subjects)

    results, selection = select_and_evaluate(args, X_train, y_train, train_groups, X_test, y_test, test_groups)
    data_desc = (
        f"{data_kind} ({len(train_subjects)} sujet(s) train / {len(test_subjects)} test, "
        f"{len(y_train)} / {len(y_test)} époques ; {n_skipped} sujet(s) rejeté(s) à l'extraction)"
    )
    checkpoint_desc = "ALÉATOIRE (pas de poids SleepFM officiels)" if args.checkpoint is None else str(args.checkpoint)
    print_results(
        results, data_desc, checkpoint_desc, data_kind == "SYNTHÉTIQUES" or args.checkpoint is None, selection
    )
    if args.results_json is not None:
        save_results_json(
            args.results_json,
            results,
            selection,
            data_desc,
            checkpoint_desc,
            args,
            train_subjects,
            test_subjects,
            y_test,
            test_groups,
            manifest,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=None,
        help="Dossier des sujets PhysioNet 2018 (.hea/.mat/-arousal.mat). Omis = synthétique.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Checkpoint SleepFM officiel. Omis = poids aléatoires."
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction de sujets pour le test : 0,2 = split 80/20, par sujet (jamais par époque)",
    )
    parser.add_argument("--num-synthetic-subjects", type=int, default=6)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Workers du DataLoader des embeddings ; 0 = dans le processus principal. Par défaut : --num-threads.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda (GPU Colab/Kaggle) ou cpu. Par défaut : cuda si disponible.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Mode par lots : un .npz d'embeddings par sujet dans ce dossier, brut et époques "
        "supprimés au fur et à mesure, reprise automatique (voir embed_subjects_in_chunks).",
    )
    parser.add_argument(
        "--physionet-subjects",
        type=int,
        default=None,
        help="Avec --embeddings-dir : télécharge au fil de l'eau les N premiers sujets "
        "d'entraînement PhysioNet 2018 (au lieu de --raw-data).",
    )
    parser.add_argument("--chunk-size", type=int, default=8, help="Sujets par lot en mode --embeddings-dir")
    parser.add_argument(
        "--save-models",
        type=Path,
        default=None,
        help="Avec --results-json : exporte les sondes entraînées dans ce dossier (rechargeables, "
        "voir scripts/predict_night.py)",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Mode par lots : calcule seulement les embeddings, sans entraîner les sondes. Sur GPU loué ou "
        "à quota (Kaggle, Colab), les sondes tournent ensuite sur CPU à partir du cache : GPU économisé.",
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Mode par lots : retente les sujets rejetés (<sujet>.skipped), par exemple après une correction",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Découpage train/test figé (JSON) : un sujet listé garde son rôle, les nouveaux sont "
        "répartis puis ajoutés. Ex. configs/split_physionet2018.json",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=LOGISTIC_MAX_ITER,
        help="Itérations max de la régression logistique (convergence vérifiée et affichée)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Enregistre les métriques (test, train, sélection) dans ce fichier JSON",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help="Plis de la validation croisée par sujet (sur le train) qui départage MLP et XGBoost",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=MIN_XGBOOST_GAIN,
        help="Gain moyen minimal en F1 macro de validation pour préférer XGBoost au MLP (plus rapide)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=MAX_F1_GAP,
        help="Gap F1 moyen (apprentissage - validation) au-delà duquel MLP/XGBoost est éliminé",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=PROBE_KINDS,
        default=list(PROBE_KINDS),
        help="Sondes à comparer sur les mêmes embeddings. La régression logistique est la "
        "baseline (protocole officiel SleepFM).",
    )
    args = parser.parse_args()

    if args.physionet_subjects and args.embeddings_dir is None:
        parser.error("--physionet-subjects nécessite --embeddings-dir")
    if args.physionet_subjects and args.raw_data is not None:
        parser.error("--physionet-subjects et --raw-data sont exclusifs")
    if args.embed_only and args.embeddings_dir is None:
        parser.error("--embed-only demande --embeddings-dir (mode par lots)")
    if args.num_workers is None:
        args.num_workers = args.num_threads
    if args.num_workers < 0:
        parser.error("--num-workers doit être positif ou nul")
    if sys.platform == "darwin" and "xgboost" in args.probes and not args.device.startswith("cuda"):
        # Sur Mac, XGBoost et torch chargent chacun leur OpenMP : XGBoost plante
        # (segmentation fault) ou se bloque dès que torch a calculé dans le
        # même processus, même sur un seul thread. Linux (Kaggle, Colab) n'est
        # pas concerné.
        args.probes = [p for p in args.probes if p != "xgboost"]
        print("ATTENTION : XGBoost retiré des sondes sur Mac (conflit OpenMP avec torch) ; à lancer sur Linux.")

    is_synthetic = args.raw_data is None
    is_random_checkpoint = args.checkpoint is None

    if args.embeddings_dir is not None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = args.checkpoint
            if is_random_checkpoint:
                checkpoint_path = Path(tmp) / "random_checkpoint.pt"
                build_random_checkpoint(checkpoint_path)
            run_in_chunks(args, checkpoint_path, Path(tmp))
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = args.raw_data
        if is_synthetic:
            raw_dir = tmp_path / "raw"
            print(f"[synthétique] Génération de {args.num_synthetic_subjects} sujets factices...")
            generate_synthetic_raw(raw_dir, args.num_synthetic_subjects, args.seed)

        train_subjects, test_subjects = load_or_extend_split(
            args.split_file,
            sorted(p.name for p in raw_dir.iterdir() if p.is_dir()),
            args.test_fraction,
            args.seed,
        )
        if not test_subjects:
            print("ATTENTION : un seul sujet disponible -> pas de split train/test possible.")
            print(
                "Les résultats ci-dessous sont une simple vérification de plomberie, "
                "pas une mesure de généralisation.\n"
            )
            train_subjects = test_subjects = train_subjects  # même sujet des deux côtés, assumé et annoncé

        processed_train, processed_test = tmp_path / "train", tmp_path / "test"
        for subset, out_dir in [(train_subjects, processed_train), (test_subjects, processed_test)]:
            subset_raw = tmp_path / f"raw_{out_dir.name}"
            subset_raw.mkdir(parents=True, exist_ok=True)
            for subject in subset:
                target = subset_raw / subject
                if not target.exists():
                    shutil.copytree(raw_dir / subject, target)
            extract(subset_raw, out_dir, args.num_threads)

        checkpoint_path = args.checkpoint
        if is_random_checkpoint:
            checkpoint_path = tmp_path / "random_checkpoint.pt"
            build_random_checkpoint(checkpoint_path)

        encoders = SleepFMEncoders(checkpoint_path, embedding_dim=512).to(args.device)
        print(f"Embeddings sur : {args.device}")
        X_train, y_train, train_groups = collect_embeddings(
            encoders, processed_train, args.batch_size, args.device, args.num_workers
        )
        X_test, y_test, test_groups = collect_embeddings(
            encoders, processed_test, args.batch_size, args.device, args.num_workers
        )
        n_train, n_test = len(y_train), len(y_test)

        results, selection = select_and_evaluate(args, X_train, y_train, train_groups, X_test, y_test, test_groups)

    data_desc = (
        f"{'SYNTHÉTIQUES' if is_synthetic else 'RÉELLES'} "
        f"({len(train_subjects)} sujet(s) train / {len(test_subjects)} test, {n_train} / {n_test} époques)"
    )
    checkpoint_desc = "ALÉATOIRE (pas de poids SleepFM officiels)" if is_random_checkpoint else str(args.checkpoint)
    print_results(results, data_desc, checkpoint_desc, is_synthetic or is_random_checkpoint, selection)
    if args.results_json is not None:
        save_results_json(
            args.results_json,
            results,
            selection,
            data_desc,
            checkpoint_desc,
            args,
            train_subjects,
            test_subjects,
            y_test,
            test_groups,
            embeddings_manifest(args.checkpoint),
        )


if __name__ == "__main__":
    main()
