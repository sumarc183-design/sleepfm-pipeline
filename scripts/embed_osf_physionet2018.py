"""Embeddings OSF figés sur PhysioNet 2018, mêmes fenêtres que SleepFM (option A, exploratoire).

Pour les 500 sujets de développement du découpage figé (400 d'entraînement, 100 de
test), par lots : téléchargement, extraction annotée identique à celle de SleepFM,
conversion en entrée OSF (`src/data/osf_physionet2018.py` : 8 canaux sur 12, 64 Hz,
z-score par nuit), puis embedding CLS du backbone OSF préentraîné (768 valeurs par
époque). Un <sujet>.npz par sujet (X, y, epoch_idx, onset_s) et un manifest.json.
Le lot suivant se télécharge pendant le calcul du lot en cours ; relancer reprend
où le run s'est arrêté. Aucune sonde ni score ici : la comparaison se fait ensuite.

Les 494 sujets réservés ne sont pas utilisés.

    python scripts/embed_osf_physionet2018.py --split-file configs/split_physionet2018.json \\
        --osf-checkpoint third_party/osf/osf_backbone.pth --out embeddings/physionet2018_osf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import epoch_groups, epoch_onsets, extract  # noqa: E402
from scripts.download_physionet2018 import download_subject  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL, PhysioNet2018EpochDataset  # noqa: E402
from src.data.extract_physionet2018 import PREPROCESSING_VERSION  # noqa: E402
from src.data.osf_physionet2018 import night_to_osf, preparation_contract  # noqa: E402
from src.data.subject_ids import validate_subject_ids  # noqa: E402
from src.models.osf_vit1d import load_osf_backbone  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

OSF_EMBEDDING_DIM = 768


@torch.no_grad()
def embed_night(model, epochs: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, list[str]]:
    """Embeddings CLS d'une nuit (n, 768) et canaux plats signalés par la conversion."""
    series, flat = night_to_osf(epochs)
    chunks = [
        model(torch.from_numpy(series[i : i + batch_size]).to(device)).cpu().numpy()
        for i in range(0, len(series), batch_size)
    ]
    X = np.concatenate(chunks).astype(np.float32)
    if X.shape != (len(epochs), OSF_EMBEDDING_DIM) or not np.isfinite(X).all():
        raise ValueError("Embeddings OSF invalides")
    return X, flat


def save_chunk(processed_dir: Path, chunk: list[str], model, out: Path, batch_size: int, device: str) -> dict:
    """Embeddings des sujets extraits d'un lot ; rejets de l'extraction marqués avec leur raison."""
    flats = {}
    try:
        dataset = PhysioNet2018EpochDataset(processed_dir)
    except FileNotFoundError:
        dataset = None
    if dataset is not None:
        subjects, epoch_idx = epoch_groups(dataset)
        onsets = epoch_onsets(dataset, processed_dir)
        labels = np.array([STAGE_NAME_TO_LABEL[label] for _, label in dataset.samples])
    for subject in chunk:
        rows = np.flatnonzero(subjects == subject) if dataset is not None else np.zeros(0, dtype=int)
        if not len(rows):
            reason_file = processed_dir / "rejected" / f"{subject}.txt"
            reason = reason_file.read_text() if reason_file.exists() else "aucune époque valide\n"
            (out / f"{subject}.skipped").write_text(reason)
            continue
        rows = rows[np.argsort(epoch_idx[rows], kind="stable")]  # ordre chronologique
        epochs = np.stack([np.load(dataset.samples[i][0]).astype(np.float32) for i in rows])
        X, flat = embed_night(model, epochs, batch_size, device)
        if flat:
            flats[subject] = flat
        partial = out / f"{subject}.tmp.npz"
        np.savez(partial, X=X, y=labels[rows], epoch_idx=epoch_idx[rows], onset_s=onsets[rows])
        os.replace(partial, out / f"{subject}.npz")
    return flats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--osf-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="seulement les N premiers sujets (essai court)")
    args = parser.parse_args()
    if min(args.chunk_size, args.batch_size, args.num_threads) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--chunk-size, --batch-size, --num-threads et --limit doivent être positifs")

    split = json.loads(args.split_file.read_text())
    subjects = (split["train"] + split["test"])[: args.limit]
    validate_subject_ids(subjects)
    contract = {
        "model": "OSF",
        "osf_checkpoint_sha256": file_sha256(args.osf_checkpoint),
        "embedding": "CLS",
        "extraction_preprocessing_version": PREPROCESSING_VERSION,
        **preparation_contract(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != contract:
            raise SystemExit(f"{manifest_path} : contrat différent, utiliser un nouveau dossier")
    elif any(args.out.glob("*.npz")) or any(args.out.glob("*.skipped")):
        raise SystemExit(f"{args.out} : fichiers sans manifeste, provenance inconnue")
    else:
        write_json_atomic(manifest_path, contract)
    for leftover in args.out.glob("*.tmp.npz"):
        leftover.unlink()

    def done(s: str) -> bool:
        return (args.out / f"{s}.npz").exists() or (args.out / f"{s}.skipped").exists()

    todo = [s for s in subjects if not done(s)]
    chunks = [todo[i : i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    print(f"{len(subjects) - len(todo)} sujet(s) déjà faits, {len(todo)} à traiter sur {args.device}")
    model = load_osf_backbone(args.osf_checkpoint).to(args.device)
    status_path = args.out / "run_status.json"
    flats = json.loads(status_path.read_text()).get("flat_channels", {}) if status_path.exists() else {}

    def fetch(chunk: list[str]) -> tuple[str, float]:
        tmp = tempfile.mkdtemp()
        try:
            began = time.perf_counter()
            (Path(tmp) / "raw").mkdir()
            for subject in chunk:
                download_subject(subject, Path(tmp) / "raw")
            return tmp, time.perf_counter() - began
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(fetch, chunks[0]) if chunks else None
        try:
            for index, chunk in enumerate(chunks):
                t0 = time.perf_counter()
                tmp, download_s = upcoming.result()
                upcoming = pool.submit(fetch, chunks[index + 1]) if index + 1 < len(chunks) else None
                t1 = time.perf_counter()
                try:
                    extract(Path(tmp) / "raw", Path(tmp) / "processed", args.num_threads)
                    t2 = time.perf_counter()
                    flats |= save_chunk(Path(tmp) / "processed", chunk, model, args.out, args.batch_size, args.device)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                print(
                    f"Lot {index + 1}/{len(chunks)} : téléchargement {download_s:.0f} s (attente {t1 - t0:.0f} s), "
                    f"extraction {t2 - t1:.0f} s, embeddings {time.perf_counter() - t2:.0f} s"
                )
        finally:
            if upcoming is not None:
                try:
                    shutil.rmtree(upcoming.result()[0], ignore_errors=True)
                except Exception:
                    pass

    embedded = [s for s in subjects if (args.out / f"{s}.npz").exists()]
    rejected = {s: (args.out / f"{s}.skipped").read_text().strip() for s in subjects if s not in embedded and done(s)}
    status = {
        "requested": subjects,
        "embedded": embedded,
        "rejected": rejected,
        "missing": [s for s in subjects if not done(s)],
        "complete": all(done(s) for s in subjects),
        "flat_channels": flats,
        "environment": run_environment(),
    }
    write_json_atomic(status_path, status)
    print(f"{len(embedded)} sujet(s) faits, {len(rejected)} rejeté(s) ; complet : {status['complete']}")
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
