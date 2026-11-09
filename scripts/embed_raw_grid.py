"""Embeddings SleepFM sur la grille fixe de 30 s, pour évaluer la prédiction sur PSG brute.

Pour chaque sujet : téléchargement PhysioNet 2018, fenêtres de 30 s depuis le
début de l'enregistrement (`RawNightDataset`, sans annotations), embeddings sur
GPU, puis stade de l'expert par fenêtre lu à part (`grid_labels`), seulement pour
l'évaluation. Un fichier <sujet>.npz par sujet (X, epoch_idx, onset_s, y_grid) et
un manifest.json ; les fichiers bruts sont supprimés après chaque sujet. Le sujet
suivant se télécharge pendant le calcul du sujet en cours. Relancer reprend où le
run s'est arrêté. Aucun entraînement, aucun score : l'évaluation se fait ensuite
sur le Mac avec scripts/evaluate_raw_grid.py.

    python scripts/embed_raw_grid.py --split-file configs/split_physionet2018.json \\
        --out embeddings/physionet2018_raw_grid --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt
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

from run_baseline import EMBEDDING_DIM, embeddings_manifest  # noqa: E402
from scripts.download_physionet2018 import download_subject  # noqa: E402
from src.data.cache_contract import write_json_atomic  # noqa: E402
from src.data.raw_grid_labels import grid_labels  # noqa: E402
from src.data.raw_night import RawNightDataset  # noqa: E402
from src.data.subject_ids import validate_subject_ids  # noqa: E402
from src.models.embeddings import SleepFMEncoders  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

GRID = "fixed_30s_from_recording_start"


@torch.no_grad()
def embed_subject(subject_dir: Path, encoders, batch_size: int, device: str) -> dict:
    dataset = RawNightDataset(subject_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    X = np.concatenate([encoders.embed_combined(epochs.to(device)).cpu().numpy() for epochs in loader])
    if X.shape != (len(dataset), EMBEDDING_DIM) or not np.isfinite(X).all():
        raise ValueError(f"{dataset.subject} : embeddings invalides")
    y_grid, counts = grid_labels(subject_dir, len(dataset))
    return {
        "X": X.astype(np.float32),
        "epoch_idx": dataset.epoch_idx,
        "onset_s": dataset.onset_s,
        "y_grid": y_grid,
        "counts": counts,
        "metadata": dataset.metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", type=Path, required=True, help="sujets de test du découpage figé")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="seulement les N premiers sujets (essai court)")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size et --limit doivent être positifs")

    subjects = json.loads(args.split_file.read_text())["test"][: args.limit]
    validate_subject_ids(subjects)
    contract = embeddings_manifest(args.checkpoint) | {"epoch_selection": GRID}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != contract:
            raise SystemExit(f"{manifest_path} : contrat différent, utiliser un nouveau dossier")
    elif any(args.out.glob("*.npz")):
        raise SystemExit(f"{args.out} : fichiers sans manifeste, provenance inconnue")
    else:
        write_json_atomic(manifest_path, contract)

    todo = [s for s in subjects if not (args.out / f"{s}.npz").exists() and not (args.out / f"{s}.skipped").exists()]
    print(f"{len(subjects) - len(todo)} sujet(s) déjà faits, {len(todo)} à traiter sur {args.device}")
    encoders = SleepFMEncoders(args.checkpoint, embedding_dim=512).to(args.device).eval()
    counts = {}

    def fetch(subject: str) -> tuple[str, float]:
        tmp = tempfile.mkdtemp()
        try:
            began = time.perf_counter()
            download_subject(subject, tmp)
            return tmp, time.perf_counter() - began
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(fetch, todo[0]) if todo else None
        try:
            for index, subject in enumerate(todo):
                t0 = time.perf_counter()
                tmp, download_s = upcoming.result()
                upcoming = pool.submit(fetch, todo[index + 1]) if index + 1 < len(todo) else None
                t1 = time.perf_counter()
                try:
                    result = embed_subject(Path(tmp) / subject, encoders, args.batch_size, args.device)
                except ValueError as exc:
                    # Données refusées (canaux, signal, annotations) : rejet motivé, sans arrêter le run.
                    (args.out / f"{subject}.skipped").write_text(f"{exc}\n")
                    print(f"[{index + 1}/{len(todo)}] {subject} rejeté : {exc}")
                    continue
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                partial = args.out / f"{subject}.tmp.npz"
                np.savez(partial, **{k: result[k] for k in ("X", "epoch_idx", "onset_s", "y_grid")})
                os.replace(partial, args.out / f"{subject}.npz")
                counts[subject] = result["counts"] | {"discarded_tail_s": result["metadata"]["discarded_tail_s"]}
                print(
                    f"[{index + 1}/{len(todo)}] {subject} : {len(result['y_grid'])} fenêtres, "
                    f"{result['counts']['scored']} notées ; téléchargement {download_s:.0f} s "
                    f"(attente {t1 - t0:.0f} s), calcul {time.perf_counter() - t1:.0f} s"
                )
        finally:
            if upcoming is not None:
                try:
                    shutil.rmtree(upcoming.result()[0], ignore_errors=True)
                except Exception:
                    pass

    done = sorted(p.name.removesuffix(".npz") for p in args.out.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    status_path = args.out / "run_status.json"
    previous = json.loads(status_path.read_text()).get("counts", {}) if status_path.exists() else {}
    skipped = {
        s: (args.out / f"{s}.skipped").read_text().strip() for s in subjects if (args.out / f"{s}.skipped").exists()
    }
    status = {
        "requested": subjects,
        "done": [s for s in subjects if s in done],
        "rejected": skipped,
        "missing": [s for s in subjects if s not in done and s not in skipped],
        "complete": all(s in done or s in skipped for s in subjects),
        "counts": previous | counts,
        "environment": run_environment(),
    }
    write_json_atomic(status_path, status)
    print(f"{len(status['done'])} sujet(s) sur {len(subjects)} ; complet : {status['complete']}")
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
