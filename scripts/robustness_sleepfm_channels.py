"""Robustesse de la sonde respiratoire SleepFM aux canaux absents et à la respiration sous-échantillonnée.

Plan fixé avant calcul : docs/plans/Plan robustesse SleepFM canaux PhysioNet 16-11.md (passer son chemin
avec --plan, empreinte enregistrée). Diagnostic descriptif sur les 40 nuits de validation : aucune règle,
aucun modèle ni seuil ne change. Téléchargement nuit par nuit (la suivante pendant le calcul),
fichiers supprimés après lecture ; reprise après arrêt.

    python scripts/robustness_sleepfm_channels.py --plan "docs/plans/Plan robustesse SleepFM canaux PhysioNet 16-11.md"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import mne
import numpy as np
import scipy.signal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_physionet2018 import download_subject  # noqa: E402
from src.data.ahi import AHI_CUTOFF, auc, positive_rate, spearman  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.extract_physionet2018 import ALL_CHANNELS  # noqa: E402
from src.data.raw_night import EPOCH_SAMPLES, SAMPLE_RATE, RawNightDataset  # noqa: E402
from src.models.embeddings import SleepFMEncoders  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESP = ("ABD", "CHEST", "AIRFLOW")
CONDITIONS = {
    "intact": {"zero": (), "rates": {}},
    "eeg_mesa": {"zero": ("F3-M2", "F4-M1", "C3-M2", "O1-M2", "O2-M1"), "rates": {}},
    "resp_8hz": {"zero": (), "rates": {**dict.fromkeys(RESP, 8), "SaO2": 8}},
    "mesa_like": {
        "zero": ("F3-M2", "F4-M1", "C3-M2", "O1-M2", "O2-M1"),
        "rates": {**dict.fromkeys(RESP, 32), "SaO2": 1},
    },
    "shhs_like": {"zero": ("F3-M2", "F4-M1", "O1-M2", "O2-M1"), "rates": {**dict.fromkeys(RESP, 10), "SaO2": 1}},
    "ucddb_like": {"zero": ("F3-M2", "F4-M1", "O1-M2", "O2-M1"), "rates": {**dict.fromkeys(RESP, 8), "SaO2": 8}},
}


def through_rate(signal: np.ndarray, rate: int) -> np.ndarray:
    """Signal à 256 Hz passé à `rate` Hz puis remonté à 256 Hz (même méthode que la préparation NSRR)."""
    low = mne.filter.resample(signal, up=rate, down=SAMPLE_RATE, npad="auto", verbose=False)
    back = mne.filter.resample(low, up=SAMPLE_RATE, down=rate, npad="auto", verbose=False)
    out = np.zeros_like(signal)
    n = min(len(back), len(signal))
    out[:n] = back[:n]
    return out


def degrade(signal: np.ndarray, condition: dict) -> np.ndarray:
    """Copie de la nuit (canaux ALL_CHANNELS × échantillons à 256 Hz) avec la dégradation demandée."""
    out = signal.copy()
    for channel in condition["zero"]:
        out[ALL_CHANNELS.index(channel)] = 0.0
    for channel, rate in condition["rates"].items():
        index = ALL_CHANNELS.index(channel)
        out[index] = through_rate(out[index], rate)
    return out


@torch.no_grad()
def night_rates(dataset: RawNightDataset, encoders, model, threshold: float, batch: int) -> dict:
    out = {}
    for name, condition in CONDITIONS.items():
        signal = degrade(dataset.signal, condition)
        X = []
        for start in range(0, dataset.n_epochs, batch):
            idx = range(start, min(start + batch, dataset.n_epochs))
            epochs = np.stack(
                [
                    scipy.signal.resample(signal[:, i * EPOCH_SAMPLES : (i + 1) * EPOCH_SAMPLES], EPOCH_SAMPLES, axis=1)
                    for i in idx
                ]
            ).astype(np.float32)
            X.append(encoders.embed_combined(torch.from_numpy(epochs)).numpy())
        probability = model.predict_proba(np.concatenate(X))[:, 1]
        rate, _, _ = positive_rate(probability, threshold)
        out[name] = {"rate": float(rate), "max_probability": float(probability.max())}
    return out


def summary(nights: dict, reference: dict) -> dict:
    subjects = sorted(nights)
    truth = np.array([reference[s] >= AHI_CUTOFF for s in subjects])
    ref = np.array([reference[s] for s in subjects])
    intact = np.array([nights[s]["intact"]["rate"] for s in subjects])
    out = {}
    for name in CONDITIONS:
        rate = np.array([nights[s][name]["rate"] for s in subjects])
        out[name] = {
            "auc_rate": auc(rate, truth),
            "spearman_reference": spearman(rate, ref),
            "zero_rate_nights": int((rate == 0).sum()),
            "median_rate": float(np.median(rate)),
            "median_rate_over_intact": float(np.median(rate) / np.median(intact)) if np.median(intact) > 0 else None,
            "prohibitive": bool(auc(rate, truth) < 0.70 or (rate == 0).sum() > len(rate) / 2),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/robustness_sleepfm_channels.json")
    parser.add_argument("--limit", type=int, default=None, help="essai court")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    frozen_path = ROOT / "results/respiratory_ahi_physionet2018.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    baseline = ROOT / "results/respiratory_baseline_physionet2018.json"
    bundle = joblib.load(baseline.with_suffix(".joblib"))
    if file_sha256(baseline.with_suffix(".joblib")) != json.loads(baseline.read_text(encoding="utf-8"))["model_sha256"]:
        raise SystemExit("Sonde SleepFM modifiée")
    threshold = float(frozen["thresholds"]["sleepfm"])
    nights_npz = np.load(ROOT / "results/respiratory_ahi_physionet2018.nights.npz", allow_pickle=True)
    validation = nights_npz["partition"] == "validation"
    reference = dict(
        zip(nights_npz["subject"][validation].tolist(), nights_npz["reference_ahi"][validation].tolist(), strict=True)
    )
    subjects = sorted(reference)[: args.limit]
    checkpoint = ROOT / "third_party/sleepfm/sleepfm/checkpoint/best.pt"
    encoders = SleepFMEncoders(checkpoint, embedding_dim=512).eval()
    partial = args.out.with_suffix(".partial.json")
    nights = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    todo = [s for s in subjects if s not in nights]
    print(f"{len(nights)} nuit(s) déjà faites, {len(todo)} à traiter")

    def fetch(subject: str) -> str:
        tmp = tempfile.mkdtemp()
        try:
            download_subject(subject, tmp)
            return tmp
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(fetch, todo[0]) if todo else None
        for index, subject in enumerate(todo):
            tmp = upcoming.result()
            upcoming = pool.submit(fetch, todo[index + 1]) if index + 1 < len(todo) else None
            try:
                dataset = RawNightDataset(Path(tmp) / subject)
                nights[subject] = night_rates(dataset, encoders, bundle["model"], threshold, args.batch_size)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            write_json_atomic(partial, nights)
            print(
                f"[{index + 1}/{len(todo)}] {subject} : "
                + ", ".join(f"{k} {v['rate']:.3f}" for k, v in nights[subject].items())
            )

    done = {s: nights[s] for s in subjects}
    write_json_atomic(
        args.out,
        {
            "plan": args.plan.name,
            "plan_sha256": file_sha256(args.plan),
            "frozen_ahi_sha256": file_sha256(frozen_path),
            "threshold": threshold,
            "conditions": CONDITIONS,
            "n_nights": len(done),
            "summary": summary(done, reference),
            "nights": done,
            "environment": run_environment(),
        },
    )
    partial.unlink(missing_ok=True)
    print(json.dumps(summary(done, reference), indent=1))


if __name__ == "__main__":
    main()
