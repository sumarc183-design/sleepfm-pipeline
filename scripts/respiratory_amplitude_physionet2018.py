"""Amplitude des sangles et du débit sur les 360 nuits d'apprentissage respiratoires (descriptif).

Plan : docs/plans/Plan amplitude sangles débit PhysioNet 13-11.md. Valeurs brutes du .mat,
telles que SleepFM les reçoit. Téléchargement sujet par sujet (le suivant pendant le calcul),
fichiers supprimés après lecture. Aucun label, aucun modèle.

    python scripts/respiratory_amplitude_physionet2018.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_physionet2018 import download_subject  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.extract_physionet2018 import import_signal_names  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

CHANNELS = ("ABD", "CHEST", "AIRFLOW", "SaO2")
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/Plan amplitude sangles débit PhysioNet 13-11.md"


def night_stats(signal: np.ndarray) -> dict:
    x = np.asarray(signal, dtype=np.float64)
    median = float(np.median(x))
    return {
        "median": median,
        "robust_scale": float(1.4826 * np.median(np.abs(x - median))),
        "std": float(x.std()),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
        "constant_fraction": float(np.mean(np.diff(x) == 0)) if x.size > 1 else 1.0,
    }


def subject_stats(subject_dir: Path, channels=CHANNELS) -> dict:
    subject = subject_dir.name
    names = list(import_signal_names(str(subject_dir / f"{subject}.hea"))[0])
    values = loadmat(subject_dir / f"{subject}.mat")["val"]
    if values.shape[0] != len(names):
        raise ValueError(f"{subject} : {values.shape[0]} signaux pour {len(names)} noms")
    return {channel: night_stats(values[names.index(channel)]) for channel in channels}


def summary(nights: dict, channels=CHANNELS) -> dict:
    out = {}
    for channel in channels:
        scale = np.array([n[channel]["robust_scale"] for n in nights.values()])
        median = np.array([n[channel]["median"] for n in nights.values()])
        p5, q1, q2, q3, p95 = np.percentile(scale, [5, 25, 50, 75, 95])
        out[channel] = {
            "robust_scale": {"p5": p5, "q1": q1, "median": q2, "q3": q3, "p95": p95},
            "p95_over_p5": float(p95 / p5) if p5 > 0 else float("inf"),
            "flat_nights": int((scale == 0).sum()),
            "median_of_medians": float(np.median(median)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/respiratory_baseline_physionet2018.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/respiratory_amplitude_physionet2018.json")
    parser.add_argument("--limit", type=int, default=None, help="essai court")
    parser.add_argument("--channels", default=",".join(CHANNELS), help="voies, séparées par des virgules")
    parser.add_argument("--plan", type=Path, default=PLAN, help="plan fixé avant calcul (empreinte enregistrée)")
    args = parser.parse_args()
    channels = tuple(args.channels.split(","))
    subjects = sorted(json.loads(args.baseline.read_text())["partitions"]["fit"])[: args.limit]
    partial_path = args.out.with_suffix(".partial.json")
    nights = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    todo = [s for s in subjects if s not in nights]
    print(f"{len(nights)} nuit(s) déjà lues, {len(todo)} à traiter")

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
                nights[subject] = subject_stats(Path(tmp) / subject, channels)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            write_json_atomic(partial_path, nights)
            if (index + 1) % 20 == 0 or index + 1 == len(todo):
                print(f"[{index + 1}/{len(todo)}] {subject}")

    write_json_atomic(
        args.out,
        {
            "plan_sha256": file_sha256(args.plan),
            "plan": args.plan.name,
            "channels": list(channels),
            "baseline_sha256": file_sha256(args.baseline),
            "subjects_sha256": hashlib.sha256(json.dumps(subjects).encode()).hexdigest(),
            "n_nights": len(nights),
            "summary": summary({s: nights[s] for s in subjects}, channels),
            "nights": {s: nights[s] for s in subjects},
            "environment": run_environment(),
        },
    )
    partial_path.unlink(missing_ok=True)
    print(json.dumps(summary({s: nights[s] for s in subjects}, channels), indent=1))


if __name__ == "__main__":
    main()
