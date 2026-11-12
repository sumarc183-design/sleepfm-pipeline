"""Des fenêtres à l'IAH ≥ 15 par nuit, avec les modèles et seuils respiratoires figés.

Calibration sur les 40 nuits de validation, un seul passage sur les 100 nuits de test.
Aucun entraînement, aucun nouveau seuil, aucune donnée de confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyse_respiratory_runs import consecutive_runs
from scripts.compare_respiratory_osf import align_subject
from src.data.ahi import (
    AHI_CUTOFF,
    MIN_SLEEP_H,
    WINDOW_S,
    auc,
    bland_altman,
    bootstrap_draws,
    fit_line,
    linear_weighted_kappa,
    percentile_ci,
    positive_rate,
    reference_ahi,
    sensitivity_specificity,
    severity,
    spearman,
)
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.extract_physionet2018 import import_signal_names
from src.data.respiratory_event_eval import covered_duration, individual_events
from src.data.respiratory_events import TARGET_EVENTS, event_intervals
from src.models.probe_export import run_environment

MODELS = ("sleepfm", "osf")
N_BOOT, SEED = 1000, 42
PROBABILITY_TOLERANCE = 1e-5
REFERENCE_TOLERANCE = 0.01
MIN_EVENT_OVERLAP_S = 10.0
CODE = [
    "scripts/estimate_respiratory_ahi.py",
    "src/data/ahi.py",
    "src/data/respiratory_event_eval.py",
    "src/data/respiratory_events.py",
    "scripts/analyse_respiratory_runs.py",
    "scripts/compare_respiratory_osf.py",
]


def load_sources(args) -> dict:
    """Modèles, seuils, partitions et manifestes, chacun contrôlé par son empreinte."""
    baseline = json.loads(args.baseline.read_text())
    comparison = json.loads(args.comparison.read_text())
    if file_sha256(args.baseline.with_suffix(".joblib")) != baseline["model_sha256"]:
        raise ValueError("Modèle SleepFM différent de l'archive")
    artifacts = comparison["artifacts"]
    folder = args.comparison.parent
    for key in ("joblib", "inputs.json", "predictions.npz"):
        if file_sha256(folder / artifacts[key]["file"]) != artifacts[key]["sha256"]:
            raise ValueError(f"Archive OSF modifiée : {key}")
    if comparison["baseline_sha256"] != file_sha256(args.baseline):
        raise ValueError("Comparaison OSF non rattachée à cette baseline")
    if file_sha256(args.baseline.with_suffix(".predictions.npz")) != baseline["predictions_sha256"]:
        raise ValueError("Prédictions SleepFM modifiées")
    if file_sha256(args.labels_dir / "manifest.json") != baseline["label_manifest_sha256"]:
        raise ValueError("Manifeste des labels modifié")
    if baseline["partitions"] != comparison["partitions"]:
        raise ValueError("Partitions différentes entre SleepFM et OSF")
    models = {
        "sleepfm": joblib.load(args.baseline.with_suffix(".joblib")),
        "osf": joblib.load(folder / artifacts["joblib"]["file"]),
    }
    thresholds = {m: float(models[m]["threshold"]) for m in MODELS}
    if thresholds != {m: float(comparison["thresholds"][m]) for m in MODELS}:
        raise ValueError("Seuils différents des résultats publiés")
    return {
        "models": {m: models[m]["model"] for m in MODELS},
        "thresholds": thresholds,
        "partitions": {k: sorted(baseline["partitions"][k]) for k in ("validation", "test")},
        "labels": json.loads((args.labels_dir / "manifest.json").read_text())["sources"],
        "osf_inputs": json.loads((folder / artifacts["inputs.json"]["file"]).read_text()),
        "archives": {
            "test": folder / artifacts["predictions.npz"]["file"],
            "validation": args.validation_archive,
        },
    }


def archived_probabilities(path: Path) -> dict:
    with np.load(path) as data:
        keys = list(zip(data["subject"].tolist(), data["epoch_idx"].tolist(), strict=True))
        return {m: dict(zip(keys, data[f"probability_{m}"].tolist(), strict=True)) for m in MODELS}


def night_windows(subject: str, args, sources: dict) -> dict:
    """Toutes les fenêtres en cache de la nuit, triées par indice d'époque."""
    label_path = args.labels_dir / f"{subject}.npz"
    sleepfm_path = args.sleepfm_dir / f"{subject}.npz"
    osf_path = args.osf_dir / f"{subject}.npz"
    expected = sources["labels"][subject]
    if (
        file_sha256(label_path) != expected["labels_sha256"]
        or file_sha256(sleepfm_path) != expected["embedding_sha256"]
    ):
        raise ValueError(f"Cache SleepFM ou labels modifié : {subject}")
    if file_sha256(osf_path) != sources["osf_inputs"][subject]["osf_sha256"]:
        raise ValueError(f"Cache OSF modifié : {subject}")
    with np.load(label_path) as target, np.load(sleepfm_path) as sleepfm, np.load(osf_path) as osf:
        target, sleepfm, osf = dict(target), dict(sleepfm), dict(osf)
    order = align_subject(target, sleepfm, osf)
    chrono = np.argsort(target["epoch_idx"], kind="stable")
    return {
        "epoch_idx": target["epoch_idx"][chrono],
        "onset_s": target["onset_s"][chrono],
        "y": target["y"][chrono],
        "X_sleepfm": sleepfm["X"][chrono],
        "X_osf": osf["X"][order][chrono],
    }


def night_annotations(subject: str, args, sources: dict, wfdb) -> dict:
    record = args.annotations_dir / subject / subject
    expected = sources["labels"][subject]
    for path, key in (
        (record.with_suffix(".hea"), "header_sha256"),
        (record.with_suffix(".arousal"), "annotation_sha256"),
    ):
        if file_sha256(path) != expected[key]:
            raise ValueError(f"Annotation modifiée : {subject}")
    _, rate, n_samples = import_signal_names(str(record.with_suffix(".hea")))
    annotation = wfdb.rdann(str(record), "arousal")
    fs = annotation.fs or rate
    if fs != rate:
        raise ValueError("Fréquence d'annotation différente de l'en-tête")
    reference = reference_ahi(annotation.sample, annotation.aux_note, fs)
    events = individual_events(annotation.sample, annotation.aux_note, n_samples)
    union, _ = event_intervals(annotation.sample, annotation.aux_note, n_samples)
    return {"reference": reference, "events": events, "union": union, "rate": rate}


def read_reference_csv(path: Path) -> dict:
    with path.open(newline="") as handle:
        return {row["sujet"]: (float(row["sommeil_h"]), float(row["iah"])) for row in csv.DictReader(handle)}


def event_detection(windows: dict, positive: np.ndarray, notes: dict) -> dict:
    """Rappel par type (≥ 10 s couvertes) et séquences prédites sans aucun événement."""
    rate = notes["rate"]
    begins = np.rint(windows["onset_s"] * rate).astype(np.int64)
    spans = np.column_stack((begins, begins + int(WINDOW_S * rate))) / rate
    events = notes["events"]
    found = covered_duration(events[:, :2] / rate, spans[positive]) >= MIN_EVENT_OVERLAP_S - 1e-9
    subject = np.zeros(len(positive), dtype="U1")
    starts, ends = consecutive_runs(positive, subject, windows["epoch_idx"], windows["onset_s"])
    runs = np.column_stack((spans[starts, 0], spans[ends - 1, 1])) if len(starts) else np.empty((0, 2))
    union = notes["union"] / rate
    touched = covered_duration(runs, union) > 0 if len(runs) and len(union) else np.zeros(len(runs), dtype=bool)
    return {
        "events": {kind: int((events[:, 2] == k).sum()) for k, kind in enumerate(TARGET_EVENTS)},
        "found": {kind: int(found[events[:, 2] == k].sum()) for k, kind in enumerate(TARGET_EVENTS)},
        "runs": len(runs),
        "false_runs": int((~touched).sum()),
    }


def classification(estimated, reference) -> dict:
    truth, predicted = reference >= AHI_CUTOFF, estimated >= AHI_CUTOFF
    sens, spec = sensitivity_specificity(predicted, truth)
    return {"sensitivity": sens, "specificity": spec}


def evaluate_test(night: dict, calibration: dict, eligible: np.ndarray) -> tuple[dict, dict]:
    reference = night["reference_ahi"][eligible]
    truth = reference >= AHI_CUTOFF
    results, estimates = {}, {}
    for model in MODELS:
        x = night[f"rate_{model}"][eligible]
        a, b = calibration[model]["rate"]["a"], calibration[model]["rate"]["b"]
        estimated = a + b * x
        sa, sb = calibration[model]["probability_sum"]["a"], calibration[model]["probability_sum"]["b"]
        secondary = sa + sb * night[f"probability_sum_{model}"][eligible]
        estimates[model] = estimated
        results[model] = {
            "primary": classification(estimated, reference),
            "secondary_probability_sum": classification(secondary, reference),
            "auc_rate": auc(x, truth),
            "mae": float(np.abs(estimated - reference).mean()),
            "spearman": spearman(estimated, reference),
            "bland_altman": bland_altman(estimated, reference),
            "weighted_kappa_linear": linear_weighted_kappa(severity(estimated), severity(reference)),
            "confusion_15": {
                "true_positive": int(((estimated >= AHI_CUTOFF) & truth).sum()),
                "false_negative": int(((estimated < AHI_CUTOFF) & truth).sum()),
                "false_positive": int(((estimated >= AHI_CUTOFF) & ~truth).sum()),
                "true_negative": int(((estimated < AHI_CUTOFF) & ~truth).sum()),
            },
        }
    return results, estimates


def bootstrap(night: dict, estimates: dict, eligible: np.ndarray) -> dict:
    reference = night["reference_ahi"][eligible]
    truth = reference >= AHI_CUTOFF
    rates = {m: night[f"rate_{m}"][eligible] for m in MODELS}
    draws = bootstrap_draws(len(reference), N_BOOT, SEED)
    collected = {m: {"sensitivity": [], "specificity": [], "auc_rate": []} for m in MODELS}
    differences, skipped = [], 0
    for rows in draws:
        sample_truth = truth[rows]
        if sample_truth.all() or not sample_truth.any():
            skipped += 1
            continue
        aucs = {}
        for model in MODELS:
            sens, spec = sensitivity_specificity(estimates[model][rows] >= AHI_CUTOFF, sample_truth)
            aucs[model] = auc(rates[model][rows], sample_truth)
            collected[model]["sensitivity"].append(sens)
            collected[model]["specificity"].append(spec)
            collected[model]["auc_rate"].append(aucs[model])
        differences.append(aucs["osf"] - aucs["sleepfm"])
    return {
        "draws": N_BOOT,
        "seed": SEED,
        "skipped_draws": skipped,
        "ci95": {m: {k: percentile_ci(v) for k, v in collected[m].items()} for m in MODELS},
        "auc_difference_osf_minus_sleepfm_ci95": percentile_ci(differences),
    }


def run(args) -> dict:
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "nights.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucune référence écrasée")
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Lecteur WFDB absent : voir requirements-respiratory.txt") from exc
    inputs = {
        "plan": args.plan,
        "baseline": args.baseline,
        "comparison": args.comparison,
        "validation_archive": args.validation_archive,
        "reference_csv": args.reference_csv,
    }
    fingerprints = {k: file_sha256(p) for k, p in inputs.items()}
    sources = load_sources(args)
    archives = {part: archived_probabilities(path) for part, path in sources["archives"].items()}
    reference_csv = read_reference_csv(args.reference_csv)

    rows = {key: [] for key in ("subject", "partition", "sleep_h", "reference_ahi", "windows")}
    for model in MODELS:
        rows[f"rate_{model}"], rows[f"probability_sum_{model}"] = [], []
    detections = {m: [] for m in MODELS}
    checked, max_gap = 0, 0.0
    for partition in ("validation", "test"):
        for subject in sources["partitions"][partition]:
            windows = night_windows(subject, args, sources)
            notes = night_annotations(subject, args, sources, wfdb)
            sleep_csv, ahi_csv = reference_csv[subject]
            ref = notes["reference"]
            if abs(ref["sleep_h"] - sleep_csv) > REFERENCE_TOLERANCE or (
                np.isfinite(ref["ahi"]) and abs(ref["ahi"] - ahi_csv) > REFERENCE_TOLERANCE
            ):
                raise ValueError(f"IAH de référence différent de l'étude de faisabilité : {subject}")
            rows["subject"].append(subject)
            rows["partition"].append(partition)
            rows["sleep_h"].append(ref["sleep_h"])
            rows["reference_ahi"].append(ref["ahi"])
            rows["windows"].append(len(windows["epoch_idx"]))
            for model in MODELS:
                probability = sources["models"][model].predict_proba(windows[f"X_{model}"])[:, 1]
                archive = archives[partition][model]
                evaluated = windows["y"] >= 0
                expected = np.array([archive[(subject, int(i))] for i in windows["epoch_idx"][evaluated]])
                gap = float(np.abs(probability[evaluated] - expected).max())
                threshold = sources["thresholds"][model]
                crossed = int(((probability[evaluated] >= threshold) != (expected >= threshold)).sum())
                if gap > PROBABILITY_TOLERANCE or crossed:
                    raise ValueError(f"Probabilités {model} différentes des archives : {subject} ({gap:.2e})")
                max_gap, checked = max(max_gap, gap), checked + int(evaluated.sum())
                rate, probability_sum, _ = positive_rate(probability, sources["thresholds"][model])
                rows[f"rate_{model}"].append(rate)
                rows[f"probability_sum_{model}"].append(probability_sum)
                if partition == "test" and ref["sleep_h"] >= MIN_SLEEP_H:
                    detections[model].append(
                        event_detection(windows, probability >= sources["thresholds"][model], notes)
                    )
            print(f"{partition} {subject} : IAH {ref['ahi']:.1f}", flush=True)

    night = {k: np.asarray(v) for k, v in rows.items()}
    eligible = night["sleep_h"] >= MIN_SLEEP_H
    validation = eligible & (night["partition"] == "validation")
    test = eligible & (night["partition"] == "test")
    calibration = {}
    for model in MODELS:
        calibration[model] = {}
        for key, column in (("rate", f"rate_{model}"), ("probability_sum", f"probability_sum_{model}")):
            a, b = fit_line(night[column][validation], night["reference_ahi"][validation])
            calibration[model][key] = {"a": a, "b": b}
    results, estimates = evaluate_test(night, calibration, test)
    uncertainty = bootstrap(night, estimates, test)
    hours = float(night["windows"][test].sum() * WINDOW_S / 3600)
    events = {}
    for model in MODELS:
        total = {kind: sum(d["events"][kind] for d in detections[model]) for kind in TARGET_EVENTS}
        found = {kind: sum(d["found"][kind] for d in detections[model]) for kind in TARGET_EVENTS}
        false_runs = sum(d["false_runs"] for d in detections[model])
        events[model] = {
            "events": total,
            "found": found,
            "recall": {k: found[k] / total[k] if total[k] else None for k in TARGET_EVENTS},
            "recall_all": sum(found.values()) / sum(total.values()),
            "predicted_runs": sum(d["runs"] for d in detections[model]),
            "false_runs": false_runs,
            "false_runs_per_window_hour": false_runs / hours,
        }
    if any(file_sha256(p) != fingerprints[k] for k, p in inputs.items()):
        raise ValueError("Entrée modifiée pendant le calcul")
    for model in MODELS:
        night[f"estimated_ahi_{model}"] = (
            calibration[model]["rate"]["a"] + calibration[model]["rate"]["b"] * night[f"rate_{model}"]
        )
    report = {
        "status": "terminé",
        "kind": "IAH ≥ 15 par nuit, modèles respiratoires figés, exploratoire",
        "source_sha256": fingerprints,
        "thresholds": sources["thresholds"],
        "nights": {
            "validation": int((night["partition"] == "validation").sum()),
            "test": int((night["partition"] == "test").sum()),
            "validation_used": int(validation.sum()),
            "test_used": int(test.sum()),
            "excluded_short_sleep": night["subject"][~eligible].tolist(),
            "test_positive_15": int((night["reference_ahi"][test] >= AHI_CUTOFF).sum()),
        },
        "probability_control": {"windows_checked": checked, "max_abs_difference": max_gap},
        "reference_control": "IAH et sommeil identiques à iah_500_sujets_ouverts.csv à 0,01 près",
        "calibration": calibration,
        "test": results,
        "bootstrap": uncertainty,
        "events_test": events,
        "test_window_hours": hours,
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Nuits de test déjà explorées ; aucun critère de réussite ; même cohorte que le développement",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["nights.npz"], **night)
    report["code_files"] = {p: file_sha256(Path(p)) for p in CODE}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in CODE:
            z.write(p, p)
        z.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    for model in MODELS:
        primary = results[model]["primary"]
        print(f"{model} : sensibilité {primary['sensitivity']:.3f}, spécificité {primary['specificity']:.3f}")
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "docs/plans/Plan événements et IAH SleepFM OSF 12-11.md")
    parser.add_argument("--baseline", type=Path, default=Path("results/respiratory_baseline_physionet2018.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("results/respiratory_osf_vs_sleepfm_physionet2018.json")
    )
    parser.add_argument(
        "--validation-archive", type=Path, default=Path("results/respiratory_thresholds_physionet2018.validation.npz")
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=root.parent / "Faisabilite_apnee_donnees/iah_500_sujets_ouverts.csv",
        help="IAH de l'étude de faisabilité, contrôle de la référence",
    )
    parser.add_argument("--sleepfm-dir", type=Path, default=Path("embeddings/physionet2018_v2"))
    parser.add_argument("--osf-dir", type=Path, default=Path("embeddings/physionet2018_osf"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/respiratory_labels"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Analyse arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
