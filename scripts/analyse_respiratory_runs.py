"""Regroupement descriptif de fenêtres respiratoires, sans nouvelle règle d'alerte."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.subject_ids import validate_subject_ids
from src.models.probe_export import run_environment

WINDOW_S = 30.0
TIME_TOLERANCE_S = 0.01
MODELS = ("sleepfm", "osf")
POINTS = ("initial_f1", "sensitivity_70", "sensitivity_80", "sensitivity_90")


def check_rows(subject, epoch_idx, onset_s):
    subject, epoch_idx, onset_s = map(np.asarray, (subject, epoch_idx, onset_s))
    if (
        subject.ndim != 1
        or not len(subject)
        or epoch_idx.shape != subject.shape
        or onset_s.shape != subject.shape
        or not np.issubdtype(epoch_idx.dtype, np.integer)
        or (epoch_idx < 0).any()
        or not np.isfinite(onset_s).all()
        or (onset_s < 0).any()
        or len(set(zip(subject, epoch_idx, strict=True))) != len(subject)
    ):
        raise ValueError("Identités ou temps de fenêtres invalides")
    for sid in np.unique(subject):
        rows = np.flatnonzero(subject == sid)
        if (np.diff(epoch_idx[rows]) <= 0).any() or (np.diff(onset_s[rows]) < WINDOW_S - TIME_TOLERANCE_S).any():
            raise ValueError("Chronologie ou chevauchement de fenêtres invalide")
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("Les lignes d'une nuit doivent être regroupées")


def consecutive_runs(mask, subject, epoch_idx, onset_s):
    """Bornes de lignes [début, fin[ ; les lacunes et fenêtres exclues coupent."""
    mask = np.asarray(mask)
    if mask.shape != np.asarray(subject).shape or mask.dtype.kind != "b":
        raise ValueError("Masque de fenêtres booléen requis")
    selected = np.flatnonzero(mask)
    if not len(selected):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    before, after = selected[:-1], selected[1:]
    linked = (
        (after == before + 1)
        & (subject[after] == subject[before])
        & (epoch_idx[after] == epoch_idx[before] + 1)
        & np.isclose(onset_s[after] - onset_s[before], WINDOW_S, atol=TIME_TOLERANCE_S, rtol=0)
    )
    starts = selected[np.r_[True, ~linked]]
    ends = selected[np.r_[~linked, True]] + 1
    if int((ends - starts).sum()) != int(mask.sum()):
        raise ValueError("Couverture des séquences invalide")
    return starts, ends


def duration_summary(lengths):
    seconds = np.asarray(lengths) * WINDOW_S
    return {
        "count": len(seconds),
        "single_window": int((seconds == WINDOW_S).sum()),
        "single_window_fraction": float((seconds == WINDOW_S).mean()) if len(seconds) else None,
        "median_s": float(np.median(seconds)) if len(seconds) else None,
        "p90_s": float(np.percentile(seconds, 90)) if len(seconds) else None,
        "max_s": float(seconds.max()) if len(seconds) else None,
        "nominal_minutes": float(seconds.sum() / 60),
    }


def rate_interval(counts, hours, n_boot=500):
    counts, hours = np.asarray(counts), np.asarray(hours)
    if (
        counts.ndim != 1
        or not len(counts)
        or hours.shape != counts.shape
        or n_boot <= 0
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.isfinite(hours).all()
        or (hours <= 0).any()
    ):
        raise ValueError("Comptes et expositions valides requis")
    draws = np.random.default_rng(42).integers(len(counts), size=(n_boot, len(counts)))
    values = counts[draws].sum(axis=1) / hours[draws].sum(axis=1)
    return {
        "value": float(counts.sum() / hours.sum()),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        "n_boot": n_boot,
        "seed": 42,
        "n_subjects": len(counts),
    }


def analyse_configuration(y, probability, threshold, subject, epoch_idx, onset_s):
    y, probability, subject, epoch_idx, onset_s = map(np.asarray, (y, probability, subject, epoch_idx, onset_s))
    if (
        y.shape != subject.shape
        or probability.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Labels, probabilités ou seuil invalides")
    check_rows(subject, epoch_idx, onset_s)
    prediction = probability >= threshold
    false_positive = prediction & (y == 0)
    starts, ends = consecutive_runs(prediction, subject, epoch_idx, onset_s)
    fp_starts, fp_ends = consecutive_runs(false_positive, subject, epoch_idx, onset_s)
    cumulative = np.r_[0, np.cumsum(y)]
    positive = cumulative[ends] - cumulative[starts]
    lengths = ends - starts
    classification = np.where(positive == 0, "negative_only", np.where(positive == lengths, "positive_only", "mixed"))
    unique, inverse = np.unique(subject, return_inverse=True)
    hours = np.bincount(inverse, minlength=len(unique)) * WINDOW_S / 3600
    count_runs = np.bincount(inverse[fp_starts], minlength=len(unique))
    count_pred = np.bincount(inverse[starts], minlength=len(unique))
    fp_windows = np.bincount(inverse, weights=false_positive, minlength=len(unique)).astype(np.int64)
    if int((fp_ends - fp_starts).sum()) != int(false_positive.sum()):
        raise ValueError("Les séquences ne retrouvent pas toutes les fausses fenêtres positives")
    by_night = {
        str(s): {
            "evaluated_hours": float(hours[i]),
            "false_positive_runs": int(count_runs[i]),
            "predicted_positive_runs": int(count_pred[i]),
            "false_positive_windows": int(fp_windows[i]),
            "false_positive_minutes": float(fp_windows[i] * WINDOW_S / 60),
            "false_positive_runs_per_evaluated_hour": float(count_runs[i] / hours[i]),
        }
        for i, s in enumerate(unique)
    }
    matrix = [
        [int(((y == 0) & ~prediction).sum()), int(false_positive.sum())],
        [int(((y == 1) & ~prediction).sum()), int(((y == 1) & prediction).sum())],
    ]
    summary = {
        "threshold": float(threshold),
        "confusion_matrix": matrix,
        "predicted_positive_windows": int(prediction.sum()),
        "false_positive_windows": int(false_positive.sum()),
        "predicted_positive_runs": {
            "count": len(starts),
            **{k: int((classification == k).sum()) for k in ("negative_only", "positive_only", "mixed")},
        },
        "false_positive_runs": duration_summary(fp_ends - fp_starts),
        "evaluated_hours": float(hours.sum()),
        "false_positive_runs_per_evaluated_hour": rate_interval(count_runs, hours),
        "per_night_false_positive_runs": {
            "median": float(np.median(count_runs)),
            "q25": float(np.percentile(count_runs, 25)),
            "q75": float(np.percentile(count_runs, 75)),
            "min": int(count_runs.min()),
            "max": int(count_runs.max()),
        },
        "by_night": by_night,
    }
    intervals = []
    for kind, a, b, labels, tp in [
        ("predicted_positive", starts, ends, classification, positive),
        (
            "false_positive",
            fp_starts,
            fp_ends,
            np.full(len(fp_starts), "negative_only"),
            np.zeros(len(fp_starts), dtype=np.int64),
        ),
    ]:
        for j, (start, end) in enumerate(zip(a, b, strict=True)):
            intervals.append(
                {
                    "kind": kind,
                    "subject": str(subject[start]),
                    "classification": str(labels[j]),
                    "start_epoch_idx": int(epoch_idx[start]),
                    "end_epoch_idx": int(epoch_idx[end - 1]),
                    "start_s": float(onset_s[start]),
                    "end_s": float(onset_s[end - 1] + WINDOW_S),
                    "windows": int(end - start),
                    "true_positive_windows": int(tp[j]),
                    "false_positive_windows": int(end - start - tp[j]),
                }
            )
    return summary, intervals


def run(args):
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "intervals.npz", "ordering.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucun résultat écrasé")
    sources = {"thresholds": args.thresholds, "comparison": args.comparison, "plan": args.plan}
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    reference = json.loads(args.thresholds.read_text())
    comparison = json.loads(args.comparison.read_text())
    if (
        reference.get("status") != "terminé"
        or comparison.get("status") != "terminé"
        or reference["source_sha256"]["comparison"] != hashes["comparison"]
        or reference["partitions"] != comparison["partitions"]
    ):
        raise ValueError("Références ou partitions non concordantes")
    info = comparison["artifacts"]["predictions.npz"]
    archive = (args.comparison.parent / info["file"]).resolve()
    if archive.parent != args.comparison.parent.resolve() or file_sha256(archive) != info["sha256"]:
        raise ValueError("Archive de prédictions modifiée ou hors dossier")
    with np.load(archive) as data:
        arrays = {
            k: data[k] for k in ("y_true", "subject", "epoch_idx", "onset_s", "probability_sleepfm", "probability_osf")
        }
    n = len(arrays["subject"])
    if any(v.shape != (n,) for v in arrays.values()):
        raise ValueError("Vecteurs de test non alignés")
    subjects = validate_subject_ids(np.unique(arrays["subject"]).tolist())
    if set(subjects) != set(reference["partitions"]["test"]):
        raise ValueError("Nuits différentes du test de référence")
    order = np.lexsort((arrays["epoch_idx"], arrays["subject"]))
    reordered_nights = sum(
        bool((np.diff(arrays["epoch_idx"][arrays["subject"] == subject]) <= 0).any()) for subject in subjects
    )
    arrays = {key: values[order] for key, values in arrays.items()}
    check_rows(arrays["subject"], arrays["epoch_idx"], arrays["onset_s"])
    summaries, intervals = {}, []
    for model in MODELS:
        if set(reference["results"][model]) != set(POINTS):
            raise ValueError("Configurations de seuils différentes de la référence")
        summaries[model] = {}
        for point in POINTS:
            row = reference["results"][model][point]
            summary, records = analyse_configuration(
                arrays["y_true"],
                arrays[f"probability_{model}"],
                row["threshold"],
                arrays["subject"],
                arrays["epoch_idx"],
                arrays["onset_s"],
            )
            if (
                summary["confusion_matrix"] != row["test"]["confusion_matrix"]
                or summary["false_positive_windows"] != row["test"]["false_positive_windows"]
                or summary["predicted_positive_windows"] != row["test"]["predicted_positive"]
            ):
                raise ValueError("Les comptes ne retrouvent pas l'analyse des seuils")
            summaries[model][point] = summary
            intervals.extend({"model": model, "point": point, **record} for record in records)
        print(f"Séquences {model} : quatre configurations contrôlées", flush=True)
    if any(file_sha256(path) != hashes[name] for name, path in sources.items()):
        raise ValueError("Référence ou plan modifié pendant le calcul")
    report = {
        "status": "terminé",
        "kind": "regroupement descriptif de fenêtres respiratoires",
        "source_sha256": hashes,
        "prediction_sha256": info["sha256"],
        "nights": len(subjects),
        "windows": n,
        "window_s": WINDOW_S,
        "time_tolerance_s": TIME_TOLERANCE_S,
        "ordering": {
            "rule": "subject, integer epoch_idx",
            "rows_moved": int((order != np.arange(n)).sum()),
            "nights_reordered": reordered_nights,
        },
        "results": summaries,
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Séquences de fenêtres, pas événements ou alertes cliniques ; exposition = fenêtres évaluées",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "model",
        "point",
        "kind",
        "subject",
        "classification",
        "start_epoch_idx",
        "end_epoch_idx",
        "start_s",
        "end_s",
        "windows",
        "true_positive_windows",
        "false_positive_windows",
    )
    strings = {"model", "point", "kind", "subject", "classification"}
    columns = {
        k: np.asarray(
            [r[k] for r in intervals], dtype=str if k in strings else float if k in {"start_s", "end_s"} else np.int64
        )
        for k in keys
    }
    np.savez_compressed(outputs["intervals.npz"], **columns)
    np.savez_compressed(
        outputs["ordering.npz"],
        original_row=order,
        subject=arrays["subject"],
        epoch_idx=arrays["epoch_idx"],
        onset_s=arrays["onset_s"],
    )
    code = Path(__file__).resolve()
    report["code_sha256"] = file_sha256(code)
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(code, "scripts/analyse_respiratory_runs.py")
        z.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(f"Terminé : {n} fenêtres, {len(subjects)} nuits, {len(intervals)} séquences archivées.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, default=Path("results/respiratory_thresholds_physionet2018.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("results/respiratory_osf_vs_sleepfm_physionet2018.json")
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan séquences respiratoires SleepFM OSF 12-11 v2.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Analyse arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
