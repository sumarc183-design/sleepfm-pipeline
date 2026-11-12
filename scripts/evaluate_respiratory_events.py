"""Diagnostic événementiel sur les 40 nuits de validation, aux seuils existants."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyse_respiratory_runs import MODELS, POINTS, check_rows, consecutive_runs
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.extract_physionet2018 import import_signal_names
from src.data.respiratory_event_eval import covered_duration, individual_events, match_one_to_one, union_intervals
from src.data.respiratory_events import TARGET_EVENTS, event_intervals, window_targets
from src.models.probe_export import run_environment

MINIMUM_S, TOLERANCE_S = 10.0, 0.01


def event_scores(events, windows, prediction, starts, ends):
    """Couverture disponible, recouvrement positif et paires globales uniques."""
    duration = events[:, 1] - events[:, 0]
    available = covered_duration(events, windows)
    full = (available > 0) & (duration - available <= TOLERANCE_S)
    eligible = full & (duration + TOLERANCE_S >= MINIMUM_S)
    positive = covered_duration(events, windows[prediction])
    covered = eligible & (positive + TOLERANCE_S >= MINIMUM_S)
    selected = np.flatnonzero(eligible)
    overlaps = np.zeros((len(selected), len(starts)))
    run_duration = np.zeros(len(starts))
    for j, (start, end) in enumerate(zip(starts, ends, strict=True)):
        overlaps[:, j] = covered_duration(events[selected], windows[start:end])
        union = union_intervals(windows[start:end])
        run_duration[j] = (union[:, 1] - union[:, 0]).sum()
    pairs = match_one_to_one(overlaps, duration[selected], run_duration, MINIMUM_S, TOLERANCE_S)
    matched_run, matched_overlap = np.full(len(events), -1, dtype=np.int64), np.zeros(len(events))
    matched_run[selected] = pairs
    paired = np.flatnonzero(pairs >= 0)
    matched_overlap[selected[paired]] = overlaps[paired, pairs[paired]]
    if ((matched_run >= 0) & ~covered).any():
        raise ValueError("Appariement sans recouvrement positif suffisant")
    return {
        "available_s": available,
        "full": full,
        "eligible": eligible,
        "positive_s": positive,
        "covered": covered,
        "matched_run": matched_run,
        "matched_overlap_s": matched_overlap,
    }


def recall_summary(denominators, covered, matched):
    denominator, c, m = map(np.asarray, (denominators, covered, matched))
    total = int(denominator.sum())
    draws = np.random.default_rng(42).integers(len(denominator), size=(500, len(denominator)))
    sampled_den = denominator[draws].sum(1)
    valid = sampled_den > 0
    result = {
        "eligible_events": total,
        "bootstrap": {"n_boot": 500, "n_subjects": len(denominator), "seed": 42, "valid_draws": int(valid.sum())},
    }
    for name, numerator in [("positive_window_coverage", c), ("one_to_one_matching", m)]:
        values = numerator[draws].sum(1)[valid] / sampled_den[valid]
        result[name] = {
            "count": int(numerator.sum()),
            "recall": float(numerator.sum() / total) if total else None,
            "ci95": np.percentile(values, [2.5, 97.5]).tolist() if valid.any() else None,
        }
    return result


def source_inputs(args):
    reference = json.loads(args.thresholds.read_text())
    baseline = json.loads(args.baseline.read_text())
    if (
        reference.get("status") != "terminé"
        or baseline.get("status") != "terminé"
        or reference["source_sha256"]["baseline"] != file_sha256(args.baseline)
        or reference["partitions"] != baseline["partitions"]
        or file_sha256(args.labels_manifest) != baseline["label_manifest_sha256"]
    ):
        raise ValueError("Sources ou partitions différentes des références")
    parts = reference["partitions"]
    if any(set(parts[a]) & set(parts[b]) for a, b in [("fit", "validation"), ("fit", "test"), ("validation", "test")]):
        raise ValueError("Partitions non disjointes")
    info = reference["artifacts"]["validation.npz"]
    archive = (args.thresholds.parent / info["file"]).resolve()
    if archive.parent != args.thresholds.parent.resolve() or file_sha256(archive) != info["sha256"]:
        raise ValueError("Archive de validation modifiée ou hors dossier")
    with np.load(archive) as f:
        arrays = {
            key: f[key]
            for key in ("y_true", "subject", "epoch_idx", "onset_s", "probability_sleepfm", "probability_osf")
        }
    n = len(arrays["subject"])
    if any(a.shape != (n,) for a in arrays.values()) or set(arrays["subject"]) != set(parts["validation"]):
        raise ValueError("Fenêtres différentes de la validation")
    if not np.isin(arrays["y_true"], [0, 1]).all():
        raise ValueError("Labels binaires invalides")
    for model in MODELS:
        p = arrays[f"probability_{model}"]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Probabilités invalides")
        if set(reference["results"][model]) != set(POINTS):
            raise ValueError("Configurations différentes")
        for point in POINTS:
            row = reference["results"][model][point]
            if (
                confusion_matrix(arrays["y_true"], p >= row["threshold"], labels=[0, 1]).tolist()
                != row["validation"]["confusion_matrix"]
            ):
                raise ValueError("Matrice par fenêtre de validation non retrouvée")
    order = np.lexsort((arrays["epoch_idx"], arrays["subject"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    check_rows(arrays["subject"], arrays["epoch_idx"], arrays["onset_s"])
    return reference, json.loads(args.labels_manifest.read_text())["sources"], arrays, order, info["sha256"]


def run(args):
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "events.npz", "runs.npz", "ordering.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucune référence écrasée")
    if args.out.resolve().is_relative_to(args.annotations_dir.resolve()):
        raise ValueError("Sortie dans un dossier d'entrée refusée")
    inputs = {
        "thresholds": args.thresholds,
        "baseline": args.baseline,
        "labels_manifest": args.labels_manifest,
        "plan": args.plan,
    }
    fingerprints = {k: file_sha256(p) for k, p in inputs.items()}
    reference, sources, arrays, order, validation_sha = source_inputs(args)
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Lecteur WFDB absent : voir requirements-respiratory.txt") from exc
    subjects = sorted(set(arrays["subject"]))
    event_parts, records, daily, inventory = [], [], {}, {}
    offset = 0
    for i, subject in enumerate(subjects):
        rows = np.flatnonzero(arrays["subject"] == subject)
        record = args.annotations_dir / subject / subject
        source = sources[subject]
        for path, key in [
            (record.with_suffix(".hea"), "header_sha256"),
            (record.with_suffix(".arousal"), "annotation_sha256"),
        ]:
            if file_sha256(path) != source[key]:
                raise ValueError(f"Annotation modifiée : {subject}")
        _, rate, n_samples = import_signal_names(str(record.with_suffix(".hea")))
        annotation = wfdb.rdann(str(record), "arousal")
        if annotation.fs is not None and annotation.fs != rate:
            raise ValueError("Fréquence d'annotation différente de l'en-tête")
        raw = individual_events(annotation.sample, annotation.aux_note, n_samples)
        if any(int((raw[:, 2] == k).sum()) != source["event_counts"][kind] for k, kind in enumerate(TARGET_EVENTS)):
            raise ValueError("Événements individuels différents du manifeste")
        union, _ = event_intervals(annotation.sample, annotation.aux_note, n_samples)
        y, _ = window_targets(arrays["onset_s"][rows], union, rate, n_samples)
        if not np.array_equal(y, arrays["y_true"][rows]):
            raise ValueError("Labels respiratoires reconstruits différents")
        begins = np.rint(arrays["onset_s"][rows] * rate).astype(np.int64)
        windows = np.column_stack((begins, begins + 30 * rate)) / rate
        events = raw[:, :2] / rate
        duration = events[:, 1] - events[:, 0]
        available = covered_duration(events, windows)
        full = (available > 0) & (duration - available <= TOLERANCE_S)
        eligible = full & (duration + TOLERANCE_S >= MINIMUM_S)
        category = np.where(full, "complete", np.where(available > 0, "partial", "none"))
        part = {
            "subject": np.full(len(raw), subject),
            "start_sample": raw[:, 0],
            "end_sample": raw[:, 1],
            "event_type": np.asarray(TARGET_EVENTS)[raw[:, 2]],
            "sample_rate": np.full(len(raw), rate),
            "duration_s": duration,
            "available_s": available,
            "coverage": category,
            "eligible": eligible,
        }
        daily[subject] = {
            "total": len(raw),
            "complete": int(full.sum()),
            "partial": int((category == "partial").sum()),
            "none": int((category == "none").sum()),
            "too_short": int((duration + TOLERANCE_S < MINIMUM_S).sum()),
            "eligible": int(eligible.sum()),
            "configurations": {},
        }
        for model in MODELS:
            p = arrays[f"probability_{model}"][rows]
            for point in POINTS:
                key = f"{model}_{point}"
                threshold = reference["results"][model][point]["threshold"]
                prediction = p >= threshold
                starts, ends = consecutive_runs(
                    prediction, arrays["subject"][rows], arrays["epoch_idx"][rows], arrays["onset_s"][rows]
                )
                result = event_scores(events, windows, prediction, starts, ends)
                for field in ("positive_s", "matched_run", "matched_overlap_s"):
                    part[f"{field}_{key}"] = result[field]
                daily[subject]["configurations"][key] = {
                    "covered": int(result["covered"].sum()),
                    "matched": int((result["matched_run"] >= 0).sum()),
                    "runs": len(starts),
                }
                run_matches = np.full(len(starts), -1, dtype=np.int64)
                matched_events = np.flatnonzero(result["matched_run"] >= 0)
                run_matches[result["matched_run"][matched_events]] = offset + matched_events
                for j, (start, end) in enumerate(zip(starts, ends, strict=True)):
                    records.append(
                        {
                            "model": model,
                            "point": point,
                            "subject": subject,
                            "run_id": j,
                            "start_sorted_row": int(rows[start]),
                            "end_sorted_row_exclusive": int(rows[end - 1] + 1),
                            "start_sample": int(begins[start]),
                            "end_sample": int(begins[end - 1] + 30 * rate),
                            "matched_event_index": int(run_matches[j]),
                        }
                    )
        event_parts.append(part)
        offset += len(raw)
        inventory[subject] = {k: source[k] for k in ("header_sha256", "annotation_sha256")}
        print(f"Événements validation : {i + 1}/{len(subjects)} nuits", flush=True) if (i + 1) % 10 == 0 else None
    events = {key: np.concatenate([part[key] for part in event_parts]) for key in event_parts[0]}
    results = {}
    for model in MODELS:
        results[model] = {}
        for point in POINTS:
            key = f"{model}_{point}"
            results[model][point] = recall_summary(
                [daily[s]["eligible"] for s in subjects],
                [daily[s]["configurations"][key]["covered"] for s in subjects],
                [daily[s]["configurations"][key]["matched"] for s in subjects],
            )
            results[model][point]["threshold"] = reference["results"][model][point]["threshold"]
            results[model][point]["by_type"] = {
                kind: {
                    "eligible_events": int((events["eligible"] & (events["event_type"] == kind)).sum()),
                    "covered_events": int(
                        (
                            events["eligible"]
                            & (events["event_type"] == kind)
                            & (events[f"positive_s_{key}"] + TOLERANCE_S >= MINIMUM_S)
                        ).sum()
                    ),
                }
                for kind in TARGET_EVENTS
            }
    coverage = {
        key: sum(daily[s][key] for s in subjects)
        for key in ("total", "complete", "partial", "none", "too_short", "eligible")
    }
    if coverage["complete"] + coverage["partial"] + coverage["none"] != coverage["total"]:
        raise ValueError("Couverture des références incohérente")
    if any(file_sha256(p) != fingerprints[k] for k, p in inputs.items()):
        raise ValueError("Référence ou plan modifié pendant le calcul")
    report = {
        "status": "terminé",
        "kind": "diagnostic événementiel sur validation seulement",
        "source_sha256": fingerprints,
        "validation_archive_sha256": validation_sha,
        "subjects": subjects,
        "windows": len(order),
        "coverage": coverage,
        "minimum_overlap_s": MINIMUM_S,
        "tolerance_s": TOLERANCE_S,
        "results": results,
        "by_night": daily,
        "annotation_inventory": inventory,
        "environment": run_environment(),
        "wfdb_version": wfdb.__version__,
        "command": sys.argv,
        "caveat": "Validation déjà utilisée pour les seuils ; pas confirmation, précision événementielle ou IAH",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["events.npz"], **events)
    fields = (
        "model",
        "point",
        "subject",
        "run_id",
        "start_sorted_row",
        "end_sorted_row_exclusive",
        "start_sample",
        "end_sample",
        "matched_event_index",
    )
    np.savez_compressed(
        outputs["runs.npz"],
        **{
            key: np.asarray([r[key] for r in records], dtype=str if key in {"model", "point", "subject"} else np.int64)
            for key in fields
        },
    )
    np.savez_compressed(
        outputs["ordering.npz"],
        original_row=order,
        subject=arrays["subject"],
        epoch_idx=arrays["epoch_idx"],
        onset_s=arrays["onset_s"],
    )
    code = [
        "scripts/evaluate_respiratory_events.py",
        "src/data/respiratory_event_eval.py",
        "src/data/respiratory_events.py",
        "scripts/analyse_respiratory_runs.py",
    ]
    report["code_files"] = {p: file_sha256(Path(p)) for p in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in code:
            z.write(p, p)
        z.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(json.dumps(coverage, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, default=Path("results/respiratory_thresholds_physionet2018.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/respiratory_baseline_physionet2018.json"))
    parser.add_argument(
        "--labels-manifest", type=Path, default=Path("results/respiratory_baseline_physionet2018.labels.json")
    )
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/plans/Plan événements respiratoires validation SleepFM OSF 12-11.md",
        help="Plan fixé avant calcul ; sera conservé dans l'archive",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Évaluation arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
