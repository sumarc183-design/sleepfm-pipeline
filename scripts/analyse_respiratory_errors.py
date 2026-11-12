"""Erreurs respiratoires par type, stade et nuit, à modèles et seuils inchangés."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.extract_physionet2018 import import_signal_names
from src.data.respiratory_events import TARGET_EVENTS, event_intervals, window_targets
from src.data.subject_ids import validate_subject_ids
from src.models.probe_export import run_environment

STAGES = ("wake", "nonrem1", "nonrem2", "nonrem3", "rem")
GROUPS = (*TARGET_EVENTS, "plusieurs_types")
SEED, N_BOOT = 42, 1000


def subtype_groups(y: np.ndarray, overlap_by_type: np.ndarray) -> np.ndarray:
    """Un groupe exclusif par fenêtre, fondé sur les types réellement présents."""
    y, coverage = np.asarray(y), np.asarray(overlap_by_type)
    if (
        y.ndim != 1
        or coverage.shape != (len(y), len(TARGET_EVENTS))
        or not np.isin(y, [-1, 0, 1]).all()
        or not np.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 30).any()
    ):
        raise ValueError("Labels ou couvertures par type invalides")
    present = coverage > 0
    count = present.sum(axis=1)
    if ((y == 1) & (count == 0)).any() or ((y == 0) & (count != 0)).any():
        raise ValueError("Types d'événements incompatibles avec les labels")
    groups = np.full(len(y), "exclu", dtype="U30")
    groups[y == 0] = "aucun"
    for i, kind in enumerate(TARGET_EVENTS):
        groups[(y == 1) & (count == 1) & present[:, i]] = kind
    groups[(y == 1) & (count > 1)] = "plusieurs_types"
    return groups


def recall_intervals(y, predictions, subjects, group_mask, n_boot=N_BOOT) -> dict:
    """Même tirage de sujets pour les deux modèles ; nuits sans groupe conservées."""
    positive = (np.asarray(y) == 1) & group_mask
    unique, inverse = np.unique(subjects, return_inverse=True)
    denominator = np.bincount(inverse, weights=positive, minlength=len(unique))
    numerators = {
        kind: np.bincount(inverse, weights=positive & pred, minlength=len(unique)) for kind, pred in predictions.items()
    }
    draws = np.random.default_rng(SEED).integers(len(unique), size=(n_boot, len(unique)))
    sampled_denominator = denominator[draws].sum(axis=1)
    valid = sampled_denominator > 0
    sampled = {
        kind: numerator[draws].sum(axis=1)[valid] / sampled_denominator[valid] for kind, numerator in numerators.items()
    }
    if valid.any():
        individual = {kind: np.percentile(values, [2.5, 97.5]).tolist() for kind, values in sampled.items()}
        difference = np.percentile(sampled["osf"] - sampled["sleepfm"], [2.5, 97.5]).tolist()
    else:
        individual = {kind: None for kind in predictions}
        difference = None
    return {
        "recall": individual,
        "difference_osf_minus_sleepfm": difference,
        "n_boot": n_boot,
        "valid_draws": int(valid.sum()),
        "n_subjects": len(unique),
        "seed": SEED,
    }


def summary(y, probability, threshold) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    prediction = probability >= threshold
    positive, negative = y == 1, y == 0
    tp, fn = int((positive & prediction).sum()), int((positive & ~prediction).sum())
    fp, tn = int((negative & prediction).sum()), int((negative & ~prediction).sum())
    return {
        "windows": len(y),
        "positive": tp + fn,
        "negative": tn + fp,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "average_precision": float(average_precision_score(y, probability))
        if positive.any() and negative.any()
        else None,
    }


def annotations_by_type(record: Path, onsets) -> tuple[np.ndarray, np.ndarray]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Lecteur WFDB absent : voir requirements-respiratory.txt") from exc
    _, frequency, n_samples = import_signal_names(str(record.with_suffix(".hea")))
    annotation = wfdb.rdann(str(record), "arousal")
    if annotation.fs is not None and annotation.fs != frequency:
        raise ValueError("Fréquence d'annotation différente de l'en-tête")
    intervals, _ = event_intervals(annotation.sample, annotation.aux_note, n_samples)
    y, _ = window_targets(onsets, intervals, frequency, n_samples)
    coverage = []
    for kind in TARGET_EVENTS:
        keep = np.array([note in (f"({kind}", f"{kind})") for note in annotation.aux_note])
        intervals, _ = event_intervals(annotation.sample[keep], np.asarray(annotation.aux_note)[keep], n_samples)
        _, seconds = window_targets(onsets, intervals, frequency, n_samples)
        coverage.append(seconds)
    return y, np.column_stack(coverage)


def load_inputs(args) -> tuple[dict, dict, dict, dict]:
    report = json.loads(args.comparison.read_text())
    if report.get("status") != "terminé":
        raise ValueError("Comparaison non terminée")
    baseline = json.loads(args.baseline.read_text())
    if file_sha256(args.baseline) != report["baseline_sha256"]:
        raise ValueError("Baseline différente de la comparaison")
    inventory = json.loads((args.labels_dir / "manifest.json").read_text())
    if file_sha256(args.labels_dir / "manifest.json") != report["label_manifest_sha256"]:
        raise ValueError("Labels différents de la comparaison")
    archive = args.comparison.with_suffix(".predictions.npz")
    if file_sha256(archive) != report["artifacts"]["predictions.npz"]["sha256"]:
        raise ValueError("Archive de comparaison modifiée")
    with np.load(archive) as data:
        arrays = {key: data[key] for key in data.files}
    n = len(arrays["y_true"])
    if not n or not np.isin(arrays["y_true"], [0, 1]).all():
        raise ValueError("Labels binaires invalides")
    for key in ("y_true", "subject", "epoch_idx", "onset_s", "probability_sleepfm", "probability_osf"):
        if arrays[key].shape != (n,):
            raise ValueError(f"Vecteur {key} invalide")
    unique = sorted(set(arrays["subject"].tolist()))
    validate_subject_ids(unique)
    if set(unique) != set(report["partitions"]["test"]):
        raise ValueError("Sujets différents du test")
    if len(set(zip(arrays["subject"], arrays["epoch_idx"], strict=True))) != n:
        raise ValueError("Couples sujet/époque en double")
    for kind in ("sleepfm", "osf"):
        p = arrays[f"probability_{kind}"]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Probabilités invalides")
    return report, baseline, inventory, arrays


def run(args) -> dict:
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "groups.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucun résultat n'est écrasé")
    for root in (args.sleepfm_dir, args.labels_dir, args.annotations_dir):
        if args.out.resolve().is_relative_to(root.resolve()):
            raise ValueError("Sortie dans un dossier d'entrée refusée")
    reference, baseline, inventory, arrays = load_inputs(args)
    subjects, y = arrays["subject"], arrays["y_true"]
    stages, groups = np.full(len(y), -1, dtype=np.int64), np.full(len(y), "", dtype="U30")
    annotation_inventory = {}
    for i, subject in enumerate(sorted(set(subjects))):
        source = inventory["sources"][subject]
        record = args.annotations_dir / subject / subject
        for path, expected in [
            (record.with_suffix(".hea"), source["header_sha256"]),
            (record.with_suffix(".arousal"), source["annotation_sha256"]),
            (args.sleepfm_dir / f"{subject}.npz", source["embedding_sha256"]),
            (args.labels_dir / f"{subject}.npz", source["labels_sha256"]),
        ]:
            if file_sha256(path) != expected:
                raise ValueError(f"Entrée modifiée : {path}")
        with (
            np.load(args.labels_dir / f"{subject}.npz") as target,
            np.load(args.sleepfm_dir / f"{subject}.npz") as cache,
        ):
            if not all(np.array_equal(target[key], cache[key]) for key in ("epoch_idx", "onset_s")):
                raise ValueError("Labels non alignés avec le cache")
            reconstructed, coverage = annotations_by_type(record, target["onset_s"])
            if not np.array_equal(reconstructed, target["y"]):
                raise ValueError("Labels reconstruits différents")
            labels = subtype_groups(reconstructed, coverage)
            rows = np.flatnonzero(subjects == subject)
            lookup = {int(value): k for k, value in enumerate(target["epoch_idx"])}
            order = np.array([lookup[int(value)] for value in arrays["epoch_idx"][rows]], dtype=np.int64)
            if (
                len(order) != int((target["y"] >= 0).sum())
                or not np.array_equal(y[rows], reconstructed[order])
                or not np.array_equal(arrays["onset_s"][rows], target["onset_s"][order])
            ):
                raise ValueError("Fenêtres différentes du test respiratoire")
            stages[rows], groups[rows] = cache["y"][order], labels[order]
        annotation_inventory[subject] = {
            "header_sha256": source["header_sha256"],
            "annotation_sha256": source["annotation_sha256"],
        }
        if (i + 1) % 25 == 0:
            print(f"Annotations et stades alignés : {i + 1}/100 nuits", flush=True)
    if not np.isin(stages, range(len(STAGES))).all():
        raise ValueError("Stades inconnus")
    predictions = {kind: arrays[f"probability_{kind}"] >= reference["thresholds"][kind] for kind in ("sleepfm", "osf")}

    def both(mask):
        return {
            kind: summary(y[mask], arrays[f"probability_{kind}"][mask], reference["thresholds"][kind])
            for kind in predictions
        }

    by_type = {}
    for kind in GROUPS:
        mask = groups == kind
        counts = both(mask)
        by_type[kind] = {
            "positive_windows": int(mask.sum()),
            "nights": len(set(subjects[mask].tolist())),
            "models": {
                model: {key: values[key] for key in ("true_positive", "false_negative", "recall")}
                for model, values in counts.items()
            },
            "ci95": recall_intervals(y, predictions, subjects, mask),
        }
    if sum(row["positive_windows"] for row in by_type.values()) != int(y.sum()):
        raise ValueError("Les groupes ne couvrent pas exactement les fenêtres positives")
    by_stage = {
        stage: {"nights": len(set(subjects[stages == i].tolist())), "models": both(stages == i)}
        for i, stage in enumerate(STAGES)
    }
    by_night = {s: both(subjects == s) for s in sorted(set(subjects))}
    eligible = [s for s, values in by_night.items() if values["osf"]["positive"] >= 10]
    worst = sorted(eligible, key=lambda s: (by_night[s]["osf"]["recall"], s))[:5]
    report = {
        "status": "terminé",
        "kind": "analyse descriptive des erreurs respiratoires",
        "plan_sha256": file_sha256(args.plan),
        "comparison_sha256": file_sha256(args.comparison),
        "baseline_sha256": file_sha256(args.baseline),
        "environment": run_environment(),
        "command": sys.argv,
        "thresholds": reference["thresholds"],
        "nights": len(by_night),
        "windows": len(y),
        "positive_windows": int(y.sum()),
        "by_type": by_type,
        "by_stage": by_stage,
        "by_night": by_night,
        "worst_osf_recall_at_least_10_positives": worst,
        "annotation_inventory": annotation_inventory,
        "global": both(np.ones(len(y), dtype=bool)),
        "caveat": "Exploratoire, par fenêtre ; associations descriptives, ni causes ni diagnostics",
    }
    for kind in predictions:
        if abs(report["global"][kind]["recall"] - reference["metrics"][kind]["sensitivity"]) > 1e-12:
            raise ValueError("Rappel global différent de la référence")
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    partial = outputs["groups.npz"].with_name(f".{outputs['groups.npz'].stem}.tmp.npz")
    np.savez_compressed(
        partial,
        subject=subjects,
        epoch_idx=arrays["epoch_idx"],
        onset_s=arrays["onset_s"],
        stage=stages,
        event_group=groups,
        y_true=y,
    )
    os.replace(partial, outputs["groups.npz"])
    code = ["scripts/analyse_respiratory_errors.py", "src/data/respiratory_events.py"]
    report["code_files"] = {name: file_sha256(Path(name)) for name in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*map(Path, code), args.plan]:
            archive.write(path, str(path))
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(f"{len(by_night)} nuits, {len(y)} fenêtres, {int(y.sum())} positives ; groupes exclusifs vérifiés.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path, default=Path("results/respiratory_osf_vs_sleepfm_physionet2018.json")
    )
    parser.add_argument("--baseline", type=Path, default=Path("results/respiratory_baseline_physionet2018.json"))
    parser.add_argument("--sleepfm-dir", type=Path, default=Path("embeddings/physionet2018_v2"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/respiratory_labels"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/plans/Plan analyse erreurs respiratoires SleepFM OSF 12-11.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Analyse arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
