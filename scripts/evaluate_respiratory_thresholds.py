"""Compromis sensibilité/fausses fenêtres positives, seuils choisis sur validation."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_respiratory_osf import align_subject, load_osf_partition, validated_inputs
from scripts.respiratory_baseline import binary_metrics, load_partition, select_threshold
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.respiratory_events import MIN_OVERLAP_S, TARGET_EVENTS
from src.models.probe_export import run_environment

MODELS = ("sleepfm", "osf")
TARGETS = (0.7, 0.8, 0.9)
F1_THRESHOLD_TOLERANCE = 2e-6


def validate_probabilities(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if (
        y.ndim != 1
        or set(np.unique(y).tolist()) != {0, 1}
        or probability.shape != y.shape
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Deux classes et probabilités finies alignées requises")
    return y, probability


def sensitivity_threshold(y, probability, target) -> float:
    """Plus grand seuil atteignant la sensibilité demandée, ex æquo inclus."""
    y, probability = validate_probabilities(y, probability)
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Cible de sensibilité dans ]0, 1] requise")
    positives = np.sort(probability[y == 1])
    required = int(np.ceil(target * len(positives)))
    return float(positives[-required])


def operating_metrics(y, probability, threshold, hypopnea=None) -> dict:
    y, probability = validate_probabilities(y, probability)
    prediction = probability >= threshold
    positive, negative = y == 1, y == 0
    tp, fn = int((positive & prediction).sum()), int((positive & ~prediction).sum())
    fp, tn = int((negative & prediction).sum()), int((negative & ~prediction).sum())
    result = {
        "windows": len(y),
        "positive": tp + fn,
        "negative": tn + fp,
        "predicted_positive": tp + fp,
        "false_positive_windows": fp,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if hypopnea is not None:
        hypopnea = np.asarray(hypopnea)
        if hypopnea.shape != y.shape or hypopnea.dtype.kind != "b" or (hypopnea & negative).any():
            raise ValueError("Groupe hypopnée invalide")
        den = int(hypopnea.sum())
        num = int((hypopnea & prediction).sum())
        result["hypopnea"] = {"windows": den, "true_positive": num, "recall": num / den if den else None}
    return result


def operating_intervals(y, probability, threshold, subjects, hypopnea, n_boot=500) -> dict:
    y, probability = validate_probabilities(y, probability)
    subjects, hypopnea = np.asarray(subjects), np.asarray(hypopnea)
    if subjects.shape != y.shape or hypopnea.shape != y.shape or n_boot <= 0:
        raise ValueError("Groupes et nombre de tirages invalides")
    unique, inverse = np.unique(subjects, return_inverse=True)
    p, positive, negative = probability >= threshold, y == 1, y == 0
    masks = (negative & ~p, negative & p, positive & ~p, positive & p, hypopnea & p, hypopnea)
    counts = np.stack([np.bincount(inverse, weights=m, minlength=len(unique)) for m in masks], axis=1)
    draws = np.random.default_rng(42).integers(len(unique), size=(n_boot, len(unique)))
    tn, fp, fn, tp, htp, hn = counts[draws].sum(axis=1).T
    values = {
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "precision": (tp, tp + fp),
        "hypopnea_recall": (htp, hn),
    }
    intervals = {}
    for name, (num, den) in values.items():
        valid = den > 0
        intervals[name] = {
            "ci95": np.percentile(num[valid] / den[valid], [2.5, 97.5]).tolist() if valid.any() else None,
            "valid_draws": int(valid.sum()),
        }
    return {"metrics": intervals, "n_boot": n_boot, "seed": 42, "n_subjects": len(unique), "unit": "subject"}


def checked_artifact(report_path: Path, report: dict, key: str) -> Path:
    info = report["artifacts"][key]
    path = (report_path.parent / info["file"]).resolve()
    if path.parent != report_path.parent.resolve() or file_sha256(path) != info["sha256"]:
        raise ValueError(f"Artefact modifié ou hors dossier : {key}")
    return path


def load_sources(args):
    baseline, labels, osf_contract, _ = validated_inputs(args)
    comparison = json.loads(args.comparison.read_text())
    errors = json.loads(args.errors.read_text())
    if (
        comparison.get("status") != "terminé"
        or errors.get("status") != "terminé"
        or comparison["baseline_sha256"] != file_sha256(args.baseline)
        or errors["baseline_sha256"] != file_sha256(args.baseline)
        or errors["comparison_sha256"] != file_sha256(args.comparison)
        or comparison["label_manifest_sha256"] != baseline["label_manifest_sha256"]
        or comparison["partitions"] != baseline["partitions"]
        or comparison["osf_manifest"] != osf_contract
    ):
        raise ValueError("Sources ou partitions différentes des références")
    paths = {k: checked_artifact(args.comparison, comparison, k) for k in ("predictions.npz", "inputs.json", "joblib")}
    groups_path = checked_artifact(args.errors, errors, "groups.npz")
    with np.load(paths["predictions.npz"]) as f:
        test = {k: f[k] for k in f.files}
    with np.load(groups_path) as f:
        groups = {k: f[k] for k in f.files}
    with np.load(args.baseline.with_suffix(".predictions.npz")) as f:
        for key in ("y_true", "subject", "epoch_idx", "onset_s"):
            if not np.array_equal(test[key], f[key]) or not np.array_equal(test[key], groups[key]):
                raise ValueError("Fenêtres ou groupes de test non identiques")
        if not np.array_equal(test["probability_sleepfm"], f["probability"]):
            raise ValueError("Probabilités SleepFM différentes de la référence")
    n = len(test["y_true"])
    if any(a.shape != (n,) for a in test.values()) or set(test["subject"]) != set(baseline["partitions"]["test"]):
        raise ValueError("Couverture du test différente de la référence")
    if len(set(zip(test["subject"], test["epoch_idx"], strict=True))) != n:
        raise ValueError("Fenêtres de test en double")
    for kind in MODELS:
        validate_probabilities(test["y_true"], test[f"probability_{kind}"])
        old = binary_metrics(test["y_true"], test[f"probability_{kind}"], comparison["thresholds"][kind])
        for key in ("sensitivity", "specificity", "precision", "f1"):
            if abs(old[key] - comparison["metrics"][kind][key]) > 1e-12:
                raise ValueError("Scores de référence non retrouvés")
        hypo = groups["event_group"] == "resp_hypopnea"
        old_hypo = operating_metrics(test["y_true"], test[f"probability_{kind}"], comparison["thresholds"][kind], hypo)[
            "hypopnea"
        ]
        expected = errors["by_type"]["resp_hypopnea"]
        if (
            old_hypo["windows"] != expected["positive_windows"]
            or abs(old_hypo["recall"] - expected["models"][kind]["recall"]) > 1e-12
        ):
            raise ValueError("Groupe ou rappel hypopnée de référence non retrouvé")
    return baseline, labels, comparison, test, groups, paths


def validation_probabilities(args, baseline, comparison, paths):
    inventory = json.loads(paths["inputs.json"].read_text())
    selected = baseline["partitions"]["validation"]
    verified = {}
    for subject in selected:
        row = inventory[subject]
        for path, key in [
            (args.sleepfm_dir / f"{subject}.npz", "sleepfm_sha256"),
            (args.osf_dir / f"{subject}.npz", "osf_sha256"),
            (args.labels_dir / f"{subject}.npz", "labels_sha256"),
        ]:
            if file_sha256(path) != row[key]:
                raise ValueError(f"Cache ou labels de validation modifiés : {subject}")
        with (
            np.load(args.sleepfm_dir / f"{subject}.npz") as sleepfm,
            np.load(args.osf_dir / f"{subject}.npz") as osf,
            np.load(args.labels_dir / f"{subject}.npz") as target,
        ):
            align_subject(target, sleepfm, osf)
        verified[subject] = row
    output, checks = {}, {}
    for kind in MODELS:
        model_path = args.baseline.with_suffix(".joblib") if kind == "sleepfm" else paths["joblib"]
        expected = baseline["model_sha256"] if kind == "sleepfm" else comparison["artifacts"]["joblib"]["sha256"]
        if file_sha256(model_path) != expected:
            raise ValueError("Modèle modifié avant rechargement")
        bundle = joblib.load(model_path)
        contract = baseline["embeddings_manifest"] if kind == "sleepfm" else comparison["osf_manifest"]
        model = bundle["model"]
        if (
            bundle["contract"] != contract
            or bundle["threshold"] != comparison["thresholds"][kind]
            or tuple(bundle["target_events"]) != TARGET_EVENTS
            or bundle["min_overlap_s"] != MIN_OVERLAP_S
            or not np.array_equal(model.classes_, [0, 1])
        ):
            raise ValueError("Contrat du modèle ou seuil initial différent")
        loader = load_partition if kind == "sleepfm" else load_osf_partition
        directory = args.sleepfm_dir if kind == "sleepfm" else args.osf_dir
        X, y, groups = loader(selected, directory, args.labels_dir)
        if not np.isfinite(X).all() or X.shape[1] != model.n_features_in_:
            raise ValueError("Embeddings de validation invalides")
        probability = model.predict_proba(X)[:, 1]
        del X
        validate_probabilities(y, probability)
        if kind == "sleepfm":
            output.update(y_true=y, subject=groups[0], epoch_idx=groups[1], onset_s=groups[2])
        elif not np.array_equal(y, output["y_true"]) or any(
            not np.array_equal(g, output[k]) for k, g in zip(("subject", "epoch_idx", "onset_s"), groups, strict=True)
        ):
            raise ValueError("Validation non appariée")
        recomputed = select_threshold(y, probability)
        difference = abs(recomputed - bundle["threshold"])
        if difference > F1_THRESHOLD_TOLERANCE:
            raise ValueError("Seuil F1 de validation non reproduit dans la tolérance fixée")
        output[f"probability_{kind}"] = probability
        checks[kind] = {
            "model_sha256": expected,
            "initial_threshold": bundle["threshold"],
            "recomputed_f1_threshold": recomputed,
            "absolute_difference": difference,
            "tolerance": F1_THRESHOLD_TOLERANCE,
        }
        print(f"Validation {kind} : {len(y)} fenêtres, seuil initial vérifié", flush=True)
    return output, checks, verified


def run(args):
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "validation.npz", "inputs.json", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucune référence écrasée")
    for directory in (args.sleepfm_dir, args.osf_dir, args.labels_dir):
        if args.out.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Sortie dans un dossier d'entrée refusée")
    plan_digest = file_sha256(args.plan)
    source_paths = {"baseline": args.baseline, "comparison": args.comparison, "errors": args.errors}
    source_digests = {key: file_sha256(path) for key, path in source_paths.items()}
    baseline, _, comparison, test, groups, paths = load_sources(args)
    validation, checks, inventory = validation_probabilities(args, baseline, comparison, paths)
    # Tous les nouveaux seuils sont arrêtés sur la validation avant leurs métriques de test.
    thresholds = {
        kind: {
            f"sensitivity_{round(target * 100)}": sensitivity_threshold(
                validation["y_true"], validation[f"probability_{kind}"], target
            )
            for target in TARGETS
        }
        for kind in MODELS
    }
    for kind in MODELS:
        thresholds[kind] = {"initial_f1": comparison["thresholds"][kind], **thresholds[kind]}
    results = {}
    hypopnea = groups["event_group"] == "resp_hypopnea"
    if (hypopnea & (test["y_true"] != 1)).any():
        raise ValueError("Groupe hypopnée non positif")
    for kind in MODELS:
        results[kind] = {}
        for name, value in thresholds[kind].items():
            metrics = operating_metrics(test["y_true"], test[f"probability_{kind}"], value, hypopnea)
            results[kind][name] = {
                "threshold": value,
                "validation": operating_metrics(validation["y_true"], validation[f"probability_{kind}"], value),
                "test": metrics,
                "ci95": operating_intervals(
                    test["y_true"], test[f"probability_{kind}"], value, test["subject"], hypopnea
                ),
            }
    report = {
        "status": "terminé",
        "kind": "compromis respiratoire, seuils choisis sur validation",
        "plan_sha256": plan_digest,
        "targets": list(TARGETS),
        "source_sha256": source_digests,
        "partitions": baseline["partitions"],
        "reload_checks": checks,
        "source_artifacts": comparison["artifacts"],
        "results": results,
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Exploratoire par fenêtre, pas événements ni diagnostic ; IC conditionnels aux seuils figés",
    }
    if file_sha256(args.plan) != plan_digest or any(
        file_sha256(path) != source_digests[key] for key, path in source_paths.items()
    ):
        raise ValueError("Plan ou référence modifié pendant le calcul")
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["validation.npz"], **validation)
    write_json_atomic(outputs["inputs.json"], inventory)
    code = [
        "scripts/evaluate_respiratory_thresholds.py",
        "scripts/respiratory_baseline.py",
        "scripts/compare_respiratory_osf.py",
        "src/data/respiratory_events.py",
    ]
    report["code_files"] = {p: file_sha256(Path(p)) for p in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in code:
            archive.write(p, p)
        archive.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(f"Terminé : seuils de validation conservés, {len(test['y_true'])} fenêtres de test évaluées.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/respiratory_baseline_physionet2018.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("results/respiratory_osf_vs_sleepfm_physionet2018.json")
    )
    parser.add_argument("--errors", type=Path, default=Path("results/respiratory_errors_physionet2018.json"))
    parser.add_argument("--sleepfm-dir", type=Path, default=Path("embeddings/physionet2018_v2"))
    parser.add_argument("--osf-dir", type=Path, default=Path("embeddings/physionet2018_osf"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/respiratory_labels"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan seuils respiratoires SleepFM OSF 12-11.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Évaluation arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
