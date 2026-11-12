"""Incertitude de sélection des seuils sur validation, sans relire le test."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.subject_ids import validate_subject_ids
from src.models.probe_export import run_environment
from src.models.threshold_resampling import (
    POINTS,
    TARGETS,
    binary_counts,
    distribution,
    metrics_from_counts,
    weighted_thresholds,
)

MODELS = ("sleepfm", "osf")
N_BOOT, SEED = 500, 42


def validation_inputs(source_path):
    reference = json.loads(source_path.read_text())
    if reference.get("status") != "terminé":
        raise ValueError("Référence non terminée")
    parts = reference["partitions"]
    if any(set(parts[a]) & set(parts[b]) for a, b in [("fit", "validation"), ("fit", "test"), ("validation", "test")]):
        raise ValueError("Partitions non disjointes")
    info = reference["artifacts"]["validation.npz"]
    archive = (source_path.parent / info["file"]).resolve()
    if archive.parent != source_path.parent.resolve() or file_sha256(archive) != info["sha256"]:
        raise ValueError("Archive de validation modifiée ou hors dossier")
    with np.load(archive) as f:
        arrays = {
            key: f[key]
            for key in ("y_true", "subject", "epoch_idx", "onset_s", "probability_sleepfm", "probability_osf")
        }
    n = len(arrays["y_true"])
    if not n or any(a.shape != (n,) for a in arrays.values()):
        raise ValueError("Vecteurs non alignés")
    subjects = sorted(validate_subject_ids(np.unique(arrays["subject"]).tolist()))
    if set(subjects) != set(parts["validation"]) or len(subjects) < 2:
        raise ValueError("Nuits différentes de la validation")
    if (
        not np.issubdtype(arrays["epoch_idx"].dtype, np.integer)
        or (arrays["epoch_idx"] < 0).any()
        or len(set(zip(arrays["subject"], arrays["epoch_idx"], strict=True))) != n
        or not np.isfinite(arrays["onset_s"]).all()
        or (arrays["onset_s"] < 0).any()
    ):
        raise ValueError("Identités ou temps invalides")
    for model in MODELS:
        if set(reference["results"][model]) != set(POINTS):
            raise ValueError("Règles de seuil différentes")
        selected = weighted_thresholds(arrays["y_true"], arrays[f"probability_{model}"], np.ones(n, dtype=int))
        for point in POINTS:
            row = reference["results"][model][point]
            if abs(selected[point] - row["threshold"]) > 1e-12:
                raise ValueError("Seuil complet de validation non retrouvé")
            cm = binary_counts(arrays["y_true"], arrays[f"probability_{model}"], row["threshold"])
            if cm.tolist() != row["validation"]["confusion_matrix"]:
                raise ValueError("Matrice de validation non retrouvée")
    return reference, arrays, subjects, info["sha256"]


def bootstrap_diagnostic(y, probabilities, subject, subjects, n_boot=N_BOOT):
    if n_boot <= 0:
        raise ValueError("Nombre de tirages positif requis")
    lookup = {sid: i for i, sid in enumerate(subjects)}
    inverse = np.array([lookup[sid] for sid in subject])
    rng = np.random.default_rng(SEED)
    counts = np.stack(
        [np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    )
    reasons = np.full(n_boot, "", dtype="U30")
    threshold = {model: np.full((n_boot, len(POINTS)), np.nan) for model in MODELS}
    calibration_cm = {model: np.zeros((n_boot, len(POINTS), 2, 2), dtype=np.int64) for model in MODELS}
    oob_cm = {model: np.zeros((n_boot, len(POINTS), 2, 2), dtype=np.int64) for model in MODELS}
    for b, multiplicity in enumerate(counts):
        weights = multiplicity[inverse]
        oob = weights == 0
        if not oob.any():
            reasons[b] = "no_oob"
            continue
        if set(np.unique(y[weights > 0]).tolist()) != {0, 1}:
            reasons[b] = "calibration_one_class"
            continue
        if set(np.unique(y[oob]).tolist()) != {0, 1}:
            reasons[b] = "oob_one_class"
            continue
        for model in MODELS:
            p = probabilities[model]
            selected = weighted_thresholds(y, p, weights)
            for j, point in enumerate(POINTS):
                value = selected[point]
                threshold[model][b, j] = value
                calibration_cm[model][b, j] = binary_counts(y, p, value, weights)
                oob_cm[model][b, j] = binary_counts(y[oob], p[oob], value)
                if (
                    point in TARGETS
                    and metrics_from_counts(calibration_cm[model][b, j])["sensitivity"] + 1e-12 < TARGETS[point]
                ):
                    raise ValueError("Cible non atteinte dans la calibration bootstrap")
        if (b + 1) % 100 == 0:
            print(f"Bootstrap : {b + 1}/{n_boot}", flush=True)
    valid = reasons == ""
    if not valid.any():
        raise ValueError("Aucun tirage OOB évaluable")
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for j, point in enumerate(POINTS):
            metrics = [metrics_from_counts(cm) for cm in oob_cm[model][valid, j]]
            entry = {
                "threshold": distribution(threshold[model][valid, j]),
                "oob": {
                    name: distribution([np.nan if m[name] is None else m[name] for m in metrics])
                    for name in ("sensitivity", "specificity", "precision", "f1")
                },
            }
            if point in TARGETS:
                entry["oob_fraction_below_target"] = float(
                    np.mean([m["sensitivity"] < TARGETS[point] for m in metrics])
                )
            summary[model][point] = entry
    saved = {"subject_multiplicities": counts, "skip_reason": reasons, "subjects": np.asarray(subjects)}
    for model in MODELS:
        saved[f"threshold_{model}"] = threshold[model]
        saved[f"calibration_cm_{model}"] = calibration_cm[model]
        saved[f"oob_cm_{model}"] = oob_cm[model]
    details = {
        "n_boot": n_boot,
        "seed": SEED,
        "valid_draws": int(valid.sum()),
        "skipped": dict(Counter(reasons[~valid].tolist())),
        "oob_nights": distribution((counts[valid] == 0).sum(axis=1)),
        "unique_calibration_nights": distribution((counts[valid] > 0).sum(axis=1)),
    }
    return summary, saved, details


def leave_one_night_out(y, probabilities, subject, subjects):
    predictions = {model: np.zeros((len(y), len(POINTS)), dtype=bool) for model in MODELS}
    thresholds = {model: np.zeros((len(subjects), len(POINTS))) for model in MODELS}
    assigned = np.zeros(len(y), dtype=int)
    for i, sid in enumerate(subjects):
        heldout = subject == sid
        assigned[heldout] += 1
        weights = (~heldout).astype(int)
        for model in MODELS:
            selected = weighted_thresholds(y, probabilities[model], weights)
            for j, point in enumerate(POINTS):
                thresholds[model][i, j] = selected[point]
                predictions[model][heldout, j] = probabilities[model][heldout] >= selected[point]
    if not (assigned == 1).all():
        raise ValueError("Chaque fenêtre doit avoir une seule prédiction laissée de côté")
    results = {model: {} for model in MODELS}
    for model in MODELS:
        for j, point in enumerate(POINTS):
            code = 2 * y.astype(int) + predictions[model][:, j]
            cm = np.bincount(code, minlength=4).reshape(2, 2)
            results[model][point] = {
                "metrics": metrics_from_counts(cm),
                "thresholds": distribution(thresholds[model][:, j]),
            }
    saved = {"threshold_subjects": np.asarray(subjects)}
    for model in MODELS:
        saved[f"prediction_{model}"] = predictions[model]
        saved[f"threshold_{model}"] = thresholds[model]
    return results, saved


def run(args):
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "bootstrap.npz", "leave_one_out.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucune référence écrasée")
    fingerprints = {"source": file_sha256(args.source), "plan": file_sha256(args.plan)}
    reference, arrays, subjects, validation_sha = validation_inputs(args.source)
    probabilities = {model: arrays[f"probability_{model}"] for model in MODELS}
    boot, samples, details = bootstrap_diagnostic(arrays["y_true"], probabilities, arrays["subject"], subjects)
    loo, heldout = leave_one_night_out(arrays["y_true"], probabilities, arrays["subject"], subjects)
    if file_sha256(args.source) != fingerprints["source"] or file_sha256(args.plan) != fingerprints["plan"]:
        raise ValueError("Référence ou plan modifié pendant le calcul")
    report = {
        "status": "terminé",
        "kind": "incertitude du choix des seuils sur validation seulement",
        "source_sha256": fingerprints,
        "validation_archive_sha256": validation_sha,
        "subjects": subjects,
        "windows": len(arrays["y_true"]),
        "point_order": list(POINTS),
        "bootstrap": details,
        "bootstrap_results": boot,
        "leave_one_night_out": loo,
        "reference_thresholds": {
            model: {point: reference["results"][model][point]["threshold"] for point in POINTS} for model in MODELS
        },
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Distribution OOB de développement ; seuils variables, modèles figés, aucun test relu",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["bootstrap.npz"], **samples)
    np.savez_compressed(
        outputs["leave_one_out.npz"], **heldout, **{k: arrays[k] for k in ("y_true", "subject", "epoch_idx", "onset_s")}
    )
    code = ["scripts/analyse_threshold_stability.py", "src/models/threshold_resampling.py"]
    report["code_files"] = {p: file_sha256(Path(p)) for p in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in code:
            z.write(p, p)
        z.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(
        f"Terminé : {len(subjects)} nuits, {details['valid_draws']} tirages OOB valides, validation croisée complète.",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/respiratory_thresholds_physionet2018.json"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/plans/Plan stabilité seuils respiratoires SleepFM OSF 12-11.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Analyse arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
