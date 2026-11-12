"""Décrit les fenêtres à un échantillon minoritaire, depuis des prédictions figées."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL
from src.data.extract_physionet2018 import import_signal_names
from src.data.majority_grid_labels import labels_from_counts
from src.data.subject_ids import validate_subject_ids
from src.models.probe_export import run_environment

STAGES = tuple(STAGE_NAME_TO_LABEL)
PROBES = ("logistic", "sequence")
ISSUES = ("majority", "minority", "other")


def boundary_profiles(samples, sample_rate, y, categories) -> dict:
    """Reconstruit la référence puis localise l'unique échantillon minoritaire."""
    samples, y, categories = np.asarray(samples), np.asarray(y), np.asarray(categories)
    if (
        type(sample_rate) is not int
        or sample_rate <= 0
        or y.ndim != 1
        or categories.shape != y.shape
        or samples.shape != (len(y), sample_rate * 30)
        or not np.issubdtype(samples.dtype, np.integer)
        or not np.isin(samples, range(-1, 5)).all()
    ):
        raise ValueError("Grille d'échantillons invalide")
    counts = np.stack([(samples == label).sum(axis=1) for label in range(5)], axis=1)
    reconstructed = labels_from_counts(counts, (samples == -1).sum(axis=1), sample_rate)
    if not np.array_equal(y, reconstructed["y_majority"]) or not np.array_equal(categories, reconstructed["category"]):
        raise ValueError("Labels ou catégories différents du résultat archivé")
    minority = np.full(len(y), -1, dtype=np.int64)
    position = np.full(len(y), "na", dtype="U8")
    near = categories == "near_uniform"
    indices = (samples[near] != y[near, None]).argmax(axis=1)
    minority[near] = samples[near][np.arange(near.sum()), indices]
    position[near] = np.where(indices == 0, "first", np.where(indices == samples.shape[1] - 1, "last", "inside"))
    previous, following = np.full(len(y), -1, dtype=np.int64), np.full(len(y), -1, dtype=np.int64)
    previous[1:], following[:-1] = y[:-1], y[1:]
    return {"minority": minority, "position": position, "previous": previous, "following": following}


def read_samples(record: Path, n_windows: int) -> tuple[np.ndarray, int]:
    _, rate, n_samples = import_signal_names(str(record.with_suffix(".hea")))
    if n_windows * rate * 30 > n_samples:
        raise ValueError("Grille plus longue que l'enregistrement")
    labels, active = np.full(n_samples, -1, dtype=np.int8), np.zeros(n_samples, dtype=np.uint8)
    with h5py.File(record.parent / f"{record.name}-arousal.mat") as f:
        stages = f["data/sleep_stages"]
        if set(stages) != {*STAGE_NAME_TO_LABEL, "undefined"}:
            raise ValueError("Colonnes de stades inattendues")
        for name, dataset in stages.items():
            if dataset.shape != (1, n_samples):
                raise ValueError("Longueur d'annotation différente de l'en-tête")
            values = dataset[0]
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Annotation non binaire")
            active += values.astype(np.uint8)
            labels[values.astype(bool)] = STAGE_NAME_TO_LABEL.get(name, -1)
    if not np.all(active == 1):
        raise ValueError("Un seul stade actif requis par échantillon")
    return labels[: n_windows * rate * 30].reshape(n_windows, rate * 30), rate


def outcome_summary(y, minority, predictions, subjects, mask, bootstrap=False) -> dict:
    subjects = np.asarray(subjects)
    unique, inverse = np.unique(subjects, return_inverse=True)
    count = int(mask.sum())
    result = {"windows": count, "nights": int(len(np.unique(subjects[mask]))), "models": {}}
    if not count:
        return result
    draws = np.random.default_rng(42).integers(len(unique), size=(500, len(unique))) if bootstrap else None
    denominators = np.bincount(inverse, weights=mask, minlength=len(unique))
    for kind, pred in predictions.items():
        correct = pred == y
        neighbor = pred == minority
        issues = {"majority": correct, "minority": neighbor, "other": ~(correct | neighbor)}
        record = {}
        for name, issue in issues.items():
            n = int((mask & issue).sum())
            row = {"count": n, "rate": n / count}
            if draws is not None:
                numerator = np.bincount(inverse, weights=mask & issue, minlength=len(unique))
                den = denominators[draws].sum(axis=1)
                valid = den > 0
                values = numerator[draws].sum(axis=1)[valid] / den[valid]
                row["ci95"] = np.percentile(values, [2.5, 97.5]).tolist() if valid.any() else None
                row["valid_draws"] = int(valid.sum())
            record[name] = row
        assert sum(row["count"] for row in record.values()) == count
        result["models"][kind] = record
    return result


def run(args) -> dict:
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "groups.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : aucun résultat écrasé")
    if args.out.resolve().is_relative_to(args.annotations_dir.resolve()):
        raise ValueError("Sortie dans un dossier d'entrée refusée")
    if any(args.source.resolve() == p.resolve() for p in outputs.values()):
        raise ValueError("La source ne peut pas être écrasée")
    source_hash = file_sha256(args.source)
    reference = json.loads(args.source.read_text())
    if reference.get("status") != "terminé":
        raise ValueError("Résultat source non terminé")
    label_code = Path(__file__).resolve().parents[1] / "src/data/majority_grid_labels.py"
    if file_sha256(label_code) != reference["code_files"]["src/data/majority_grid_labels.py"]:
        raise ValueError("Le lecteur de labels a changé depuis l'évaluation source")
    artifacts = {}
    for name in ("predictions.npz", "inputs.json"):
        info = reference["artifacts"][name]
        path = (args.source.parent / info["file"]).resolve()
        if path.parent != args.source.parent.resolve() or file_sha256(path) != info["sha256"]:
            raise ValueError("Artefact source modifié ou hors dossier")
        artifacts[name] = path
    with np.load(artifacts["predictions.npz"]) as f:
        arrays = {
            k: f[k]
            for k in ("y_majority", "category", "subject", "epoch_idx", "onset_s", "pred_logistic", "pred_sequence")
        }
    n = len(arrays["subject"])
    if any(a.shape != (n,) for a in arrays.values()) or n != reference["windows"]:
        raise ValueError("Dimensions de la source invalides")
    if not np.isin(arrays["y_majority"], range(-1, 5)).all() or any(
        not np.isin(arrays[f"pred_{kind}"], range(5)).all() for kind in PROBES
    ):
        raise ValueError("Stade inconnu")
    subjects = sorted(validate_subject_ids(np.unique(arrays["subject"]).tolist()))
    inventory = json.loads(artifacts["inputs.json"].read_text())
    if len(subjects) != reference["nights"] or set(subjects) != set(inventory):
        raise ValueError("Nuits différentes de la source")
    profiles = {
        "minority": np.full(n, -1, dtype=np.int64),
        "position": np.full(n, "na", dtype="U8"),
        "previous": np.full(n, -1, dtype=np.int64),
        "following": np.full(n, -1, dtype=np.int64),
    }
    for i, subject in enumerate(subjects):
        record = args.annotations_dir / subject / subject
        for path, key in [
            (record.with_suffix(".hea"), "header_sha256"),
            (record.parent / f"{subject}-arousal.mat", "h5_sha256"),
        ]:
            if file_sha256(path) != inventory[subject][key]:
                raise ValueError(f"Annotation modifiée : {subject}")
        rows = np.flatnonzero(arrays["subject"] == subject)
        if not np.array_equal(arrays["epoch_idx"][rows], np.arange(len(rows))) or not np.array_equal(
            arrays["onset_s"][rows], np.arange(len(rows)) * 30.0
        ):
            raise ValueError("Grille fixe ou chronologie invalide")
        samples, rate = read_samples(record, len(rows))
        profile = boundary_profiles(samples, rate, arrays["y_majority"][rows], arrays["category"][rows])
        for key in profiles:
            profiles[key][rows] = profile[key]
        if (i + 1) % 25 == 0:
            print(f"Annotations reconstruites : {i + 1}/{len(subjects)} nuits", flush=True)
    y, category = arrays["y_majority"], arrays["category"]
    near, uniform = category == "near_uniform", category == "uniform"
    counts = {c: int((category == c).sum()) for c in reference["counts"]}
    if counts != reference["counts"]:
        raise ValueError("Effectifs différents de la source")
    predictions = {kind: arrays[f"pred_{kind}"] for kind in PROBES}
    for kind, pred in predictions.items():
        for group, mask in (("uniform", uniform), ("near_uniform", near), ("all_majority", y >= 0)):
            cm = confusion_matrix(y[mask], pred[mask], labels=range(5))
            if not np.array_equal(cm, reference["metrics"][kind][group]["confusion_matrix"]):
                raise ValueError("Matrice source non retrouvée")
    by_stage, pairs = {}, {}
    for label, stage in enumerate(STAGES):
        mask = near & (y == label)
        by_stage[stage] = {
            "uniform_windows": int((uniform & (y == label)).sum()),
            **outcome_summary(y, profiles["minority"], predictions, arrays["subject"], mask),
        }
        for other, other_stage in enumerate(STAGES):
            if other == label:
                continue
            group = outcome_summary(
                y, profiles["minority"], predictions, arrays["subject"], mask & (profiles["minority"] == other)
            )
            group["enough_for_description"] = group["windows"] >= 100 and group["nights"] >= 10
            pairs[f"{stage}/{other_stage}"] = group
    report = {
        "status": "terminé",
        "kind": "analyse descriptive des frontières de stade",
        "source_sha256": source_hash,
        "source_artifacts": reference["artifacts"],
        "plan_sha256": file_sha256(args.plan),
        "nights": len(subjects),
        "windows": n,
        "counts": counts,
        "position": {p: int((near & (profiles["position"] == p)).sum()) for p in ("first", "last", "inside")},
        "neighbor_match": {
            "previous": int((near & (profiles["minority"] == profiles["previous"])).sum()),
            "following": int((near & (profiles["minority"] == profiles["following"])).sum()),
        },
        "overall": outcome_summary(y, profiles["minority"], predictions, arrays["subject"], near, bootstrap=True),
        "by_stage": by_stage,
        "pairs_majority_minority": pairs,
        "bootstrap": {"n_boot": 500, "seed": 42, "unit": "subject", "n_subjects": len(subjects)},
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Descriptif sur nuits déjà examinées ; aucun signal relu, aucune cause clinique établie",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["groups.npz"], **arrays, **profiles)
    code = Path(__file__).resolve()
    report["code_sha256"] = file_sha256(code)
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(code, "scripts/analyse_stage_boundaries.py")
        archive.write(label_code, "src/data/majority_grid_labels.py")
        archive.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": path.name, "sha256": file_sha256(path)} for ext, path in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(json.dumps({k: report[k] for k in ("position", "neighbor_match", "overall")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/raw_grid_transitions_physionet2018.json"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan frontières de stade SleepFM 12-11.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Analyse arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
