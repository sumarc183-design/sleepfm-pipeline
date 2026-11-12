"""Comparaison exploratoire OSF/SleepFM sur les labels respiratoires déjà figés.

Mêmes fenêtres, partitions et réglages ; AP appariée par sujet. Ne réentraîne
pas SleepFM et ne lit pas les 494 sujets de confirmation. Voir le plan dédié.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.respiratory_baseline import binary_metrics, partition_subjects, select_threshold
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.respiratory_events import MIN_OVERLAP_S, TARGET_EVENTS
from src.models.probe_export import run_environment

N_BOOT = 1000
SEED = 42
OSF_DIM = 768
TIME_TOLERANCE_S = 0.01


def align_subject(target, sleepfm, osf) -> np.ndarray:
    """Ordre des lignes OSF dans l'ordre des labels ; identités contrôlées avant filtrage."""
    n = len(target["y"])
    if not n or target["y"].shape != (n,) or not np.isin(target["y"], [-1, 0, 1]).all():
        raise ValueError("Labels respiratoires invalides")
    for name, data in (("labels", target), ("SleepFM", sleepfm), ("OSF", osf)):
        idx, times = data["epoch_idx"], data["onset_s"]
        if (
            idx.shape != (n,)
            or not np.issubdtype(idx.dtype, np.integer)
            or (idx < 0).any()
            or len(np.unique(idx)) != n
            or times.shape != (n,)
            or not np.isfinite(times).all()
        ):
            raise ValueError(f"{name} : indices ou temps invalides")
    if not all(np.array_equal(target[k], sleepfm[k]) for k in ("epoch_idx", "onset_s")):
        raise ValueError("Labels non alignés avec SleepFM")
    index = {int(value): i for i, value in enumerate(osf["epoch_idx"])}
    if set(index) != set(target["epoch_idx"].tolist()):
        raise ValueError("Fenêtres différentes entre OSF et SleepFM")
    order = np.array([index[int(i)] for i in target["epoch_idx"]], dtype=np.int64)
    if (np.abs(target["onset_s"] - osf["onset_s"][order]) > TIME_TOLERANCE_S).any():
        raise ValueError("Temps différents entre OSF et SleepFM")
    if sleepfm["y"].shape != (n,) or not np.array_equal(sleepfm["y"], osf["y"][order]):
        raise ValueError("Stades différents entre OSF et SleepFM")
    X = osf["X"]
    if X.shape != (n, OSF_DIM) or X.dtype.kind != "f" or not np.isfinite(X).all():
        raise ValueError("Embeddings OSF invalides")
    return order


def paired_ap(y, sleepfm, osf, subjects, n_boot=N_BOOT) -> dict:
    y, sleepfm, osf, subjects = map(np.asarray, (y, sleepfm, osf, subjects))
    if (
        y.ndim != 1
        or not len(y)
        or not all(a.shape == y.shape for a in (sleepfm, osf, subjects))
        or set(np.unique(y).tolist()) != {0, 1}
    ):
        raise ValueError("Deux classes et vecteurs alignés requis")
    if any(not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in (sleepfm, osf)):
        raise ValueError("Probabilités invalides")
    if n_boot <= 0:
        raise ValueError("Nombre de tirages positif requis")
    unique, inverse = np.unique(subjects, return_inverse=True)
    draws = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        positive = np.dot(weights, y)
        if positive == 0 or positive == weights.sum():
            continue
        a = average_precision_score(y, sleepfm, sample_weight=weights)
        b = average_precision_score(y, osf, sample_weight=weights)
        draws.append((a, b, b - a))
    if not draws:
        raise ValueError("Aucun tirage bootstrap évaluable")
    ci = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0).T
    difference = float(average_precision_score(y, osf) - average_precision_score(y, sleepfm))
    return {
        "difference_osf_minus_sleepfm": difference,
        "ci95": ci[2].tolist(),
        "individual_ci95": {"sleepfm": ci[0].tolist(), "osf": ci[1].tolist()},
        "n_boot": n_boot,
        "valid_draws": len(draws),
        "n_subjects": len(unique),
        "seed": SEED,
    }


def validated_inputs(args) -> tuple[dict, dict, dict, dict]:
    baseline = json.loads(args.baseline.read_text())
    split = json.loads(args.split_file.read_text())
    if (
        baseline.get("status") != "terminé"
        or baseline.get("partitions") != partition_subjects(split)
        or baseline.get("split_sha256") != file_sha256(args.split_file)
        or baseline.get("target_events") != list(TARGET_EVENTS)
        or baseline.get("min_overlap_s") != MIN_OVERLAP_S
    ):
        raise ValueError("Baseline non terminée ou protocole/partitions différents")
    manifest_path = args.labels_dir / "manifest.json"
    if file_sha256(manifest_path) != baseline["label_manifest_sha256"]:
        raise ValueError("Inventaire des labels différent de la baseline")
    labels = json.loads(manifest_path.read_text())
    if labels["embeddings_manifest"] != json.loads((args.sleepfm_dir / "manifest.json").read_text()):
        raise ValueError("Contrat SleepFM différent de celui des labels")
    archive = args.baseline.with_suffix(".predictions.npz")
    if file_sha256(archive) != baseline["predictions_sha256"]:
        raise ValueError("Prédictions SleepFM modifiées")
    if file_sha256(args.baseline.with_suffix(".joblib")) != baseline["model_sha256"]:
        raise ValueError("Modèle SleepFM de référence modifié")
    osf_manifest = json.loads((args.osf_dir / "manifest.json").read_text())
    if (
        osf_manifest.get("model") != "OSF"
        or osf_manifest.get("extraction_preprocessing_version") != 2
        or not osf_manifest.get("osf_checkpoint_sha256")
    ):
        raise ValueError("Contrat OSF incomplet ou incompatible")
    status = json.loads((args.osf_dir / "run_status.json").read_text())
    subjects = set(split["train"] + split["test"])
    if (
        not status.get("complete")
        or status.get("rejected")
        or status.get("missing")
        or set(status.get("embedded", [])) != subjects
    ):
        raise ValueError("Couverture OSF incomplète ou différente")
    return baseline, labels, osf_manifest, split


def verify_caches(args, sources: dict, subjects: list[str]) -> dict:
    inventory = {}
    for i, subject in enumerate(subjects):
        label_path = args.labels_dir / f"{subject}.npz"
        sleepfm_path = args.sleepfm_dir / f"{subject}.npz"
        osf_path = args.osf_dir / f"{subject}.npz"
        if file_sha256(label_path) != sources[subject]["labels_sha256"]:
            raise ValueError(f"{subject} : labels modifiés")
        if file_sha256(sleepfm_path) != sources[subject]["embedding_sha256"]:
            raise ValueError(f"{subject} : cache SleepFM modifié")
        with np.load(label_path) as target, np.load(sleepfm_path) as sleepfm, np.load(osf_path) as osf:
            align_subject(target, sleepfm, osf)
        inventory[subject] = {
            "osf_sha256": file_sha256(osf_path),
            "sleepfm_sha256": sources[subject]["embedding_sha256"],
            "labels_sha256": sources[subject]["labels_sha256"],
        }
        if (i + 1) % 100 == 0:
            print(f"Entrées alignées : {i + 1}/{len(subjects)} sujets", flush=True)
    return inventory


def load_osf_partition(subjects: list[str], directory: Path, labels_dir: Path):
    sizes = []
    for subject in subjects:
        with np.load(labels_dir / f"{subject}.npz") as data:
            sizes.append(int((data["y"] >= 0).sum()))
    n = sum(sizes)
    X, y = np.empty((n, OSF_DIM), dtype=np.float32), np.empty(n, dtype=np.int64)
    subject_rows = np.empty(n, dtype=f"U{max(map(len, subjects))}")
    epoch_idx, onset_s = np.empty(n, dtype=np.int64), np.empty(n)
    offset = 0
    for subject, size in zip(subjects, sizes, strict=True):
        with np.load(labels_dir / f"{subject}.npz") as target, np.load(directory / f"{subject}.npz") as osf:
            index = {int(value): i for i, value in enumerate(osf["epoch_idx"])}
            keep = target["y"] >= 0
            selected = np.array([index[int(i)] for i in target["epoch_idx"][keep]], dtype=np.int64)
            rows = slice(offset, offset + size)
            X[rows], y[rows] = osf["X"][selected], target["y"][keep]
            subject_rows[rows], epoch_idx[rows], onset_s[rows] = (
                subject,
                target["epoch_idx"][keep],
                target["onset_s"][keep],
            )
            offset += size
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Deux classes requises dans chaque partition")
    return X, y, (subject_rows, epoch_idx, onset_s)


def run(args) -> dict:
    outputs = {
        extension: Path(f"{args.out}.{extension}")
        for extension in ("json", "predictions.npz", "joblib", "inputs.json", "code.zip")
    }
    for directory in (args.sleepfm_dir, args.osf_dir, args.labels_dir):
        if args.out.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Sorties dans un dossier d'entrée refusées")
    if any(path.exists() for path in outputs.values()):
        raise ValueError("Sortie existante : aucun résultat n'est écrasé")
    baseline, labels, osf_manifest, split = validated_inputs(args)
    subjects = sorted(split["train"] + split["test"])
    inventory = verify_caches(args, labels["sources"], subjects)
    report = {
        "status": "en cours",
        "kind": "comparaison exploratoire respiratoire OSF / SleepFM",
        "plan_sha256": file_sha256(args.plan),
        "baseline_sha256": file_sha256(args.baseline),
        "label_manifest_sha256": file_sha256(args.labels_dir / "manifest.json"),
        "split_sha256": file_sha256(args.split_file),
        "partitions": baseline["partitions"],
        "target_events": list(TARGET_EVENTS),
        "min_overlap_s": MIN_OVERLAP_S,
        "sleepfm_manifest": baseline["embeddings_manifest"],
        "osf_manifest": osf_manifest,
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Sujets déjà examinés, 4 canaux OSF absents, normalisation supposée ; pas un diagnostic",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(outputs["json"], report)
    write_json_atomic(outputs["inputs.json"], inventory)
    code = ["scripts/compare_respiratory_osf.py", "scripts/respiratory_baseline.py", "src/data/respiratory_events.py"]
    report["code_files"] = {name: file_sha256(Path(name)) for name in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*map(Path, code), args.plan]:
            archive.write(path, str(path))
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    X_fit, y_fit, _ = load_osf_partition(baseline["partitions"]["fit"], args.osf_dir, args.labels_dir)
    print(f"Ajustement OSF : {len(y_fit)} fenêtres", flush=True)
    began = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_fit, y_fit)
    del X_fit, y_fit
    report["osf_fit_seconds"] = time.perf_counter() - began
    report["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    report["n_iter"] = model.n_iter_.tolist()
    report["hyperparameters"] = model.get_params()
    X_val, y_val, _ = load_osf_partition(baseline["partitions"]["validation"], args.osf_dir, args.labels_dir)
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    del X_val, y_val
    X_test, y_test, groups = load_osf_partition(baseline["partitions"]["test"], args.osf_dir, args.labels_dir)
    osf_probability = model.predict_proba(X_test)[:, 1]
    del X_test
    with np.load(args.baseline.with_suffix(".predictions.npz")) as data:
        if not (
            np.array_equal(data["y_true"], y_test)
            and all(
                np.array_equal(data[k], group)
                for k, group in zip(("subject", "epoch_idx", "onset_s"), groups, strict=True)
            )
        ):
            raise ValueError("Archives finales non appariées : aucun score comparatif")
        sleepfm_probability = data["probability"]
    primary = paired_ap(y_test, sleepfm_probability, osf_probability, groups[0])
    low, high = primary["ci95"]
    verdict = "OSF au-dessus" if low > 0 else "SleepFM au-dessus" if high < 0 else "pas de différence établie"
    metrics = {
        "sleepfm": binary_metrics(y_test, sleepfm_probability, baseline["threshold"]["value"]),
        "osf": binary_metrics(y_test, osf_probability, threshold),
    }
    if any(
        abs(metrics["sleepfm"][k] - baseline["metrics"][k]) > 1e-12
        for k in ("average_precision", "roc_auc", "f1", "sensitivity", "specificity")
    ):
        raise ValueError("Métriques SleepFM différentes du résultat de référence")
    partial = outputs["predictions.npz"].with_name(f".{outputs['predictions.npz'].stem}.tmp.npz")
    np.savez_compressed(
        partial,
        y_true=y_test,
        subject=groups[0],
        epoch_idx=groups[1],
        onset_s=groups[2],
        probability_sleepfm=sleepfm_probability,
        probability_osf=osf_probability,
    )
    os.replace(partial, outputs["predictions.npz"])
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "contract": osf_manifest,
            "target_events": TARGET_EVENTS,
            "min_overlap_s": MIN_OVERLAP_S,
        },
        outputs["joblib"],
    )
    report.update(
        status="terminé",
        primary=primary,
        verdict=verdict,
        metrics=metrics,
        thresholds={"sleepfm": baseline["threshold"]["value"], "osf": threshold},
        test_epochs=len(y_test),
        coverage=baseline["coverage"],
        artifacts={
            ext: {"file": path.name, "sha256": file_sha256(path)} for ext, path in outputs.items() if ext != "json"
        },
    )
    write_json_atomic(outputs["json"], report)
    print(f"AP OSF - SleepFM : {primary['difference_osf_minus_sleepfm']:+.4f} [{low:+.4f} ; {high:+.4f}] : {verdict}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/respiratory_baseline_physionet2018.json"))
    parser.add_argument("--sleepfm-dir", type=Path, default=Path("embeddings/physionet2018_v2"))
    parser.add_argument("--osf-dir", type=Path, default=Path("embeddings/physionet2018_osf"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/respiratory_labels"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan comparaison OSF SleepFM respiratoire 12-11.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Comparaison arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
