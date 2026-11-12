"""Baseline exploratoire apnées/hypopnées sur embeddings SleepFM figés.

Plan initial dans docs/plans/ (voir --protocol). Ne lit pas les 494 sujets de confirmation,
ne modifie pas les stades et n'utilise aucune donnée NSRR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import EMBEDDING_DIM, cache_problem, validate_split
from scripts.download_physionet2018 import PHYSIONET_2018_URL, copy_complete
from src.data.cache_contract import file_sha256, missing_contract_keys, write_json_atomic
from src.data.respiratory_events import MIN_OVERLAP_S, TARGET_EVENTS, read_respiratory_labels
from src.data.subject_ids import validate_subject_ids
from src.models.probe_export import run_environment

SEED = 42
N_BOOT = 500


def partition_subjects(split: dict) -> dict[str, list[str]]:
    """Validation choisie parmi le train seulement ; test conservé à l'identique."""
    validate_split(split, "split respiratoire")
    for role in ("train", "test"):
        validate_subject_ids(split[role])
    if len(split["train"]) < 10:
        raise ValueError("Au moins 10 sujets train requis pour séparer la validation")
    train = np.array(sorted(split["train"]))
    shuffled = np.random.default_rng(SEED).permutation(train)
    n_val = max(1, len(train) // 10)
    return {
        "fit": sorted(shuffled[n_val:].tolist()),
        "validation": sorted(shuffled[:n_val].tolist()),
        "test": sorted(split["test"]),
    }


def fetch_annotations(folder: Path, subject: str) -> None:
    """Deux petits fichiers publics, aucun signal PSG ; reprise des fichiers complets."""
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("hea", "arousal"):
        path = folder / f"{subject}.{extension}"
        if path.exists() and path.stat().st_size:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(
                        f"{PHYSIONET_2018_URL}/training/{subject}/{path.name}", timeout=60
                    ) as response,
                    temporary.open("wb") as output,
                ):
                    copy_complete(response, output)
                if not temporary.stat().st_size:
                    raise OSError("Annotation vide")
                os.replace(temporary, path)
                break
            except OSError:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2**attempt)


def prepare_labels(subjects: list[str], embeddings: Path, annotations: Path, output: Path, download: bool) -> dict:
    """Une table de labels séparée par sujet, liée aux temps et fichiers d'origine."""
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for i, subject in enumerate(subjects):
        cache = embeddings / f"{subject}.npz"
        problem = cache_problem(cache, strict_times=True)
        if problem:
            raise ValueError(f"{subject} : {problem}")
        record = annotations / subject / subject
        if download:
            fetch_annotations(record.parent, subject)
        with np.load(cache) as data:
            onsets, indices = data["onset_s"], data["epoch_idx"]
        result = read_respiratory_labels(record, onsets)
        path = output / f"{subject}.npz"
        temporary = output / f".{subject}.tmp.npz"
        np.savez_compressed(
            temporary,
            y=result["y"],
            overlap_s=result["overlap_s"],
            onset_s=onsets,
            epoch_idx=indices,
            intervals_s=result["intervals_s"],
        )
        os.replace(temporary, path)
        sources[subject] = {
            "embedding_sha256": file_sha256(cache),
            "header_sha256": file_sha256(record.with_suffix(".hea")),
            "annotation_sha256": file_sha256(record.with_suffix(".arousal")),
            "labels_sha256": file_sha256(path),
            "event_counts": result["event_counts"],
            "windows": {
                "negative": int((result["y"] == 0).sum()),
                "positive": int((result["y"] == 1).sum()),
                "excluded": int((result["y"] == -1).sum()),
            },
        }
        if (i + 1) % 50 == 0:
            print(f"Labels respiratoires : {i + 1}/{len(subjects)} sujets", flush=True)
    return sources


def load_partition(subjects: list[str], embeddings: Path, labels: Path):
    """Préallocation pour éviter de doubler le cache complet en mémoire."""
    sizes = []
    for subject in subjects:
        with np.load(labels / f"{subject}.npz") as data:
            sizes.append(int((data["y"] >= 0).sum()))
    n = sum(sizes)
    X, y = np.empty((n, EMBEDDING_DIM), dtype=np.float32), np.empty(n, dtype=np.int64)
    subject_rows = np.empty(n, dtype=f"U{max(map(len, subjects))}")
    indices, onsets = np.empty(n, dtype=np.int64), np.empty(n)
    offset = 0
    for subject, size in zip(subjects, sizes, strict=True):
        with np.load(labels / f"{subject}.npz") as target, np.load(embeddings / f"{subject}.npz") as cache:
            if not all(np.array_equal(target[k], cache[k]) for k in ("onset_s", "epoch_idx")):
                raise ValueError(f"{subject} : labels non alignés avec le cache")
            keep = target["y"] >= 0
            rows = slice(offset, offset + size)
            X[rows], y[rows] = cache["X"][keep], target["y"][keep]
            subject_rows[rows] = subject
            indices[rows], onsets[rows] = target["epoch_idx"][keep], target["onset_s"][keep]
            offset += size
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Deux classes requises dans chaque partition")
    return X, y, (subject_rows, indices, onsets)


def select_threshold(y, probability) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def binary_metrics(y, probability, threshold) -> dict:
    pred = probability >= threshold
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "prevalence": float(np.mean(y)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": matrix.tolist(),
    }


def ap_bootstrap(y, probability, subjects, n_boot=N_BOOT) -> dict:
    unique, inverse = np.unique(subjects, return_inverse=True)
    rng, scores, differences = np.random.default_rng(SEED), [], []
    # Les poids reproduisent les nuits tirées avec remise, sans recopier les lignes.
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        positives = np.dot(weights, y)
        total = weights.sum()
        if positives == 0 or positives == total:
            continue
        score = average_precision_score(y, probability, sample_weight=weights)
        scores.append(score)
        differences.append(score - positives / total)
    if not scores:
        raise ValueError("Aucun tirage bootstrap évaluable")
    return {
        "average_precision": np.percentile(scores, [2.5, 97.5]).tolist(),
        "ap_minus_prevalence": np.percentile(differences, [2.5, 97.5]).tolist(),
        "n_boot": n_boot,
        "valid_draws": len(scores),
        "n_subjects": len(unique),
        "seed": SEED,
    }


def run(args) -> dict:
    cache_root = args.embeddings_dir.resolve()
    if args.labels_dir.resolve().is_relative_to(cache_root) or args.out.resolve().is_relative_to(cache_root):
        raise ValueError("Les sorties et labels doivent être séparés du dossier d'embeddings d'entrée")
    report_path, archive = Path(f"{args.out}.json"), Path(f"{args.out}.predictions.npz")
    model_path = Path(f"{args.out}.joblib")
    if any(p.exists() for p in (report_path, archive, model_path, Path(f"{args.out}.labels.json"))):
        raise ValueError("Sortie existante : conserver le premier résultat, choisir un nouveau préfixe")
    split = json.loads(args.split_file.read_text())
    parts = partition_subjects(split)
    manifest_path = args.embeddings_dir / "manifest.json"
    contract = json.loads(manifest_path.read_text())
    if missing_contract_keys(contract) or contract["preprocessing_version"] != 2:
        raise ValueError("Contrat SleepFM v2 complet requis")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "en cours",
        "kind": "baseline exploratoire apnées et hypopnées, tous types",
        "protocol_sha256": file_sha256(args.protocol),
        "split_sha256": file_sha256(args.split_file),
        "target_events": list(TARGET_EVENTS),
        "min_overlap_s": MIN_OVERLAP_S,
        "partitions": parts,
        "embeddings_manifest": contract,
        "environment": run_environment(),
        "code_files": {
            p: file_sha256(Path(p)) for p in ("scripts/respiratory_baseline.py", "src/data/respiratory_events.py")
        },
        "command": sys.argv,
        "caveat": "PSG, fenêtres annotées, sujets déjà examinés ; pas un IAH ni un diagnostic",
    }
    write_json_atomic(report_path, report)
    sources = prepare_labels(
        sorted(split["train"] + split["test"]),
        args.embeddings_dir,
        args.annotations_dir,
        args.labels_dir,
        args.download,
    )
    label_manifest = {
        "sources": sources,
        "split_sha256": report["split_sha256"],
        "target_events": list(TARGET_EVENTS),
        "min_overlap_s": MIN_OVERLAP_S,
        "embeddings_manifest": contract,
    }
    write_json_atomic(args.labels_dir / "manifest.json", label_manifest)
    write_json_atomic(Path(f"{args.out}.labels.json"), label_manifest)
    report["label_manifest_sha256"] = file_sha256(args.labels_dir / "manifest.json")
    report["coverage"] = {
        role: {key: sum(sources[s]["windows"][key] for s in subjects) for key in ("negative", "positive", "excluded")}
        for role, subjects in parts.items()
    }
    X_fit, y_fit, _ = load_partition(parts["fit"], args.embeddings_dir, args.labels_dir)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    began = time.perf_counter()
    print(f"Ajustement : {len(y_fit)} fenêtres, {len(parts['fit'])} sujets", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_fit, y_fit)
    del X_fit, y_fit
    report["fit_seconds"] = time.perf_counter() - began
    report["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    report["n_iter"] = model.n_iter_.tolist()
    report["hyperparameters"] = model.get_params()
    X_val, y_val, _ = load_partition(parts["validation"], args.embeddings_dir, args.labels_dir)
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    del X_val, y_val
    X_test, y_test, groups = load_partition(parts["test"], args.embeddings_dir, args.labels_dir)
    probability = model.predict_proba(X_test)[:, 1]
    del X_test
    report["threshold"] = {"value": threshold, "rule": "F1 validation, plus grand seuil en cas d'égalité"}
    report["metrics"] = binary_metrics(y_test, probability, threshold)
    report["ci95"] = ap_bootstrap(y_test, probability, groups[0])
    temporary = archive.with_name(f".{archive.stem}.tmp.npz")
    np.savez_compressed(
        temporary, y_true=y_test, probability=probability, subject=groups[0], epoch_idx=groups[1], onset_s=groups[2]
    )
    os.replace(temporary, archive)
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "contract": contract,
            "target_events": TARGET_EVENTS,
            "min_overlap_s": MIN_OVERLAP_S,
        },
        model_path,
    )
    report.update(
        status="terminé",
        predictions_sha256=file_sha256(archive),
        model_sha256=file_sha256(model_path),
        test_epochs=len(y_test),
    )
    write_json_atomic(report_path, report)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings/physionet2018_v2"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/respiratory_labels"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan SleepFM respiratoire 12-11.md",
    )
    parser.add_argument("--download", action="store_true", help="Télécharger .hea/.arousal seulement")
    parser.add_argument("--out", type=Path, required=True, help="Préfixe de nouvelles sorties")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Baseline arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
