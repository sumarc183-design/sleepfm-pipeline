"""Complète l'évaluation de la grille brute par une règle de majorité fixée à l'avance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import EMBEDDING_DIM, validate_split
from scripts.download_physionet2018 import PHYSIONET_2018_URL, copy_complete
from src.data.cache_contract import file_sha256, write_json_atomic
from src.data.majority_grid_labels import CATEGORIES, read_majority_grid
from src.data.subject_ids import validate_subject_ids
from src.models.linear_probe import bootstrap_ci, predict_probabilities
from src.models.probe_export import load_probes, run_environment

PROBES = ("logistic", "sequence")


def fetch_h5(record: Path) -> None:
    path = record.parent / f"{record.name}-arousal.mat"
    if path.exists() and path.stat().st_size:
        return
    temporary = path.with_suffix(".part")
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(
                    f"{PHYSIONET_2018_URL}/training/{record.name}/{path.name}", timeout=60
                ) as response,
                temporary.open("wb") as output,
            ):
                copy_complete(response, output)
            if not temporary.stat().st_size:
                raise OSError("Annotation HDF5 vide")
            os.replace(temporary, path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def check_grid(data) -> None:
    n = len(data["y_grid"])
    if (
        not n
        or data["X"].shape != (n, EMBEDDING_DIM)
        or data["X"].dtype.kind != "f"
        or not np.isfinite(data["X"]).all()
        or data["y_grid"].shape != (n,)
        or not np.isin(data["y_grid"], range(-1, 5)).all()
        or data["epoch_idx"].shape != (n,)
        or not np.issubdtype(data["epoch_idx"].dtype, np.integer)
        or not np.array_equal(data["epoch_idx"], np.arange(n))
        or data["onset_s"].shape != (n,)
        or not np.array_equal(data["onset_s"], np.arange(n) * 30.0)
    ):
        raise ValueError("Cache de grille fixe invalide")


def stage_scores(y, prediction, subjects) -> dict | None:
    if not len(y):
        return None
    labels = np.unique(np.r_[y, prediction])
    kappa = float(cohen_kappa_score(y, prediction)) if len(labels) > 1 else None
    ci = bootstrap_ci(y, prediction, subjects) if len(labels) > 1 else None
    if ci is not None:
        # Un bootstrap réduit peut parfois être dégénéré : ne pas publier de NaN.
        for key in ("cohen_kappa", "macro_f1"):
            if not np.isfinite(ci[key]).all():
                ci[key] = None
    per_stage = f1_score(y, prediction, labels=list(range(5)), average=None, zero_division=0)
    return {
        "windows": len(y),
        "nights": len(np.unique(subjects)),
        "classes_present": labels.tolist(),
        "accuracy": float(accuracy_score(y, prediction)),
        "cohen_kappa": kappa,
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "f1_by_stage": per_stage.tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=range(5)).tolist(),
        "ci95": ci,
    }


def run(args) -> dict:
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "predictions.npz", "inputs.json", "code.zip")}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("Sortie existante : les résultats antérieurs sont conservés")
    for root in (args.embeddings_dir, args.models, args.annotations_dir):
        if args.out.resolve().is_relative_to(root.resolve()):
            raise ValueError("Sortie dans un dossier d'entrée refusée")
    split = json.loads(args.split_file.read_text())
    validate_split(split, "grille brute")
    subjects = sorted(validate_subject_ids(split["test"]))
    status = json.loads((args.embeddings_dir / "run_status.json").read_text())
    if (
        not status.get("complete")
        or status.get("rejected")
        or status.get("missing")
        or set(status.get("done", [])) != set(subjects)
        or len(status.get("done", [])) != len(subjects)
    ):
        raise ValueError("Couverture de la grille différente du test")
    manifest = json.loads((args.embeddings_dir / "manifest.json").read_text())
    if manifest.get("epoch_selection") != "fixed_30s_from_recording_start":
        raise ValueError("Grille indépendante des annotations requise")
    frozen = json.loads(args.frozen.read_text())
    contract = {k: v for k, v in manifest.items() if k != "epoch_selection"}
    if contract != frozen["contract"] or file_sha256(args.split_file) != frozen["split_sha256"]:
        raise ValueError("Contrat ou découpage différent de la référence figée")
    reference = json.loads(args.reference.read_text())
    if reference["models_manifest_sha256"] != file_sha256(args.models / "manifest.json"):
        raise ValueError("Manifeste des sondes différent de l'évaluation uniforme")
    annotation_sources = json.loads(args.annotation_manifest.read_text())["sources"]
    rows = {key: [] for key in ("X", "subject", "epoch_idx", "onset_s", "y_uniform", "y_majority", "category")}
    inventory = {}
    # Labels et empreintes validés sur toutes les nuits avant le moindre score.
    for i, subject in enumerate(subjects):
        record = args.annotations_dir / subject / subject
        if getattr(args, "download_h5", False):
            fetch_h5(record)
        for extension, key in (("hea", "header_sha256"), ("arousal", "annotation_sha256")):
            if file_sha256(record.with_suffix(f".{extension}")) != annotation_sources[subject][key]:
                raise ValueError(f"{subject} : annotation ou en-tête modifié")
        cache_path = args.embeddings_dir / f"{subject}.npz"
        with np.load(cache_path) as cache:
            check_grid(cache)
            labels = read_majority_grid(record, cache["onset_s"])
            if not np.array_equal(labels["y_uniform"], cache["y_grid"]):
                count = int((labels["y_uniform"] != cache["y_grid"]).sum())
                raise ValueError(f"{subject} : {count} labels uniformes différents ; aucun score publié")
            rows["X"].append(cache["X"])
            rows["subject"].append(np.full(len(cache["y_grid"]), subject))
            for key in ("epoch_idx", "onset_s"):
                rows[key].append(cache[key])
            for key in ("y_uniform", "y_majority", "category"):
                rows[key].append(labels[key])
        inventory[subject] = {
            "cache_sha256": file_sha256(cache_path),
            "h5_sha256": file_sha256(record.parent / f"{subject}-arousal.mat"),
            **{k: annotation_sources[subject][k] for k in ("header_sha256", "annotation_sha256")},
        }
        if (i + 1) % 25 == 0:
            print(f"Labels uniformes retrouvés : {i + 1}/{len(subjects)} nuits", flush=True)
    arrays = {k: np.concatenate(v) for k, v in rows.items()}
    models, exported = load_probes(args.models, contract, expected_sha256=frozen["models"])
    groups = tuple(arrays[k] for k in ("subject", "epoch_idx", "onset_s"))
    subsets = {
        "uniform": arrays["category"] == "uniform",
        "near_uniform": arrays["category"] == "near_uniform",
        "transitions": arrays["category"] == "transition_majority",
        "all_majority": arrays["y_majority"] >= 0,
    }
    counts = {category: int((arrays["category"] == category).sum()) for category in CATEGORIES}
    if sum(counts.values()) != len(arrays["X"]) or len(arrays["X"]) != reference["windows"]["total"]:
        raise ValueError("Effectifs de fenêtres différents de la grille de référence")
    probabilities, predictions, metrics = {}, {}, {}
    for kind in PROBES:
        print(f"Prédiction figée {kind} : {len(arrays['X'])} fenêtres", flush=True)
        probabilities[kind] = predict_probabilities(models[kind], arrays["X"], groups)
        predictions[kind] = models[kind].classes_[probabilities[kind].argmax(axis=1)]
        metrics[kind] = {
            name: stage_scores(arrays["y_majority"][mask], predictions[kind][mask], groups[0][mask])
            for name, mask in subsets.items()
        }
        if any(
            abs(metrics[kind]["uniform"][key] - reference["grid"][kind][key]) > 1e-12
            for key in ("cohen_kappa", "macro_f1")
        ):
            raise ValueError("Scores uniformes non reproduits : aucun résultat complémentaire publié")
    report = {
        "status": "terminé",
        "kind": "évaluation exploratoire des transitions sur grille brute",
        "plan_sha256": file_sha256(args.plan),
        "reference_sha256": file_sha256(args.reference),
        "split_sha256": file_sha256(args.split_file),
        "embedding_manifest": manifest,
        "models": exported["loaded"],
        "nights": len(subjects),
        "windows": len(arrays["X"]),
        "counts": counts,
        "metrics": metrics,
        "environment": run_environment(),
        "command": sys.argv,
        "caveat": "Majorité définie pour l'évaluation, sujets déjà examinés ; ni confirmation ni cotation clinique",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    saved = {k: v for k, v in arrays.items() if k != "X"}
    for kind in PROBES:
        saved[f"pred_{kind}"] = predictions[kind]
        saved[f"proba_{kind}"] = probabilities[kind].astype(np.float32)
        saved[f"classes_{kind}"] = models[kind].classes_
    temporary = outputs["predictions.npz"].with_name(f".{outputs['predictions.npz'].stem}.tmp.npz")
    np.savez_compressed(temporary, **saved)
    os.replace(temporary, outputs["predictions.npz"])
    write_json_atomic(outputs["inputs.json"], inventory)
    code = ["scripts/evaluate_grid_transitions.py", "src/data/majority_grid_labels.py"]
    report["code_files"] = {name: file_sha256(Path(name)) for name in code}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*map(Path, code), args.plan]:
            archive.write(path, str(path))
    report["artifacts"] = {
        key: {"file": path.name, "sha256": file_sha256(path)} for key, path in outputs.items() if key != "json"
    }
    write_json_atomic(outputs["json"], report)
    print(json.dumps({"counts": counts, "metrics": metrics}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings/physionet2018_raw_grid"))
    parser.add_argument("--models", type=Path, default=Path("models/physionet2018_500"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--frozen", type=Path, default=Path("configs/confirmation_physionet2018.json"))
    parser.add_argument("--reference", type=Path, default=Path("results/raw_grid_physionet2018.json"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/respiratory_annotations"))
    parser.add_argument("--annotation-manifest", type=Path, default=Path("data/respiratory_labels/manifest.json"))
    parser.add_argument(
        "--download-h5", action="store_true", help="Télécharger les annotations HDF5 manquantes seulement"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/plans/Plan transitions PSG brute SleepFM 12-11 v2.md",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Évaluation arrêtée : {exc}\n")


if __name__ == "__main__":
    main()
