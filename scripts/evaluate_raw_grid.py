"""Performances de la prédiction sur PSG brute (grille fixe de 30 s), sans entraînement.

Lit les embeddings de scripts/embed_raw_grid.py, prédit avec les modèles figés
du run à 500 sujets (entraînés sur les fenêtres annotées), et note les fenêtres
qui portent un seul stade défini. Les modèles voient toute la nuit, y compris la
période « undefined » du début ; seules les fenêtres non notées sont exclues du
score, et leur nombre est rapporté. Comparaison descriptive aux mêmes nuits sur
la grille annotée (archive du run à 500 sujets) : les fenêtres diffèrent, il n'y
a donc pas d'écart apparié. Les fenêtres à cheval sur deux stades, exclues du score,
sont surtout des transitions, les plus difficiles : le score de la grille brute peut
être optimiste par rapport à celui de la grille annotée, qui les garde.

    python scripts/evaluate_raw_grid.py --embeddings-dir embeddings/physionet2018_raw_grid \\
        --out results/raw_grid_physionet2018
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402
from src.data.raw_grid_labels import UNSCORED  # noqa: E402
from src.models.linear_probe import bootstrap_ci, predict_labels  # noqa: E402
from src.models.probe_export import load_probes, run_environment  # noqa: E402

PROBES = {"logistic": "Baseline", "sequence": "BiGRU"}
STAGES = [name for name, _ in sorted(STAGE_NAME_TO_LABEL.items(), key=lambda item: item[1])]


def scores(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> dict:
    ci = bootstrap_ci(y, pred, subjects)
    return {
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ci95": ci,
        "f1_by_stage": dict(
            zip(STAGES, f1_score(y, pred, labels=list(range(len(STAGES))), average=None).tolist(), strict=True)
        ),
    }


def evaluate(embeddings_dir: Path, models_dir: Path, split_file: Path, reference: Path | None) -> dict:
    manifest = json.loads((embeddings_dir / "manifest.json").read_text())
    if manifest.get("epoch_selection") != "fixed_30s_from_recording_start":
        raise SystemExit("Dossier qui ne vient pas de la grille fixe (embed_raw_grid.py)")
    contract = {k: v for k, v in manifest.items() if k != "epoch_selection"}
    models, export = load_probes(models_dir, contract)
    split = json.loads(split_file.read_text())
    status = json.loads((embeddings_dir / "run_status.json").read_text())
    if not status.get("complete"):
        raise SystemExit("Run de la grille incomplet : voir run_status.json")
    subjects = status["done"]
    if set(subjects) & set(split["train"]):
        raise SystemExit("Des sujets d'entraînement des modèles figés sont dans le dossier")

    rows = {"X": [], "y": [], "subject": [], "epoch_idx": [], "onset_s": []}
    for subject in subjects:
        with np.load(embeddings_dir / f"{subject}.npz") as data:
            rows["X"].append(data["X"])
            rows["y"].append(data["y_grid"])
            rows["epoch_idx"].append(data["epoch_idx"])
            rows["onset_s"].append(data["onset_s"])
            rows["subject"].append(np.full(len(data["y_grid"]), subject))
    X, y = np.concatenate(rows["X"]), np.concatenate(rows["y"])
    groups = tuple(np.concatenate(rows[k]) for k in ("subject", "epoch_idx", "onset_s"))
    scored = y != UNSCORED
    predictions = {kind: predict_labels(models[kind], X, groups) for kind in PROBES}

    report = {
        "kind": "prédiction sur PSG brute, grille fixe de 30 s depuis le début",
        "caveat": "fenêtres à cheval sur deux stades (transitions) exclues du score : possiblement optimiste",
        "models_manifest_sha256": file_sha256(models_dir / "manifest.json"),
        "embeddings_manifest": manifest,
        "nights": len(subjects),
        "rejected": status.get("rejected", {}),
        "windows": {
            "total": int(len(y)),
            "scored": int(scored.sum()),
            **{k: int(sum(c[k] for c in status["counts"].values())) for k in ("mixed_stages", "undefined")},
        },
        "grid": {kind: scores(y[scored], predictions[kind][scored], groups[0][scored]) for kind in PROBES},
        "environment": run_environment(),
    }
    # Ce que les modèles prédisent avant le début de l'annotation (lumières allumées, en général).
    undefined = ~scored
    report["unscored_predictions"] = {
        kind: dict(zip(STAGES, np.bincount(predictions[kind][undefined], minlength=5).tolist(), strict=True))
        for kind in PROBES
    }
    if reference is not None:
        ref = np.load(reference)
        same = np.isin(ref["subject"], subjects)
        report["annotated_grid_same_nights"] = {
            "source": str(reference),
            "source_sha256": file_sha256(reference),
            "nights": len(set(ref["subject"][same])),
            **{kind: float(cohen_kappa_score(ref["y_true"][same], ref[f"pred_{kind}"][same])) for kind in PROBES},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models/physionet2018_500"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--reference", type=Path, default=Path("results/results_physionet2018_500.predictions.npz"))
    parser.add_argument("--out", type=Path, required=True, help="préfixe du JSON de sortie")
    args = parser.parse_args()
    report = evaluate(args.embeddings_dir, args.models, args.split_file, args.reference)
    write_json_atomic(Path(f"{args.out}.json"), report)

    w = report["windows"]
    print(
        f"{report['nights']} nuits, {w['total']} fenêtres, {w['scored']} notées "
        f"({w['mixed_stages']} à cheval sur deux stades, {w['undefined']} non annotées)"
    )
    for kind, name in PROBES.items():
        s = report["grid"][kind]
        low, high = s["ci95"]["cohen_kappa"]
        line = f"{name} sur la grille brute : κ {s['cohen_kappa']:.3f} [{low:.3f} ; {high:.3f}], F1 {s['macro_f1']:.3f}"
        if "annotated_grid_same_nights" in report:
            line += f" ; grille annotée, mêmes nuits : κ {report['annotated_grid_same_nights'][kind]:.3f}"
        print(line)
    print("Attention : fenêtres à cheval sur deux stades (transitions) non notées ; score possiblement optimiste.")
    print(f"Prédictions sur les fenêtres non notées : {report['unscored_predictions']}")
    print(f"Résultats : {args.out}.json")


if __name__ == "__main__":
    main()
