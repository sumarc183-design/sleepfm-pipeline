"""Confirmation de l'IAH ≥ 15 sur NSRR avec la règle figée sur PhysioNet (un seul passage).

Plan validé : docs/plans/Plan confirmation IAH NSRR SleepFM OSF 13-11.md. Entrées : embeddings de
`scripts/embed_nsrr.py` (les deux modèles, règle d'échelle `physionet_median_v1`) et annotations XML
NSRR. Modèles, seuils et droites de calibration viennent du résultat PhysioNet figé
(`results/respiratory_ahi_physionet2018.json`, empreinte contrôlée) : rien n'est réajusté.

Référence principale : 4 types d'événements par heure de sommeil expert (`reference_ahi_nsrr`).
Référence secondaire facultative : IAH officiel NSRR lu dans un CSV (`--official-csv`).
Critère de réussite, pour chaque modèle : bornes basses des IC 95 % de la sensibilité et de la
spécificité au-dessus de 60 %. Écrit une seule fois : une sortie existante n'est jamais écrasée.

    python scripts/evaluate_nsrr_ahi.py --cohort mesa --embeddings embeddings/nsrr_mesa \\
        --annotations-dir data/nsrr/mesa/annotations-events-nsrr --out results/nsrr_ahi_mesa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.estimate_respiratory_ahi import MODELS, bootstrap, evaluate_test  # noqa: E402
from src.data.ahi import AHI_CUTOFF, MIN_SLEEP_H, positive_rate, spearman  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.nsrr_annotations import parse_nsrr_xml, reference_ahi_nsrr  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FROZEN_AHI_SHA256 = "eda69eb95c167a1df729b4c728e744ef0275a4ec725b7ace2727f40954bd2990"
RESPIRATORY_RULE = "physionet_median_v1"
SUCCESS_LOWER_BOUND = 0.60
PLAN = ROOT / "docs/plans/Plan confirmation IAH NSRR SleepFM OSF 13-11.md"
CODE = (
    "scripts/evaluate_nsrr_ahi.py",
    "scripts/estimate_respiratory_ahi.py",
    "scripts/embed_nsrr.py",
    "src/data/ahi.py",
    "src/data/nsrr_annotations.py",
    "src/data/nsrr_channels.py",
    "src/data/nsrr_signals.py",
)


def load_frozen(ahi_json: Path, baseline: Path, comparison: Path, expected_sha: str = FROZEN_AHI_SHA256) -> dict:
    """Modèles, seuils, calibration et poids attendus, tous rattachés au résultat PhysioNet figé."""
    if file_sha256(ahi_json) != expected_sha:
        raise ValueError("Résultat IAH PhysioNet différent de la règle figée")
    frozen = json.loads(ahi_json.read_text())
    if file_sha256(baseline) != frozen["source_sha256"]["baseline"]:
        raise ValueError("Baseline SleepFM différente de celle de la règle figée")
    if file_sha256(comparison) != frozen["source_sha256"]["comparison"]:
        raise ValueError("Comparaison OSF différente de celle de la règle figée")
    base, comp = json.loads(baseline.read_text()), json.loads(comparison.read_text())
    paths = {"sleepfm": baseline.with_suffix(".joblib"), "osf": comparison.parent / comp["artifacts"]["joblib"]["file"]}
    expected = {"sleepfm": base["model_sha256"], "osf": comp["artifacts"]["joblib"]["sha256"]}
    models = {}
    for model in MODELS:
        if file_sha256(paths[model]) != expected[model]:
            raise ValueError(f"Modèle {model} modifié")
        bundle = joblib.load(paths[model])
        if float(bundle["threshold"]) != float(frozen["thresholds"][model]):
            raise ValueError(f"Seuil {model} différent de la règle figée")
        models[model] = bundle["model"]
    return {
        "models": models,
        "thresholds": {m: float(frozen["thresholds"][m]) for m in MODELS},
        "calibration": frozen["calibration"],
        "checkpoints": {
            "sleepfm": base["embeddings_manifest"]["checkpoint_sha256"],
            "osf": comp["osf_manifest"]["osf_checkpoint_sha256"],
        },
    }


def check_embeddings(folder: Path, cohort: str, frozen: dict) -> tuple[dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    status = json.loads((folder / "run_status.json").read_text())
    expected = {
        "cohort": cohort,
        "models": sorted(MODELS),
        "respiratory_rule": RESPIRATORY_RULE,
        "sleepfm_checkpoint_sha256": frozen["checkpoints"]["sleepfm"],
        "osf_checkpoint_sha256": frozen["checkpoints"]["osf"],
    }
    wrong = {k: manifest.get(k) for k, v in expected.items() if manifest.get(k) != v}
    if wrong:
        raise ValueError(f"Embeddings NSRR non conformes : {wrong}")
    if not status["complete"] or status["missing"]:
        raise ValueError("Extraction incomplète : terminer le run avant l'évaluation")
    return manifest, status


def official_reference(path: Path, id_column: str, value_column: str) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or id_column not in rows[0] or value_column not in rows[0]:
        raise ValueError(f"{path} : colonnes {id_column} et {value_column} attendues")
    out = {}
    for row in rows:
        value = row[value_column].strip()
        if value:
            out[row[id_column].strip()] = float(value)
    return out


def night_row(npz_path: Path, xml_path: Path, frozen: dict) -> dict:
    """Valeurs d'une nuit, ou motif d'exclusion (`excluded`)."""
    try:
        reference = reference_ahi_nsrr(parse_nsrr_xml(xml_path))
    except (OSError, ValueError) as exc:
        return {"excluded": f"annotations : {exc}"}
    if reference["sleep_h"] < MIN_SLEEP_H:
        return {"excluded": f"sommeil {reference['sleep_h']:.2f} h < {MIN_SLEEP_H:g} h"}
    data = np.load(npz_path)
    row = {"reference_ahi": reference["ahi"], "sleep_h": reference["sleep_h"], "windows": len(data["epoch_idx"])}
    for model in MODELS:
        probability = frozen["models"][model].predict_proba(data[f"X_{model}"])[:, 1]
        rate, probability_sum, _ = positive_rate(probability, frozen["thresholds"][model])
        row[f"rate_{model}"], row[f"probability_sum_{model}"] = rate, probability_sum
    return row


def criterion(uncertainty: dict) -> dict:
    out = {}
    for model in MODELS:
        low_sens = uncertainty["ci95"][model]["sensitivity"][0]
        low_spec = uncertainty["ci95"][model]["specificity"][0]
        out[model] = {
            "sensitivity_lower": low_sens,
            "specificity_lower": low_spec,
            "success": bool(low_sens > SUCCESS_LOWER_BOUND and low_spec > SUCCESS_LOWER_BOUND),
        }
    return out


def against_official(night: dict, estimates: dict, official: np.ndarray) -> dict:
    known = np.isfinite(official)
    out = {"nights": int(known.sum()), "positive_15": int((official[known] >= AHI_CUTOFF).sum())}
    for model in MODELS:
        estimated, reference = estimates[model][known], official[known]
        truth, predicted = reference >= AHI_CUTOFF, estimated >= AHI_CUTOFF
        out[model] = {
            "sensitivity": float((predicted & truth).sum() / truth.sum()) if truth.any() else None,
            "specificity": float((~predicted & ~truth).sum() / (~truth).sum()) if (~truth).any() else None,
            "mae": float(np.abs(estimated - reference).mean()),
            "spearman": spearman(estimated, reference),
        }
    out["agreement_with_primary_15"] = float(
        np.mean((night["reference_ahi"][known] >= AHI_CUTOFF) == (official[known] >= AHI_CUTOFF))
    )
    return out


def run(args) -> dict:
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "nights.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : la confirmation ne se lance qu'une fois")
    frozen = load_frozen(args.frozen_ahi, args.baseline, args.comparison, args.expected_sha)
    manifest, status = check_embeddings(args.embeddings, args.cohort, frozen)
    rows, excluded = {}, dict(status["rejected"])
    for subject in status["done"]:
        row = night_row(
            args.embeddings / f"{subject}.npz", args.annotations_dir / args.xml_pattern.format(subject=subject), frozen
        )
        if "excluded" in row:
            excluded[subject] = row["excluded"]
        else:
            rows[subject] = row
    subjects = sorted(rows)
    if len(subjects) < 2:
        raise ValueError("Moins de 2 nuits évaluables")
    night = {"subject": np.array(subjects)}
    for key in rows[subjects[0]]:
        night[key] = np.array([rows[s][key] for s in subjects], dtype=float)
    truth = night["reference_ahi"] >= AHI_CUTOFF
    if truth.all() or not truth.any():
        raise ValueError("Une seule classe IAH parmi les nuits : critère non calculable")
    eligible = np.ones(len(subjects), dtype=bool)
    results, estimates = evaluate_test(night, frozen["calibration"], eligible)
    uncertainty = bootstrap(night, estimates, eligible)
    report = {
        "kind": "nsrr_ahi_confirmation",
        "cohort": args.cohort,
        "source_sha256": {
            "frozen_ahi": file_sha256(args.frozen_ahi),
            "plan": file_sha256(args.plan),
            "embeddings_manifest": file_sha256(args.embeddings / "manifest.json"),
            "subjects": manifest["subjects_sha256"],
        },
        "nights": {
            "evaluated": len(subjects),
            "positive_15": int(truth.sum()),
            "excluded": excluded,
        },
        "thresholds": frozen["thresholds"],
        "calibration": frozen["calibration"],
        "results": results,
        "bootstrap": uncertainty,
        "success_criterion": {"lower_bound": SUCCESS_LOWER_BOUND, "models": criterion(uncertainty)},
        "environment": run_environment(),
        "command": sys.argv,
    }
    for model in MODELS:
        night[f"estimated_ahi_{model}"] = estimates[model]
    if args.official_csv is not None:
        values = official_reference(args.official_csv, args.official_id_column, args.official_column)
        night["official_ahi"] = np.array([values.get(s, np.nan) for s in subjects], dtype=float)
        report["official_reference"] = {
            "column": args.official_column,
            "csv_sha256": file_sha256(args.official_csv),
            **against_official(night, estimates, night["official_ahi"]),
        }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["nights.npz"], **night)
    report["code_files"] = {p: file_sha256(ROOT / p) for p in CODE}
    with zipfile.ZipFile(outputs["code.zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in CODE:
            archive.write(ROOT / p, p)
        archive.write(args.plan, args.plan.name)
    report["artifacts"] = {
        ext: {"file": p.name, "sha256": file_sha256(p)} for ext, p in outputs.items() if ext != "json"
    }
    write_json_atomic(outputs["json"], report)
    for model, verdict in report["success_criterion"]["models"].items():
        primary = results[model]["primary"]
        print(
            f"{model} : sensibilité {primary['sensitivity']:.3f} (borne basse {verdict['sensitivity_lower']:.3f}), "
            f"spécificité {primary['specificity']:.3f} (borne basse {verdict['specificity_lower']:.3f}) ; "
            f"critère {'atteint' if verdict['success'] else 'non atteint'}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=("mesa", "shhs1"), required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--xml-pattern", default="{subject}-nsrr.xml")
    parser.add_argument("--out", type=Path, required=True, help="préfixe des sorties (.json, .nights.npz, .code.zip)")
    parser.add_argument("--frozen-ahi", type=Path, default=ROOT / "results/respiratory_ahi_physionet2018.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/respiratory_baseline_physionet2018.json")
    parser.add_argument(
        "--comparison", type=Path, default=ROOT / "results/respiratory_osf_vs_sleepfm_physionet2018.json"
    )
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--official-csv", type=Path, default=None, help="variables NSRR (IAH officiel)")
    parser.add_argument("--official-id-column", default="nsrrid")
    parser.add_argument("--official-column", default=None, help="ex. ahi_a0h3 (nom à vérifier)")
    args = parser.parse_args()
    args.expected_sha = FROZEN_AHI_SHA256
    if args.official_csv is not None and args.official_column is None:
        parser.error("--official-column obligatoire avec --official-csv")
    run(args)


if __name__ == "__main__":
    main()
