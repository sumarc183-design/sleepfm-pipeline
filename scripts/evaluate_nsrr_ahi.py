"""Confirmation de l'IAH ≥ 15 sur NSRR avec la règle figée sur PhysioNet (un seul passage).

Plan validé : docs/plans/Plan confirmation IAH NSRR SleepFM OSF 13-11.md. Entrées : embeddings de
`scripts/embed_nsrr.py` (les deux modèles, règle d'échelle `physionet_median_v1`) et annotations XML
NSRR. Modèles, seuils et droites de calibration viennent du résultat PhysioNet figé
(`results/respiratory_ahi_physionet2018.json`, empreinte contrôlée) : rien n'est réajusté.

Référence principale : 4 types d'événements par heure de sommeil expert (`reference_ahi_nsrr`).
Référence secondaire facultative : IAH officiel NSRR à 3 % et 4 % (`--official-csv`), mêmes métriques.
Critère de réussite, pour chaque modèle : bornes basses des IC 95 % de la sensibilité et de la
spécificité au-dessus de 60 %.

Précontrôle sans cible ni prédiction (audit du 13 novembre) : code et plan commités, règle figée,
liste approuvée et couverture exacte, contrat d'extraction complet, chaque cache relu et empreinté,
concepts XML connus, CSV officiel valide. Ensuite seulement, un marqueur de passage propre à la
cohorte, à la liste et à la règle est écrit : la confirmation ne peut pas être relancée, même sous
un autre préfixe ou après un échec (états commencé, terminé, échoué conservés).

    python scripts/evaluate_nsrr_ahi.py --cohort mesa --embeddings embeddings/nsrr_mesa \\
        --annotations-dir data/nsrr/mesa/annotations-events-nsrr --out results/nsrr_ahi_mesa
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embed_nsrr import PREPARATION_CODE, expected_contract, subject_list  # noqa: E402
from scripts.estimate_respiratory_ahi import MODELS, bootstrap, evaluate_test  # noqa: E402
from src.data.ahi import AHI_CUTOFF, MIN_SLEEP_H, positive_rate  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.nsrr_annotations import parse_nsrr_xml, reference_ahi_nsrr  # noqa: E402
from src.data.nsrr_cache import EPOCH_S, check_cache  # noqa: E402
from src.data.ucddb_annotations import parse_ucddb  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FROZEN_AHI_SHA256 = "eda69eb95c167a1df729b4c728e744ef0275a4ec725b7ace2727f40954bd2990"
RESPIRATORY_RULE = "physionet_median_v1"
SUCCESS_LOWER_BOUND = 0.60
PLAN = ROOT / "docs/plans/Plan confirmation IAH NSRR SleepFM OSF 13-11.md"
CODE = tuple(
    sorted(
        {
            "scripts/evaluate_nsrr_ahi.py",
            "scripts/estimate_respiratory_ahi.py",
            "src/data/ahi.py",
            "src/data/nsrr_annotations.py",
            "src/data/ucddb_annotations.py",
            *PREPARATION_CODE,
        }
    )
)
# Listes figées le 13 novembre depuis l'index public NSRR (commit 0136eab).
APPROVED_SUBJECTS = {
    "mesa": (
        ROOT / "configs/nsrr_mesa_subjects.json",
        "aef9d758bca2f478642031fbe0c9f2869facf88bedc873e00c00ddcc122a1fe7",
    ),
    "shhs1": (
        ROOT / "configs/nsrr_shhs1_subjects.json",
        "c06453bd4e3c5ba69b3751739b72ab211991b0d673f0bd3a0d5b262ac9144baf",
    ),
    # Contrôle externe UCDDB, liste figée le 13 novembre depuis l'index public PhysioNet.
    "ucddb": (ROOT / "configs/ucddb_subjects.json", "ee768b212108e647f9dbf6b8aedb398fd82385da67e005c93cf3dfaf376a5411"),
}
OFFICIAL_ID_COLUMN = {"mesa": "mesaid", "shhs1": "nsrrid"}
# Contrôle externe UCDDB (plan validé le 13 novembre) : 25 nuits, descriptif seulement, sans verdict.
DESCRIPTIVE_COHORTS = frozenset({"ucddb"})
OFFICIAL_COLUMNS = ("ahi_a0h3", "ahi_a0h4")
# Politique fixée avant le passage : annotations dépassant la fin du signal de plus d'une époque = nuit exclue.
XML_OVERRUN_TOLERANCE_S = EPOCH_S
MARKER_DIR = ROOT / "results/nsrr_runs"


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


def check_embeddings(folder: Path, cohort: str, frozen: dict, subjects_path: Path, approved_sha: str) -> dict:
    """Liste approuvée, contrat complet et couverture exacte ; chaque cache relu. Aucune cible lue."""
    try:
        subjects, list_sha = subject_list(subjects_path, cohort)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    if list_sha != approved_sha:
        raise ValueError(f"{subjects_path} : liste différente de la liste approuvée")
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = expected_contract(
        cohort, list_sha, MODELS, frozen["checkpoints"]["sleepfm"], frozen["checkpoints"]["osf"], RESPIRATORY_RULE
    )
    wrong = sorted(k for k in expected.keys() | manifest.keys() if manifest.get(k) != expected.get(k))
    if wrong:
        raise ValueError(f"Embeddings NSRR non conformes au contrat : {wrong}")
    status = json.loads((folder / "run_status.json").read_text())
    done, rejected = status.get("done", []), status.get("rejected", {})
    if not status.get("complete") or status.get("missing"):
        raise ValueError("Extraction incomplète : terminer le run avant l'évaluation")
    if len(set(done)) != len(done):
        raise ValueError("Sujets en double dans l'extraction")
    if set(done) & set(rejected):
        raise ValueError("Sujets à la fois extraits et rejetés")
    covered, expected_set = set(done) | set(rejected), set(subjects)
    if covered != expected_set:
        raise ValueError(
            f"Couverture différente de la liste : absents {sorted(expected_set - covered)[:5]}, "
            f"étrangers {sorted(covered - expected_set)[:5]}"
        )
    for subject, reason in rejected.items():
        if not isinstance(reason, str) or not reason.strip() or not (folder / f"{subject}.skipped").exists():
            raise ValueError(f"{subject} : rejet sans motif")
    stray = sorted(p.stem for p in folder.glob("*.npz") if p.stem not in set(done))
    if stray:
        raise ValueError(f"NPZ hors extraction : {stray[:5]}")
    records = {}
    for subject in done:
        records[subject] = check_cache(folder, subject, MODELS)
    return {"subjects": subjects, "list_sha": list_sha, "done": done, "rejected": rejected, "records": records}


def uncommitted(paths) -> list[str]:
    """Fichiers modifiés ou non suivis par git parmi `paths` (tout si git est indisponible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "-z", "--", *map(str, paths)], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return [str(p) for p in paths]
    if out.returncode != 0:
        return [str(p) for p in paths]
    return [entry[3:] for entry in out.stdout.split("\0") if entry.strip()]


def annotation_files(cohort: str, folder: Path, pattern: str, subject: str) -> tuple[Path, ...]:
    if cohort == "ucddb":
        return folder / f"{subject}_stage.txt", folder / f"{subject}_respevt.txt"
    return (folder / pattern.format(subject=subject),)


def read_annotations(cohort: str, files: tuple[Path, ...], record: dict) -> dict:
    if cohort == "ucddb":
        return parse_ucddb(*files, start_s=record["edf_start_s"])
    return parse_nsrr_xml(files[0])


def annotation_problems(cohort: str, files: tuple[Path, ...], record: dict) -> str | None:
    """Motif d'exclusion d'annotations illisibles ; erreur si un concept est inconnu (à décider avant)."""
    try:
        parsed = read_annotations(cohort, files, record)
    except (OSError, ValueError) as exc:
        return f"annotations : {exc}"
    if parsed["unknown_concepts"]:
        raise ValueError(f"{files[-1].name} : concepts inconnus {sorted(parsed['unknown_concepts'])}")
    return None


def subject_number(subject: str) -> int:
    """`mesa-sleep-0001` -> 1, `shhs1-200001` -> 200001 : identifiant des CSV NSRR (mesaid, nsrrid)."""
    match = re.fullmatch(r"(?:mesa-sleep|shhs1)-(\d+)", subject)
    if match is None:
        raise ValueError(f"Sujet NSRR inattendu : {subject}")
    return int(match.group(1))


def official_reference(path: Path, id_column: str, columns=OFFICIAL_COLUMNS) -> dict[str, dict[int, float]]:
    """IAH officiels par identifiant ; vide = manquant, négatif ou non numérique = erreur."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(c not in rows[0] for c in (id_column, *columns)):
        raise ValueError(f"{path} : colonnes {id_column} et {list(columns)} attendues")
    out = {column: {} for column in columns}
    seen = set()
    for row in rows:
        raw_id = row[id_column].strip()
        if not re.fullmatch(r"\d+", raw_id):
            raise ValueError(f"{path} : identifiant non entier {raw_id!r}")
        key = int(raw_id)
        if key in seen:
            raise ValueError(f"{path} : identifiant {key} en double")
        seen.add(key)
        for column in columns:
            text = row[column].strip()
            if not text:
                continue
            value = float(text)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{path} : {column} invalide pour {key} ({text})")
            out[column][key] = value
    return out


def night_row(npz_path: Path, parsed_or_error, frozen: dict, n_epochs: int) -> dict:
    """Valeurs d'une nuit, ou motif d'exclusion (`excluded`). Le cache a été contrôlé avant.

    `parsed_or_error` : fonction sans argument qui lit les annotations de la nuit.
    """
    try:
        parsed = parsed_or_error()
        reference = reference_ahi_nsrr(parsed)
    except (OSError, ValueError) as exc:
        return {"excluded": f"annotations : {exc}"}
    if parsed["recording_end_s"] > n_epochs * EPOCH_S + XML_OVERRUN_TOLERANCE_S:
        end, signal = parsed["recording_end_s"], n_epochs * EPOCH_S
        return {"excluded": f"annotations jusqu'à {end:.0f} s pour {signal:.0f} s de signal"}
    if reference["sleep_h"] < MIN_SLEEP_H:
        return {"excluded": f"sommeil {reference['sleep_h']:.2f} h < {MIN_SLEEP_H:g} h"}
    with np.load(npz_path) as data:
        row = {"reference_ahi": reference["ahi"], "sleep_h": reference["sleep_h"], "windows": n_epochs}
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


def against_official(night: dict, official: np.ndarray, calibration: dict) -> dict:
    """Mêmes métriques que la référence principale, sur les nuits ayant un IAH officiel."""
    known = np.isfinite(official)
    out = {
        "nights": int(known.sum()),
        "missing": night["subject"][~known].tolist(),
        "positive_15": int((official[known] >= AHI_CUTOFF).sum()),
    }
    truth = official[known] >= AHI_CUTOFF
    if known.sum() < 2 or truth.all() or not truth.any():
        out["not_computable"] = "moins de 2 nuits ou une seule classe IAH"
        return out
    reference_night = {**night, "reference_ahi": official}
    results, estimates = evaluate_test(reference_night, calibration, known)
    out["results"] = results
    out["bootstrap"] = bootstrap(reference_night, estimates, known)
    out["agreement_with_primary_15"] = float(np.mean((night["reference_ahi"][known] >= AHI_CUTOFF) == truth))
    return out


def precheck(args) -> dict:
    """Tout ce qui peut être vérifié sans lire une cible ni prédire. Lève une erreur sinon."""
    outputs = {ext: Path(f"{args.out}.{ext}") for ext in ("json", "nights.npz", "code.zip")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Sortie existante : la confirmation ne se lance qu'une fois")
    if not args.plan.exists():
        raise ValueError(f"Plan absent : {args.plan}")
    dirty = uncommitted([*CODE, args.plan.relative_to(ROOT) if args.plan.is_relative_to(ROOT) else args.plan])
    if dirty:
        raise ValueError(f"Code ou plan non commités : {dirty}")
    frozen = load_frozen(args.frozen_ahi, args.baseline, args.comparison, args.expected_sha)
    subjects_path, approved_sha = args.approved_subjects
    extraction = check_embeddings(args.embeddings, args.cohort, frozen, subjects_path, approved_sha)
    xml_paths = {
        s: annotation_files(args.cohort, args.annotations_dir, args.xml_pattern, s) for s in extraction["done"]
    }
    absent = sorted(s for s, paths in xml_paths.items() if not all(p.exists() for p in paths))
    if absent:
        raise ValueError(f"Annotations absentes : {absent[:5]}")
    unreadable = {}
    for subject, paths in xml_paths.items():
        problem = annotation_problems(args.cohort, paths, extraction["records"][subject])
        if problem:
            unreadable[subject] = problem
    official = None
    if args.official_csv is not None and args.cohort not in OFFICIAL_ID_COLUMN:
        raise ValueError(f"{args.cohort} : pas d'IAH officiel prévu")
    if args.official_csv is not None:
        official = official_reference(args.official_csv, OFFICIAL_ID_COLUMN[args.cohort])
        listed = {subject_number(s) for s in extraction["done"]}
        if not any(listed & set(values) for values in official.values()):
            raise ValueError(f"{args.official_csv} : aucun sujet de l'extraction n'a d'IAH officiel")
    marker = args.marker_dir / f"{args.cohort}-{approved_sha[:12]}-{args.expected_sha[:12]}.json"
    if marker.exists():
        raise ValueError(f"Confirmation déjà lancée ({marker.name}) : un seul passage")
    return {
        "outputs": outputs,
        "frozen": frozen,
        "extraction": extraction,
        "xml_paths": xml_paths,
        "unreadable": unreadable,
        "official": official,
        "marker": marker,
    }


def run(args) -> dict:
    checked = precheck(args)
    marker = checked["marker"]
    state = {
        "state": "started",
        "cohort": args.cohort,
        "out": str(args.out),
        "subjects_sha256": checked["extraction"]["list_sha"],
        "frozen_ahi_sha256": args.expected_sha,
        "plan_sha256": file_sha256(args.plan),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x") as handle:  # échoue si un autre passage l'a créé entre-temps
        json.dump(state, handle)
    try:
        report = evaluate(args, checked)
    except BaseException as exc:
        write_json_atomic(marker, state | {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    write_json_atomic(marker, state | {"state": "completed", "report_sha256": file_sha256(checked["outputs"]["json"])})
    return report


def evaluate(args, checked: dict) -> dict:
    frozen, extraction, outputs = checked["frozen"], checked["extraction"], checked["outputs"]
    rows = {}
    excluded = dict(extraction["rejected"]) | checked["unreadable"]
    for subject in extraction["done"]:
        if subject in checked["unreadable"]:
            continue
        record = extraction["records"][subject]
        files = checked["xml_paths"][subject]
        row = night_row(
            args.embeddings / f"{subject}.npz",
            lambda files=files, record=record: read_annotations(args.cohort, files, record),
            frozen,
            record["n_epochs"],
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
            "subjects": extraction["list_sha"],
            "caches": {s: extraction["records"][s]["npz_sha256"] for s in extraction["done"]},
        },
        "nights": {
            "listed": len(extraction["subjects"]),
            "evaluated": len(subjects),
            "positive_15": int(truth.sum()),
            "excluded": excluded,
        },
        "policy": {"xml_overrun_tolerance_s": XML_OVERRUN_TOLERANCE_S, "min_sleep_h": MIN_SLEEP_H},
        "thresholds": frozen["thresholds"],
        "calibration": frozen["calibration"],
        "results": results,
        "bootstrap": uncertainty,
        "success_criterion": {
            "lower_bound": SUCCESS_LOWER_BOUND,
            "applies": args.cohort not in DESCRIPTIVE_COHORTS,
            "models": criterion(uncertainty),
        },
        "environment": run_environment(),
        "command": sys.argv,
    }
    for model in MODELS:
        night[f"estimated_ahi_{model}"] = estimates[model]
    if checked["official"] is not None:
        report["official_reference"] = {
            "csv_sha256": file_sha256(args.official_csv),
            "id_column": OFFICIAL_ID_COLUMN[args.cohort],
        }
        for column, values in checked["official"].items():
            official = np.array([values.get(subject_number(s), np.nan) for s in subjects], dtype=float)
            night[f"official_{column}"] = official
            report["official_reference"][column] = against_official(night, official, frozen["calibration"])
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
            + (
                "descriptif, pas de verdict"
                if args.cohort in DESCRIPTIVE_COHORTS
                else f"critère {'atteint' if verdict['success'] else 'non atteint'}"
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=sorted(APPROVED_SUBJECTS), required=True)
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
    parser.add_argument("--official-csv", type=Path, default=None, help="CSV NSRR (ahi_a0h3 et ahi_a0h4 lus ensemble)")
    args = parser.parse_args()
    args.expected_sha = FROZEN_AHI_SHA256
    args.approved_subjects = APPROVED_SUBJECTS[args.cohort]
    args.marker_dir = MARKER_DIR
    run(args)


if __name__ == "__main__":
    main()
