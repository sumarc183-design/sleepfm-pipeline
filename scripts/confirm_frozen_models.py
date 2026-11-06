"""Confirmation des modèles figés sur les sujets réservés : un seul passage, sans entraînement.

Suit docs/PROTOCOLE_CONFIRMATION.md. Avant toute prédiction, tout est vérifié, et le
moindre écart arrête le run sans afficher de score :
- empreintes des modèles, de la liste des sujets réservés et du découpage figé
  (configs/confirmation_physionet2018.json) ;
- sujets réservés sans intersection avec le découpage ;
- contrat du cache identique à celui du protocole ;
- dossier d'embeddings : chaque sujet réservé a un seul résultat (cache valide ou
  rejet de l'extraction avec sa raison), et aucun sujet étranger ;
- code au commit annoncé, sans modification ;
- pas de résultat déjà écrit sous ce nom.

Les rejets automatiques sont les seules exclusions, listés avec leur raison. Sorties
<out>.json (état, critère, métriques, rejets, environnement) et <out>.predictions.npz.

Répétition sur des sujets déjà ouverts, pour tester la commande sans toucher aux sujets
réservés : --rehearsal tr03-0052 ... Le résultat est marqué comme une répétition.

    python scripts/confirm_frozen_models.py --embeddings-dir embeddings/physionet2018_reserved_v2 \
        --out results/confirmation_physionet2018
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import cache_problem, load_cached_embeddings  # noqa: E402
from src.data.cache_contract import contract_differences, file_sha256, write_json_atomic  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402
from src.models.linear_probe import (  # noqa: E402
    evaluate_probe,
    paired_bootstrap_diff,
    predict_labels,
    predict_probabilities,
)
from src.models.probe_export import load_probes, run_environment  # noqa: E402

BASELINE, CANDIDATE = "logistic", "sequence"
LABEL_TO_STAGE = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}


def frozen_inputs_problems(config: dict, models_dir: Path, reserved_file: Path, split_file: Path) -> list[str]:
    """Écarts avec ce que le protocole a figé."""
    problems = []
    for kind in (BASELINE, CANDIDATE):
        path = models_dir / f"{kind}.joblib"
        if not path.exists() or file_sha256(path) != config["models"][kind]:
            problems.append(f"modèle {kind} absent ou modifié")
    if file_sha256(reserved_file) != config["reserved_sha256"]:
        problems.append("liste des sujets réservés modifiée")
    if file_sha256(split_file) != config["split_sha256"]:
        problems.append("découpage figé modifié")
    return problems


def subjects_problems(subjects: list[str], split: dict, embeddings_dir: Path, rehearsal: bool) -> list[str]:
    """Couverture exacte du dossier : un seul résultat par sujet, cache valide, pas d'étranger."""
    problems = []
    if len(set(subjects)) != len(subjects):
        problems.append("sujets en double dans la liste")
    cached = {p.name.removesuffix(".npz") for p in embeddings_dir.glob("*.npz")}
    rejected = {p.name.removesuffix(".skipped") for p in embeddings_dir.glob("*.skipped")}
    requested = set(subjects)
    if not rehearsal:
        opened = requested & (set(split["train"]) | set(split["test"]))
        if opened:
            problems.append(f"{len(opened)} sujet(s) réservé(s) présents dans le découpage")
        foreign = (cached | rejected) - requested
        if foreign:
            problems.append(f"{len(foreign)} sujet(s) étranger(s) dans le dossier, dont {sorted(foreign)[0]}")
    missing = requested - cached - rejected
    if missing:
        problems.append(f"{len(missing)} sujet(s) sans résultat, dont {sorted(missing)[0]}")
    both = requested & cached & rejected
    if both:
        problems.append(f"{len(both)} sujet(s) à la fois en cache et rejetés, dont {sorted(both)[0]}")
    for subject in sorted(requested & cached - both):
        problem = cache_problem(embeddings_dir / f"{subject}.npz", strict_times=True)
        if problem:
            problems.append(f"{subject} : {problem}")
    return problems


def pip_freeze() -> list[str] | None:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except Exception:
        return None


def run_confirmation(
    config_path: Path,
    models_dir: Path,
    embeddings_dir: Path,
    reserved_file: Path,
    split_file: Path,
    out: Path,
    rehearsal: list[str] | None = None,
    require_clean: bool = True,
) -> dict:
    json_path, predictions_path = Path(f"{out}.json"), Path(f"{out}.predictions.npz")
    if json_path.exists() or predictions_path.exists():
        raise SystemExit(f"{json_path} existe déjà : un seul passage. Rien n'a été calculé.")
    config = json.loads(config_path.read_text())
    split = json.loads(split_file.read_text())
    reserved = json.loads(reserved_file.read_text())["subjects"]
    is_rehearsal = rehearsal is not None
    if is_rehearsal and set(rehearsal) & set(reserved):
        raise SystemExit("La répétition ne doit utiliser aucun sujet réservé. Rien n'a été calculé.")
    if is_rehearsal and not set(rehearsal) <= set(split["train"]) | set(split["test"]):
        raise SystemExit("La répétition doit utiliser des sujets déjà ouverts du découpage. Rien n'a été calculé.")
    subjects = rehearsal if is_rehearsal else reserved

    problems = frozen_inputs_problems(config, models_dir, reserved_file, split_file)
    if set(reserved) & (set(split["train"]) | set(split["test"])):
        problems.append("sujets réservés présents dans le découpage")
    manifest_path = embeddings_dir / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if cache_manifest is None:
        problems.append("dossier d'embeddings sans manifeste")
    else:
        differs = contract_differences(cache_manifest, config["contract"])
        if differs:
            problems.append(f"contrat du cache différent ({', '.join(differs)})")
    problems += subjects_problems(subjects, split, embeddings_dir, is_rehearsal)
    environment = run_environment()
    if require_clean and not is_rehearsal and (environment["code_modified"] is not False):
        problems.append("code modifié ou commit inconnu")
    if problems:
        raise SystemExit("Confirmation refusée, aucun score calculé :\n- " + "\n- ".join(problems))

    report = {
        "status": "en cours",
        "kind": "répétition sur des sujets déjà ouverts" if is_rehearsal else "confirmation sur les sujets réservés",
        "protocol_version": config["protocol_version"],
        "config_sha256": file_sha256(config_path),
        "models": config["models"],
        "reserved_sha256": config["reserved_sha256"],
        "split_sha256": config["split_sha256"],
        "embeddings_manifest": cache_manifest,
        "command": sys.argv,
        "environment": environment,
        "pip_freeze": pip_freeze(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(json_path, report)  # trace le passage avant le premier score

    rejected = {
        s: (embeddings_dir / f"{s}.skipped").read_text().strip()
        for s in subjects
        if not (embeddings_dir / f"{s}.npz").exists()
    }
    evaluated = [s for s in subjects if s not in rejected]
    models, _ = load_probes(models_dir, cache_manifest)
    X, y, groups = load_cached_embeddings(embeddings_dir, evaluated)
    subject_rows = groups[0]

    saved = {"y_true": y, "subject": subject_rows, "epoch_idx": groups[1], "onset_s": groups[2]}
    metrics = {}
    predictions = {}
    for kind in (BASELINE, CANDIDATE):
        model = models[kind]
        predictions[kind] = predict_labels(model, X, groups)
        saved[f"pred_{kind}"] = predictions[kind]
        saved[f"proba_{kind}"] = predict_probabilities(model, X, groups).astype(np.float32)
        saved[f"classes_{kind}"] = model.classes_
        metrics[kind] = evaluate_probe(model, X, y, groups, with_ci=True)
        metrics[kind]["f1_by_stage"] = {
            LABEL_TO_STAGE[int(c)]: float(f1_score(y, predictions[kind], labels=[c], average="macro"))
            for c in model.classes_
        }

    bootstrap = config["bootstrap"]
    differences = paired_bootstrap_diff(
        y,
        predictions[BASELINE],
        predictions[CANDIDATE],
        subject_rows,
        n_boot=bootstrap["n_boot"],
        seed=bootstrap["seed"],
    )
    primary = differences["cohen_kappa"]
    confirmed = bool(primary["ci95"][0] > 0)
    prefixes = sorted({s.split("-")[0] for s in evaluated})
    by_prefix = {}
    for prefix in prefixes:
        rows = np.char.startswith(subject_rows.astype(str), f"{prefix}-")
        by_prefix[prefix] = {
            "nights": int(len(set(subject_rows[rows].tolist()))),
            **{kind: float(cohen_kappa_score(y[rows], predictions[kind][rows])) for kind in predictions},
        }

    temporary = predictions_path.with_name(f".{predictions_path.stem}.tmp.npz")
    np.savez_compressed(temporary, **saved)
    os.replace(temporary, predictions_path)
    report |= {
        "status": "terminé",
        "n_requested": len(subjects),
        "n_evaluated": len(evaluated),
        "n_epochs": len(y),
        "rejected": rejected,
        "primary": {"kappa_candidate_minus_baseline": primary, "gain_confirmed": confirmed and not is_rehearsal},
        "secondary": {"macro_f1_difference": differences["macro_f1"], "metrics": metrics, "kappa_by_prefix": by_prefix},
        "predictions": predictions_path.name,
    }
    write_json_atomic(json_path, json.loads(json.dumps(report, default=_json_default)))
    print(report["kind"])
    print(f"{len(evaluated)} nuits évaluées sur {len(subjects)}, {len(rejected)} rejetée(s), {len(y)} époques")
    print(f"Baseline : κ {metrics[BASELINE]['cohen_kappa']:.4f}, BiGRU : κ {metrics[CANDIDATE]['cohen_kappa']:.4f}")
    verdict = "confirmé" if confirmed else "non confirmé"
    if is_rehearsal:
        verdict += " sur cette répétition, ce qui ne confirme rien : ces sujets sont déjà ouverts"
    low, high = primary["ci95"]
    print(f"Critère principal, écart de κ : {primary['diff']:+.4f} [{low:+.4f} ; {high:+.4f}] : gain {verdict}")
    print(f"Résultats : {json_path}, {predictions_path}")
    return report


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/confirmation_physionet2018.json"))
    parser.add_argument("--models", type=Path, default=Path("models/physionet2018_500"))
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--reserved-file", type=Path, default=Path("configs/reserved_physionet2018.json"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--out", type=Path, required=True, help="préfixe des sorties, sans extension")
    parser.add_argument("--rehearsal", nargs="+", default=None, help="sujets déjà ouverts : teste la commande")
    args = parser.parse_args()
    run_confirmation(
        args.config, args.models, args.embeddings_dir, args.reserved_file, args.split_file, args.out, args.rehearsal
    )


if __name__ == "__main__":
    main()
