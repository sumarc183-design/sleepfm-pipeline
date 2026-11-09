"""Consolide les métriques déjà enregistrées, sans modèle ni recalcul de score.

Accepte les JSON de run_baseline.py et de confirmation terminée. Les champs
historiques manquants restent absents ; les runs ne sont ni classés ni agrégés.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

DEFAULT_SOURCES = [
    Path("results/results_physionet2018_100_auprc.json"),
    Path("results/results_physionet2018_200.json"),
    Path("results/results_physionet2018_500.json"),
    Path("results/confirmation_physionet2018.json"),
]
METRICS = ("accuracy", "cohen_kappa", "macro_f1", "roc_auc_macro", "auprc_macro")
FIELDS = (
    "source",
    "source_sha256",
    "kind",
    "probe",
    "train_nights",
    "test_nights",
    "test_epochs",
    "seed",
    "preprocessing_version",
    "checkpoint_sha256",
    *METRICS,
    "kappa_ci95_low",
    "kappa_ci95_high",
)


def metric_value(value, key: str):
    if value is None:
        return None
    lower = -1 if key == "cohen_kappa" else 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} : valeur numérique finie attendue")
    if not lower <= value <= 1:
        raise ValueError(f"{key} : valeur hors de [{lower}, 1]")
    return value


def subject_count(value, key: str) -> int:
    if not isinstance(value, list) or not value or not all(isinstance(s, str) and s for s in value):
        raise ValueError(f"{key} : liste non vide de sujets attendue")
    if len(set(value)) != len(value):
        raise ValueError(f"{key} : sujets en double")
    return len(value)


def epoch_count(metrics: dict) -> int | None:
    matrix = metrics.get("confusion_matrix")
    if matrix is None:
        return None
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("Matrice de confusion vide ou invalide")
    if any(not isinstance(row, list) or len(row) != len(matrix) for row in matrix):
        raise ValueError("Matrice de confusion non carrée")
    counts = [n for row in matrix for n in row]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts) or not sum(counts):
        raise ValueError("Effectifs de la matrice de confusion invalides")
    return sum(counts)


def read_result(path: Path) -> list[dict]:
    """Une ligne par sonde, avec empreinte du fichier réellement lu."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Objet JSON attendu")
    if payload.get("status") not in (None, "terminé"):
        raise ValueError("Run non terminé")
    confirmation = "secondary" in payload
    if confirmation:
        if payload.get("status") != "terminé":
            raise ValueError("Statut terminé obligatoire pour une confirmation ou répétition")
        kind = payload.get("kind")
        if kind not in ("confirmation sur les sujets réservés", "répétition sur des sujets déjà ouverts"):
            raise ValueError("Type de confirmation ou répétition inconnu")
        train_nights = None  # L'absence de liste n'autorise pas à inventer l'effectif d'entraînement.
        test_nights = payload.get("n_evaluated")
        requested = payload.get("n_requested")
        if (
            type(test_nights) is not int
            or type(requested) is not int
            or not 0 < test_nights <= requested
            or type(payload.get("n_epochs")) is not int
            or payload["n_epochs"] <= 0
        ):
            raise ValueError("Effectifs de confirmation invalides")
        probes = payload["secondary"].get("metrics")
    else:
        kind = "évaluation enregistrée"
        train_nights = subject_count(payload.get("train_subjects"), "train_subjects")
        test_nights = subject_count(payload.get("test_subjects"), "test_subjects")
        if set(payload["train_subjects"]) & set(payload["test_subjects"]):
            raise ValueError("Chevauchement des sujets train/test")
        probes = payload.get("probes")
    if not isinstance(probes, dict) or not probes:
        raise ValueError("Aucune sonde enregistrée")
    contract = payload.get("embeddings_manifest") or {}
    rows, counts_seen = [], set()
    for probe, result in sorted(probes.items()):
        metrics = result if confirmation else result.get("test")
        if not isinstance(metrics, dict):
            raise ValueError(f"{probe} : métriques de test absentes")
        count = epoch_count(metrics)
        if count is not None:
            counts_seen.add(count)
            if confirmation and count != payload["n_epochs"]:
                raise ValueError("Effectifs de la matrice différents du nombre d'époques déclaré")
        ci = metrics.get("ci95") or {}
        if ci.get("n_subjects", test_nights) != test_nights:
            raise ValueError("Effectif des IC différent du nombre de nuits de test")
        bounds = ci.get("cohen_kappa")
        low = high = None
        if bounds is not None:
            if not isinstance(bounds, list) or len(bounds) != 2 or any(v is None for v in bounds):
                raise ValueError("IC du kappa invalide")
            low, high = [metric_value(v, "cohen_kappa") for v in bounds]
            if low > high:
                raise ValueError("Bornes de l'IC du kappa inversées")
        rows.append(
            {
                "source": str(path),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "kind": kind,
                "probe": probe,
                "train_nights": train_nights,
                "test_nights": test_nights,
                "test_epochs": payload["n_epochs"] if confirmation else count,
                "seed": payload.get("seed"),
                "preprocessing_version": contract.get("preprocessing_version"),
                "checkpoint_sha256": contract.get("checkpoint_sha256"),
                **{key: metric_value(metrics.get(key), key) for key in METRICS},
                "kappa_ci95_low": low,
                "kappa_ci95_high": high,
            }
        )
    if len(counts_seen) > 1:
        raise ValueError("Les sondes n'ont pas le même nombre d'époques de test")
    return rows


def consolidate(paths: list[Path]) -> list[dict]:
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Sources vides ou présentes plusieurs fois")
    rows = []
    for path in paths:
        try:
            rows.extend(read_result(path))
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"{path} : {exc}") from exc
    return rows


def render_csv(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_markdown(rows: list[dict]) -> str:
    def value(v):
        return "non renseigné" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    def cell(v):
        return value(v).replace("|", "\\|").replace("\n", " ")

    sources = list(dict.fromkeys(row["source"] for row in rows))
    lines = [
        "# Résultats consolidés enregistrés",
        "",
        "Lecture des JSON sauvegardés, sans entraînement ni recalcul des scores. "
        "Les cohortes de test et les prétraitements peuvent différer : ces lignes ne forment pas un classement.",
        "",
        "Les champs historiques absents restent « non renseigné ». "
        "La confirmation est conservée séparément des évaluations de développement et des répétitions.",
        "",
        "| Source | Sonde | Nuits train | Nuits test | Époques test | κ [IC95 %] | F1 macro | ROC-AUC | AUPRC |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        kappa = value(row["cohen_kappa"])
        if row["kappa_ci95_low"] is not None:
            kappa += f" [{value(row['kappa_ci95_low'])} ; {value(row['kappa_ci95_high'])}]"
        cells = [
            f"R{sources.index(row['source']) + 1}",
            row["probe"],
            row["train_nights"],
            row["test_nights"],
            row["test_epochs"],
            kappa,
            row["macro_f1"],
            row["roc_auc_macro"],
            row["auprc_macro"],
        ]
        lines.append("| " + " | ".join(cell(v) for v in cells) + " |")
    lines.extend(["", "## Sources", ""])
    for i, source in enumerate(sources, 1):
        row = next(r for r in rows if r["source"] == source)
        lines.append(
            f"- R{i} : `{cell(source)}` ; {cell(row['kind'])} ; "
            f"prétraitement {value(row['preprocessing_version'])} ; "
            f"graine {value(row['seed'])} ; SHA-256 `{row['source_sha256']}`."
        )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(content)
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out", type=Path, required=True, help="Tableau CSV ; valeurs numériques sans arrondi")
    parser.add_argument("--markdown", type=Path, help="Tableau lisible, arrondi à trois décimales")
    args = parser.parse_args()
    outputs = [args.out] + ([args.markdown] if args.markdown else [])
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        p.resolve() == source.resolve() for p in outputs for source in args.sources
    ):
        parser.error("Les sorties doivent être distinctes entre elles et des sources")
    try:
        rows = consolidate(args.sources)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    csv_text, markdown_text = render_csv(rows), render_markdown(rows)
    write_atomic(args.out, csv_text)
    if args.markdown:
        write_atomic(args.markdown, markdown_text)
    print(f"{len(args.sources)} sources, {len(rows)} lignes -> {args.out}")


if __name__ == "__main__":
    main()
