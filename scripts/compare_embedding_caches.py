"""Compare deux dossiers d'embeddings sur leurs sujets communs.

Usage prévu : vérifier que le prétraitement version 2 (début utile pris sur
la fin de la période « undefined ») redonne, pour les 100 sujets déjà
calculés en version 1, les mêmes époques, les mêmes stades et les mêmes
embeddings. Sujet par sujet : nombre d'époques, temps d'origine, stades
identiques (alignés par numéro d'époque), écart maximal et cosinus minimal
entre embeddings. Sans temps d'origine complets, l'identité reste non vérifiée.

    python scripts/compare_embedding_caches.py embeddings/physionet2018 embeddings/physionet2018_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings  # noqa: E402


def load_one(directory: Path, subject: str):
    X, y, (_, epoch_idx, onsets) = load_cached_embeddings(directory, [subject])
    order = np.argsort(epoch_idx)
    return X[order], y[order], epoch_idx[order], onsets[order]


def validate_tolerances(atol: float, onset_atol: float) -> None:
    if not np.isfinite([atol, onset_atol]).all() or min(atol, onset_atol) < 0:
        raise ValueError("Les tolérances doivent être finies, positives ou nulles")


def compare_one(old: Path, new: Path, subject: str, atol: float, onset_atol: float = 0.01) -> tuple[str, str]:
    """Distingue différence établie, identité vérifiée et temps non vérifiables."""
    validate_tolerances(atol, onset_atol)
    X_a, y_a, idx_a, onsets_a = load_one(old, subject)
    X_b, y_b, idx_b, onsets_b = load_one(new, subject)
    if len(y_a) != len(y_b) or not np.array_equal(idx_a, idx_b):
        return "différent", f"époques différentes ({len(y_a)} contre {len(y_b)})"
    if not np.array_equal(y_a, y_b):
        return "différent", f"{int((y_a != y_b).sum())} stade(s) différent(s)"
    gap = float(np.abs(X_a - X_b).max())
    cosine = float((np.sum(X_a * X_b, 1) / (np.linalg.norm(X_a, axis=1) * np.linalg.norm(X_b, axis=1) + 1e-12)).min())
    if gap > atol:
        return "différent", f"embeddings différents (écart max {gap:.2e}, cosinus min {cosine:.4f})"
    if not (np.isfinite(onsets_a).all() and np.isfinite(onsets_b).all()) or min(onsets_a.min(), onsets_b.min()) < 0:
        return "non vérifié", "embeddings compatibles, mais temps d'origine absents ou invalides"
    if not np.allclose(onsets_a, onsets_b, atol=onset_atol, rtol=0):
        return "différent", f"temps d'origine différents (écart max {float(np.abs(onsets_a - onsets_b).max()):.3f} s)"
    gaps = int((np.diff(onsets_b) > 31).sum())
    return "identique", f"{gaps} trou(s) d'enregistrement" if gaps else ""


def cache_subjects(directory: Path) -> set[str]:
    return {p.name.removesuffix(".npz") for p in directory.glob("*.npz") if not p.name.endswith(".tmp.npz")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--atol", type=float, default=1e-3, help="écart maximal toléré entre embeddings")
    parser.add_argument("--onset-atol", type=float, default=0.01, help="écart maximal entre temps d'origine (s)")
    args = parser.parse_args()
    try:
        validate_tolerances(args.atol, args.onset_atol)
    except ValueError as exc:
        parser.error(str(exc))

    for directory in (args.old, args.new):
        manifest = directory / "manifest.json"
        version = json.loads(manifest.read_text()).get("preprocessing_version") if manifest.exists() else "?"
        print(f"{directory} : {len(cache_subjects(directory))} sujet(s), prétraitement version {version}")
    old, new = cache_subjects(args.old), cache_subjects(args.new)
    common = sorted(old & new)
    print(f"{len(common)} sujet(s) en commun ; {len(new - old)} nouveau(x) ; {len(old - new)} absent(s) du nouveau")

    identical, unverified, differ = 0, [], []
    for subject in common:
        status, message = compare_one(args.old, args.new, subject, args.atol, args.onset_atol)
        if status == "différent":
            differ.append(f"{subject} : {message}")
        elif status == "non vérifié":
            unverified.append(f"{subject} : {message}")
        else:
            identical += 1
            if message:
                print(f"  {subject} : {message}")
    print(
        f"Identiques (mêmes époques, temps à {args.onset_atol} s près, "
        f"mêmes stades, embeddings à {args.atol} près) : {identical} / {len(common)}"
    )
    for line in differ:
        print(f"  DIFFÉRENT {line}")
    for line in unverified:
        print(f"  NON VÉRIFIÉ {line}")


if __name__ == "__main__":
    main()
