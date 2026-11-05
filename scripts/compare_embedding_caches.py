"""Compare deux dossiers d'embeddings sur leurs sujets communs.

Usage prévu : vérifier que le prétraitement version 2 (début utile pris sur
la fin de la période « undefined ») redonne, pour les 100 sujets déjà
calculés en version 1, les mêmes époques, les mêmes stades et les mêmes
embeddings. Sujet par sujet : nombre d'époques, stades identiques (alignés
par numéro d'époque), écart maximal et cosinus minimal entre embeddings.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--atol", type=float, default=1e-3, help="écart maximal toléré entre embeddings")
    args = parser.parse_args()

    for directory in (args.old, args.new):
        manifest = directory / "manifest.json"
        version = json.loads(manifest.read_text()).get("preprocessing_version") if manifest.exists() else "?"
        print(f"{directory} : {len(list(directory.glob('*.npz')))} sujet(s), prétraitement version {version}")
    old = {p.name.removesuffix(".npz") for p in args.old.glob("*.npz")}
    new = {p.name.removesuffix(".npz") for p in args.new.glob("*.npz")}
    common = sorted(old & new)
    print(f"{len(common)} sujet(s) en commun ; {len(new - old)} nouveau(x) ; {len(old - new)} absent(s) du nouveau")

    identical, differ = 0, []
    for subject in common:
        X_a, y_a, idx_a, _ = load_one(args.old, subject)
        X_b, y_b, idx_b, onsets_b = load_one(args.new, subject)
        if len(y_a) != len(y_b) or not np.array_equal(idx_a, idx_b):
            differ.append(f"{subject} : {len(y_a)} époques contre {len(y_b)}")
            continue
        if not np.array_equal(y_a, y_b):
            differ.append(f"{subject} : {int((y_a != y_b).sum())} stade(s) différent(s)")
            continue
        gap = float(np.abs(X_a - X_b).max())
        cosine = float(
            (np.sum(X_a * X_b, 1) / (np.linalg.norm(X_a, axis=1) * np.linalg.norm(X_b, axis=1) + 1e-12)).min()
        )
        if gap > args.atol:
            differ.append(f"{subject} : embeddings différents (écart max {gap:.2e}, cosinus min {cosine:.4f})")
            continue
        identical += 1
        if np.isfinite(onsets_b).all() and (np.diff(onsets_b) > 31).any():
            print(f"  {subject} : {(np.diff(onsets_b) > 31).sum()} trou(s) d'enregistrement en cours de nuit")
    print(f"Identiques (mêmes époques, mêmes stades, embeddings à {args.atol} près) : {identical} / {len(common)}")
    for line in differ:
        print(f"  DIFFÉRENT {line}")


if __name__ == "__main__":
    main()
