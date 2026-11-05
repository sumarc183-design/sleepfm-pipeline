"""Prédit l'hypnogramme d'une ou plusieurs nuits avec des sondes exportées.

Lit des sondes exportées par `run_baseline.py --save-models DIR` et des
embeddings en cache (`<sujet>.npz` du mode par lots), vérifie qu'ils viennent
du même checkpoint et du même prétraitement, puis écrit un CSV par nuit :
numéro d'époque, début (s), stade prédit et probabilités, pour chaque sonde.

    python scripts/predict_night.py --models models/physionet2018 \
        --embeddings-dir embeddings/physionet2018 tr03-0005 --out predictions/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402
from src.models.linear_probe import predict_probabilities  # noqa: E402
from src.models.probe_export import load_probes  # noqa: E402

LABEL_TO_STAGE = {v: k for k, v in STAGE_NAME_TO_LABEL.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.embeddings_dir / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    models, _ = load_probes(args.models, cache_manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        X, _, groups = load_cached_embeddings(args.embeddings_dir, [subject])
        order = np.argsort(groups[1])
        columns = {"epoch_idx": groups[1][order], "onset_s": groups[2][order]}
        for kind, model in models.items():
            proba = predict_probabilities(model, X, groups)[order]
            columns[f"{kind}_stade"] = [LABEL_TO_STAGE[int(c)] for c in model.classes_[proba.argmax(1)]]
            for j, c in enumerate(model.classes_):
                columns[f"{kind}_p_{LABEL_TO_STAGE[int(c)]}"] = np.round(proba[:, j], 4)
        path = args.out / f"{subject}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values(), strict=True))
        print(f"{subject} : {len(order)} époques -> {path}")


if __name__ == "__main__":
    main()
