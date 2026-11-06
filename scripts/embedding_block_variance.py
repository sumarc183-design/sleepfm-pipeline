"""Le bloc respiration des embeddings bouge-t-il au cours d'une nuit ?

Pour chaque bloc (respiration, EEG/EOG, ECG), calcule la variance d'une nuit : la
somme sur les dimensions de la variance des époques de cette nuit, puis sa moyenne
sur les nuits. La part intra-nuit la compare à la variance des moyennes entre nuits.
Les nuits sont celles du test figé, triées par identifiant ; --nights en garde les N
premières. Compte aussi les époques prédites REM par sonde depuis une archive
d'ablation : ce comportement dépend de la sonde, pas seulement du bloc.

Description seulement, sur des nuits déjà examinées : ne donne pas la cause.

    python scripts/embedding_block_variance.py --embeddings-dir embeddings/physionet2018_v2 --nights 30 \
        --predictions results/ablation_modalites_physionet2018_500.predictions.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablation_modalites import BLOCK_DIM, BLOCKS, select_blocks  # noqa: E402
from src.data.epoch_dataset import STAGE_NAME_TO_LABEL  # noqa: E402


def block_variances(nights: list[np.ndarray]) -> dict[str, dict[str, float]]:
    """{bloc: variance intra-nuit moyenne, variance entre nuits, part intra-nuit}."""
    out = {}
    for block in BLOCKS:
        parts = [select_blocks(X, (block,)) for X in nights]
        within = float(np.mean([part.var(axis=0).sum() for part in parts]))
        between = float(np.array([part.mean(axis=0) for part in parts]).var(axis=0).sum())
        out[block] = {"within": within, "between": between, "within_share": within / (within + between)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--nights", type=int, default=None, help="les N premières nuits de test, par identifiant")
    parser.add_argument("--predictions", type=Path, default=None, help="archive d'ablation, pour compter REM")
    args = parser.parse_args()

    subjects = sorted(json.loads(args.split_file.read_text())["test"])[: args.nights]
    nights = []
    for subject in subjects:
        with np.load(args.embeddings_dir / f"{subject}.npz") as data:
            nights.append(data["X"])
    print(f"{len(subjects)} nuits de test ({subjects[0]} à {subjects[-1]}), blocs de {BLOCK_DIM} dimensions")
    for block, values in block_variances(nights).items():
        print(
            f"  {block:<5} intra-nuit {values['within']:.3f}  entre nuits {values['between']:.3f}  "
            f"part intra-nuit {values['within_share']:.3f}"
        )
    if args.predictions is not None:
        saved = np.load(args.predictions)
        rem = STAGE_NAME_TO_LABEL["rem"]
        print(f"Époques de test : {len(saved['y_true'])}, dont REM selon l'expert : {(saved['y_true'] == rem).sum()}")
        for name in ("pred_logistic_resp", "pred_sequence_resp"):
            print(f"  {name} : {(saved[name] == rem).sum()} époques prédites REM")


if __name__ == "__main__":
    main()
