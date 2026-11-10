"""Vérifie le checkpoint préentraîné OSF (`osf_backbone.pth`) sans données patient.

Charge les poids publiés dans notre backbone en mode strict (aucune clé manquante
ou inattendue), lit leurs métadonnées, puis fait un passage avant sur un tenseur
synthétique pour vérifier la forme et la finitude des sorties. Écrit un rapport
de provenance : empreinte du fichier, métadonnées, nombre de paramètres, formes.
Aucun modèle n'est évalué : ce contrôle ne dit rien des performances d'OSF.

    python scripts/check_osf_checkpoint.py third_party/osf/osf_backbone.pth \\
        --source https://huggingface.co/yang-ai-lab/OSF-Base --report results/osf_checkpoint_check.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.models.osf_vit1d import load_osf_backbone  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402


def check_checkpoint(path: Path, seed: int = 0) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    model = load_osf_backbone(path)  # strict : refuse toute clé manquante ou inattendue
    generator = torch.Generator().manual_seed(seed)
    series = torch.randn(2, meta["num_leads"], meta["seq_len"], generator=generator)
    with torch.no_grad():
        cls, patches = model.forward_encoding(series)
    if not (torch.isfinite(cls).all() and torch.isfinite(patches).all()):
        raise ValueError("Sorties OSF non finies")
    return {
        "checkpoint_sha256": file_sha256(path),
        "metadata": meta,
        "state_dict_keys": len(payload["state_dict"]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "strict_load": True,
        "synthetic_forward": {
            "input_shape": list(series.shape),
            "cls_shape": list(cls.shape),
            "patches_shape": list(patches.shape),
            "finite": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", default="https://huggingface.co/yang-ai-lab/OSF-Base")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "Chargement strict et passage avant sur tenseur synthétique ; aucune donnée patient, aucun score.",
        "source": args.source,
        **check_checkpoint(args.checkpoint),
        "environment": run_environment(),
    }
    write_json_atomic(args.report, report)
    shape = report["synthetic_forward"]["cls_shape"]
    print(f"OSF : {report['parameters']} paramètres, chargement strict, sorties {shape}")
    print(f"Rapport : {args.report}")


if __name__ == "__main__":
    main()
