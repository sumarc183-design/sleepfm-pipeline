"""Embeddings SleepFM et OSF des nuits NSRR (MESA, SHHS-1), pour la confirmation de l'IAH.

Ébauche du 13 novembre 2026, écrite avant l'accès aux données (plan de confirmation NSRR).
Pour chaque sujet de la liste figée : EDF -> époques au format PhysioNet (`nsrr_night`),
embeddings SleepFM (1536) et/ou OSF (768) sur la même nuit. Un <sujet>.npz par sujet, un
manifest.json (poids, cohorte, liste, règle d'échelle) et un run_status.json. Relancer reprend
où le run s'est arrêté ; une nuit refusée laisse un <sujet>.skipped motivé. Aucun label, aucun
score : l'IAH se calcule ensuite sur le Mac avec la règle figée.

SleepFM exige une règle d'échelle des sangles et du débit (`--respiratory-rule`) :
`physionet_median_v1`, validée par le tuteur le 13 novembre (médianes PhysioNet). OSF n'en a pas
besoin (z-score par nuit, insensible à l'échelle).

    python scripts/embed_nsrr.py --cohort mesa --edf-dir /kaggle/input/mesa-edf \\
        --subjects configs/nsrr_mesa_subjects.json --out embeddings/nsrr_mesa \\
        --checkpoint third_party/sleepfm/sleepfm/checkpoint/best.pt --respiratory-rule physionet_median_v1 \\
        --osf-checkpoint third_party/osf/osf_backbone.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embed_osf_physionet2018 import embed_night as embed_osf_night  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.data.nsrr_channels import physionet_median_scale  # noqa: E402
from src.data.nsrr_signals import nsrr_night  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

PREPARATION_VERSION = 1
SLEEPFM_DIM = 1536
# Règles d'échelle des sangles et du débit pour SleepFM (validée par le tuteur le 13 novembre 2026).
RESPIRATORY_RULES = {"physionet_median_v1": physionet_median_scale}


def identity_scale(target: str, signal: np.ndarray, unit: str) -> np.ndarray:
    """Pour OSF seul : le z-score par nuit rend l'échelle sans effet."""
    return signal


def subject_list(path: Path, cohort: str) -> tuple[list[str], str]:
    payload = json.loads(path.read_text())
    if payload.get("cohort") != cohort or not isinstance(payload.get("subjects"), list):
        raise SystemExit(f"{path} : liste figée {{cohort, subjects}} attendue pour {cohort}")
    subjects = payload["subjects"]
    if not subjects or len(set(subjects)) != len(subjects) or not all(isinstance(s, str) and s for s in subjects):
        raise SystemExit(f"{path} : sujets vides, en double ou invalides")
    return subjects, hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def embed_subject(edf: Path, cohort: str, sleepfm, osf, rule, batch_size: int, device: str) -> dict:
    """`sleepfm` : objet avec `embed_combined` ; `osf` : backbone appelable ; l'un ou l'autre peut manquer."""
    epochs, info = nsrr_night(edf, cohort, rule if sleepfm is not None else identity_scale)
    out = {"epoch_idx": np.arange(len(epochs)), "onset_s": 30.0 * np.arange(len(epochs)), "info": info}
    if sleepfm is not None:
        X = np.concatenate(
            [
                sleepfm.embed_combined(torch.from_numpy(epochs[i : i + batch_size]).to(device)).cpu().numpy()
                for i in range(0, len(epochs), batch_size)
            ]
        )
        if X.shape != (len(epochs), SLEEPFM_DIM) or not np.isfinite(X).all():
            raise ValueError("Embeddings SleepFM invalides")
        out["X_sleepfm"] = X.astype(np.float32)
    if osf is not None:
        out["X_osf"], info["osf_flat"] = embed_osf_night(osf, epochs, batch_size, device)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=("mesa", "shhs1"), required=True)
    parser.add_argument("--edf-dir", type=Path, required=True, help="<sujet>.edf")
    parser.add_argument("--subjects", type=Path, required=True, help="liste figée JSON {cohort, subjects}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="sleepfm,osf")
    parser.add_argument("--checkpoint", type=Path, help="best.pt SleepFM")
    parser.add_argument("--osf-checkpoint", type=Path)
    parser.add_argument("--respiratory-rule", default=None, help="nom dans RESPIRATORY_RULES (SleepFM)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = set(args.models.split(","))
    if not models or models - {"sleepfm", "osf"}:
        parser.error("--models : sleepfm, osf ou les deux")
    if "sleepfm" in models and (args.checkpoint is None or args.respiratory_rule not in RESPIRATORY_RULES):
        parser.error("SleepFM : --checkpoint et --respiratory-rule physionet_median_v1 obligatoires")
    if "osf" in models and args.osf_checkpoint is None:
        parser.error("OSF : --osf-checkpoint obligatoire")

    subjects, list_sha = subject_list(args.subjects, args.cohort)
    contract = {
        "cohort": args.cohort,
        "subjects_sha256": list_sha,
        "models": sorted(models),
        "sleepfm_checkpoint_sha256": file_sha256(args.checkpoint) if "sleepfm" in models else None,
        "osf_checkpoint_sha256": file_sha256(args.osf_checkpoint) if "osf" in models else None,
        "respiratory_rule": args.respiratory_rule if "sleepfm" in models else None,
        "preparation_version": PREPARATION_VERSION,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != contract:
            raise SystemExit(f"{manifest_path} : contrat différent, utiliser un nouveau dossier")
    elif any(args.out.glob("*.npz")):
        raise SystemExit(f"{args.out} : fichiers sans manifeste, provenance inconnue")
    else:
        write_json_atomic(manifest_path, contract)

    sleepfm = osf = None
    if "sleepfm" in models:
        from src.models.embeddings import SleepFMEncoders

        sleepfm = SleepFMEncoders(args.checkpoint, embedding_dim=512).to(args.device).eval()
    if "osf" in models:
        from src.models.osf_vit1d import load_osf_backbone

        osf = load_osf_backbone(args.osf_checkpoint).to(args.device).eval()
    rule = RESPIRATORY_RULES.get(args.respiratory_rule)

    todo = [s for s in subjects if not (args.out / f"{s}.npz").exists() and not (args.out / f"{s}.skipped").exists()]
    print(f"{len(subjects) - len(todo)} sujet(s) déjà faits, {len(todo)} à traiter sur {args.device}")
    infos = {}
    for index, subject in enumerate(todo):
        try:
            result = embed_subject(
                args.edf_dir / f"{subject}.edf", args.cohort, sleepfm, osf, rule, args.batch_size, args.device
            )
        except FileNotFoundError:
            print(f"[{index + 1}/{len(todo)}] {subject} : EDF absent, laissé manquant (repris à la relance)")
            continue
        except ValueError as exc:
            (args.out / f"{subject}.skipped").write_text(f"{exc}\n")
            print(f"[{index + 1}/{len(todo)}] {subject} rejeté : {exc}")
            continue
        partial = args.out / f"{subject}.tmp.npz"
        np.savez(partial, **{k: v for k, v in result.items() if k != "info"})
        os.replace(partial, args.out / f"{subject}.npz")
        infos[subject] = result["info"]
        print(f"[{index + 1}/{len(todo)}] {subject} : {result['info']['n_epochs']} fenêtres")

    status_path = args.out / "run_status.json"
    previous = json.loads(status_path.read_text()).get("nights", {}) if status_path.exists() else {}
    done = [s for s in subjects if (args.out / f"{s}.npz").exists()]
    skipped = {
        s: (args.out / f"{s}.skipped").read_text().strip() for s in subjects if (args.out / f"{s}.skipped").exists()
    }
    status = {
        "done": done,
        "rejected": skipped,
        "missing": [s for s in subjects if s not in done and s not in skipped],
        "complete": len(done) + len(skipped) == len(subjects),
        "nights": previous | infos,
        "environment": run_environment(),
    }
    write_json_atomic(status_path, status)
    print(f"{len(done)} sujet(s) sur {len(subjects)}, {len(skipped)} rejeté(s) ; complet : {status['complete']}")
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
