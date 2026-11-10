"""Comparaison exploratoire SleepFM / OSF sur PhysioNet 2018 (option A de docs/OSF_COMPARISON.md).

Mêmes 400 nuits d'entraînement et 100 nuits de test (découpage figé), mêmes
fenêtres, mêmes sondes et mêmes réglages pour les deux modèles : seule la
représentation change. Les deux jeux d'embeddings sont figés ; les sondes des
deux modèles sont réentraînées ici par le même code (logistique pondérée et
BiGRU, graine 42). Critère fixé avant le calcul, dans docs/OSF_COMPARISON.md :
écart de κ OSF moins SleepFM, BiGRU, 5 classes, bootstrap apparié par sujet
(1 000 tirages, graine 42), IC à 95 %.

Résultat exploratoire : 4 des 12 canaux d'OSF manquent dans PhysioNet 2018.

    python scripts/compare_osf_sleepfm.py --sleepfm-dir embeddings/physionet2018_v2 \\
        --osf-dir embeddings/physionet2018_osf --out results/osf_vs_sleepfm_physionet2018
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import load_cached_embeddings  # noqa: E402
from src.data.cache_contract import file_sha256, write_json_atomic  # noqa: E402
from src.models.linear_probe import paired_bootstrap_diff, predict_labels, train_probe  # noqa: E402
from src.models.probe_export import run_environment  # noqa: E402

PROBES = ("logistic", "sequence")
SEED, N_BOOT = 42, 1000
# 4 classes comme l'article OSF : éveil, sommeil léger (N1 + N2), profond (N3), REM.
TO_FOUR = np.array([0, 1, 1, 2, 3])
STAGES = ["wake", "nonrem1", "nonrem2", "nonrem3", "rem"]


def load_osf(directory: Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, tuple]:
    X, y, s, e, o = [], [], [], [], []
    for subject in subjects:
        with np.load(directory / f"{subject}.npz") as data:
            if data["X"].ndim != 2 or data["X"].shape[1] != 768 or not np.isfinite(data["X"]).all():
                raise ValueError(f"{subject} : embeddings OSF invalides")
            X.append(data["X"])
            y.append(data["y"])
            e.append(data["epoch_idx"])
            o.append(data["onset_s"])
            s.append(np.full(len(data["y"]), subject))
    return np.concatenate(X), np.concatenate(y), (np.concatenate(s), np.concatenate(e), np.concatenate(o))


def aligned(groups_a: tuple, groups_b: tuple, y_a: np.ndarray, y_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lignes de b dans l'ordre de a ; arrêt si les fenêtres ou les stades diffèrent."""
    key_a = list(zip(groups_a[0].tolist(), groups_a[1].tolist(), strict=True))
    index_b = {k: i for i, k in enumerate(zip(groups_b[0].tolist(), groups_b[1].tolist(), strict=True))}
    if len(index_b) != len(key_a) or set(index_b) != set(key_a):
        raise SystemExit("Les deux modèles n'ont pas exactement les mêmes fenêtres")
    order = np.array([index_b[k] for k in key_a])
    if not np.array_equal(y_a, y_b[order]) or np.abs(groups_a[2] - groups_b[2][order]).max() > 0.01:
        raise SystemExit("Stades ou temps différents entre les deux modèles")
    return np.arange(len(key_a)), order


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "cohen_kappa_4": float(cohen_kappa_score(TO_FOUR[y], TO_FOUR[pred])),
        "f1_by_stage": dict(zip(STAGES, f1_score(y, pred, labels=range(5), average=None).tolist(), strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sleepfm-dir", type=Path, required=True)
    parser.add_argument("--osf-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("configs/split_physionet2018.json"))
    parser.add_argument("--out", type=Path, required=True, help="préfixe du JSON de sortie")
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text())
    status = json.loads((args.osf_dir / "run_status.json").read_text())
    if not status["complete"] or status["rejected"]:
        raise SystemExit("Run OSF incomplet ou avec rejets : comparaison sur des nuits différentes refusée")

    data = {}
    for part in ("train", "test"):
        Xs, ys, gs = load_cached_embeddings(args.sleepfm_dir, split[part])
        Xo, yo, go = load_osf(args.osf_dir, split[part])
        rows_s, rows_o = aligned(gs, go, ys, yo)
        data[part] = {
            "sleepfm": (Xs[rows_s], (gs[0][rows_s], gs[1][rows_s], gs[2][rows_s])),
            "osf": (Xo[rows_o], (go[0][rows_o], go[1][rows_o], go[2][rows_o])),
            "y": ys[rows_s],
        }

    y_test, subjects_test = data["test"]["y"], data["test"]["sleepfm"][1][0]
    predictions, report_models = {}, {}
    for model in ("sleepfm", "osf"):
        X_train, g_train = data["train"][model]
        X_test, g_test = data["test"][model]
        for probe in PROBES:
            fitted = train_probe(probe, X_train, data["train"]["y"], seed=SEED, groups=g_train)
            predictions[(model, probe)] = predict_labels(fitted, X_test, g_test)
            report_models[f"{model}/{probe}"] = scores(y_test, predictions[(model, probe)])
            print(f"{model:8s} {probe:9s} κ {report_models[f'{model}/{probe}']['cohen_kappa']:.3f}")

    differences = {}
    for probe in PROBES:
        a, b = predictions[("sleepfm", probe)], predictions[("osf", probe)]
        five = paired_bootstrap_diff(y_test, a, b, subjects_test, n_boot=N_BOOT, seed=SEED)
        four = paired_bootstrap_diff(TO_FOUR[y_test], TO_FOUR[a], TO_FOUR[b], subjects_test, n_boot=N_BOOT, seed=SEED)
        differences[probe] = {"five_classes": five, "four_classes": four}

    primary = differences["sequence"]["five_classes"]["cohen_kappa"]
    low, high = primary["ci95"]
    verdict = "OSF au-dessus" if low > 0 else "SleepFM au-dessus" if high < 0 else "pas de différence établie"
    report = {
        "kind": "comparaison exploratoire SleepFM / OSF, PhysioNet 2018, option A",
        "caveat": "4 des 12 canaux OSF absents (mis à zéro) ; normalisation par nuit supposée ; exploratoire",
        "nights": {"train": len(split["train"]), "test": len(split["test"])},
        "test_epochs": int(len(y_test)),
        "osf_manifest": json.loads((args.osf_dir / "manifest.json").read_text()),
        "sleepfm_manifest": json.loads((args.sleepfm_dir / "manifest.json").read_text()),
        "split_sha256": file_sha256(args.split_file),
        "models": report_models,
        "difference_osf_minus_sleepfm": differences,
        "primary": {"kappa_osf_minus_sleepfm_bigru_5_classes": primary, "reading": verdict},
        "environment": run_environment(),
    }
    write_json_atomic(Path(f"{args.out}.json"), report)
    print(f"Critère principal, BiGRU 5 classes, κ OSF − SleepFM : {primary['diff']:+.3f} [{low:+.3f} ; {high:+.3f}]")
    print(f"Lecture : {verdict} (exploratoire, canaux OSF incomplets)")
    print(f"Résultats : {args.out}.json")


if __name__ == "__main__":
    main()
