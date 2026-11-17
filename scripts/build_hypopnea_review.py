"""Page HTML autonome de relecture des 40 fenêtres par le médecin (plan du 17 novembre).

Pour chaque fenêtre : 2 min de débit, sangles, SaO2 et EEG C4-M1 centrées sur la fenêtre de 30 s,
annotations des experts en surimpression. Ni groupe ni sortie des modèles dans la page : seul
l'identifiant F01 à F40 relie les réponses à l'échantillon. Réponses gardées dans le navigateur
et exportées en CSV. À lancer dans l'environnement muni de WFDB (requirements-respiratory.txt).

    python scripts/build_hypopnea_review.py --sample results/hypopnea_review_sample.json \\
        --raw data/raw/physionet2018/training --annotations data/respiratory_annotations \\
        --out results/hypopnea_review.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_contract import file_sha256  # noqa: E402

CHANNELS = [("AIRFLOW", "Débit nasal"), ("CHEST", "Sangle thoracique"), ("ABD", "Sangle abdominale"),
            ("SaO2", "SaO2 (%)"), ("C4-M1", "EEG C4-M1")]  # fmt: skip
CONTEXT_S = 120.0
WINDOW_S = 30.0
COLUMNS = 600  # colonnes min/max par voie : une tous les 0,2 s
TEMPLATE = Path(__file__).with_name("hypopnea_review_template.html")


def channel_gains(hea: Path) -> dict[str, tuple[float, float]]:
    """{voie: (gain, ligne de base)} d'après l'en-tête WFDB."""
    gains = {}
    for line in hea.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        match = re.match(r"([\d.]+)(?:\((-?\d+)\))?", parts[2])
        gains[parts[8]] = (float(match.group(1)), float(match.group(2) or 0))
    return gains


def envelope(signal: np.ndarray, columns: int = COLUMNS) -> list[float]:
    """Min puis max par colonne : la forme du signal reste lisible sans tous les échantillons."""
    chunks = np.array_split(signal, columns)
    return [round(float(v), 2) for c in chunks for v in (c.min(), c.max())]


def annotation_intervals(samples, notes, rate: int, start_s: float, end_s: float) -> list[dict]:
    """Événements « (nom … nom) » qui chevauchent [start_s, end_s], en secondes relatives."""
    opened, out = {}, []
    for sample, note in zip(samples, notes, strict=True):
        if note.startswith("("):
            opened[note[1:]] = sample / rate
        elif note.endswith(")") and note[:-1] in opened:
            begin, end = opened.pop(note[:-1]), sample / rate
            if end > start_s and begin < end_s:
                out.append({"name": note[:-1], "start": round(begin - start_s, 2), "end": round(end - start_s, 2)})
    return out


def window_payload(signals: dict[str, np.ndarray], rate: int, onset_s: float, events: list[dict]) -> dict:
    n = len(next(iter(signals.values())))
    start = min(max(0.0, onset_s + WINDOW_S / 2 - CONTEXT_S / 2), n / rate - CONTEXT_S)
    a, b = int(round(start * rate)), int(round((start + CONTEXT_S) * rate))
    return {
        "box": [round(onset_s - start, 2), round(onset_s - start + WINDOW_S, 2)],
        "traces": {name: envelope(signals[name][a:b]) for name, _ in CHANNELS},
        "events": events,
        "start": start,
    }


def load_window(raw: Path, annotations: Path, subject: str, onset_s: float) -> dict:
    import wfdb
    from scipy.io import loadmat

    hea = raw / subject / f"{subject}.hea"
    gains = channel_gains(hea)
    header = hea.read_text(encoding="utf-8").split()
    rate = int(header[2])
    names = [line.split()[8] for line in hea.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    values = loadmat(raw / subject / f"{subject}.mat")["val"]
    signals = {}
    for name, _ in CHANNELS:
        gain, base = gains[name]
        signals[name] = (values[names.index(name)].astype(np.float64) - base) / gain
    annotation = wfdb.rdann(str(annotations / subject / subject), "arousal")
    payload = window_payload(signals, rate, onset_s, [])
    payload["events"] = annotation_intervals(
        annotation.sample, annotation.aux_note, rate, payload["start"], payload["start"] + CONTEXT_S
    )
    return payload


def render(windows: list[dict], sample_sha256: str) -> str:
    data = json.dumps(
        {"windows": windows, "channels": [label for _, label in CHANNELS], "keys": [k for k, _ in CHANNELS]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__DATA__", data.replace("</", "<\\/"))
        .replace("__SAMPLE__", html.escape(sample_sha256))
        .replace("__N__", str(len(windows)))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    windows = []
    for w in sorted(sample["windows"], key=lambda w: w["order"]):
        payload = load_window(args.raw, args.annotations, w["subject"], w["onset_s"])
        windows.append({"id": w["id"], **{k: payload[k] for k in ("box", "traces", "events")}})
        print(w["id"], flush=True)
    args.out.write_text(render(windows, file_sha256(args.sample)), encoding="utf-8", newline="\n")
    print(f"{len(windows)} fenêtres -> {args.out} ({args.out.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
