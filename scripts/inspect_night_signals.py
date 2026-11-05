"""Contrôle des signaux bruts de quelques nuits, pour comprendre un échec.

Pour chaque nuit (fichiers bruts PhysioNet 2018 sous --raw-dir) :

1. qualité : part d'époques plates par canal, amplitude médiane (écart-type
   par époque) de l'EEG C3, de l'EOG E1 et de l'EMG du menton ;
2. physiologie par stade (médianes sur les époques de l'expert, en rapport à
   N2 de la même nuit, donc sans dépendre du gain d'acquisition) :
   - tonus du menton (RMS de l'EMG filtré au-dessus de 10 Hz) : en REM, il
     doit chuter (atonie) ; c'est le principal indice REM contre éveil ;
   - mouvements oculaires (puissance 0,3-4 Hz de E1-M2) : bouffées en REM ;
   - part d'alpha occipital (8-12 Hz sur O2-M1) : élevée à l'éveil.

    python scripts/inspect_night_signals.py tr03-0592 tr03-0485 tr03-0167 \
        --raw-dir data/raw/physionet2018/training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.extract_physionet2018 import extract_labels, import_signal_names  # noqa: E402


def band_power(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    freqs, power = ss.welch(x, sr, nperseg=sr * 4, axis=-1)
    return power[..., (freqs >= low) & (freqs < high)].sum(-1)


def inspect(raw_dir: Path, subject: str) -> tuple[str, str]:
    folder = raw_dir / subject
    channels, sr, _ = import_signal_names(str(folder / f"{subject}.hea"))
    signal = loadmat(str(folder / f"{subject}.mat"))["val"].astype(float)
    labels, names = extract_labels(str(folder / f"{subject}-arousal.mat"))
    n = signal.shape[1] // (30 * sr)
    epochs = signal[:, : n * 30 * sr].reshape(len(channels), n, 30 * sr)
    fractions = labels[: n * 30 * sr].reshape(n, 30 * sr, -1).mean(1)
    stage = np.array([names[i] if fractions[e, i] == 1 else "" for e, i in enumerate(fractions.argmax(1))])

    def channel(name: str) -> np.ndarray:
        return epochs[channels.index(name)]

    flat = (epochs.std(2) < 1e-6).mean(1)
    flat_txt = ", ".join(f"{c} {f:.0%}" for c, f in zip(channels, flat, strict=True) if f > 0.01) or "aucune"
    amplitudes = " / ".join(f"{np.median(channel(c).std(1)):.1f}" for c in ("C3-M2", "E1-M2", "Chin1-Chin2"))
    quality = f"{subject:<11} époques plates : {flat_txt} ; amplitude C3 / E1 / menton : {amplitudes}"

    chin = ss.sosfiltfilt(ss.butter(4, 10, "hp", fs=sr, output="sos"), channel("Chin1-Chin2"), axis=1)
    features = {
        "chin": np.sqrt((chin**2).mean(1)),
        "eog": band_power(channel("E1-M2"), sr, 0.3, 4),
        "alpha": band_power(channel("O2-M1"), sr, 8, 12) / band_power(channel("O2-M1"), sr, 0.5, 30),
    }

    def ratio(feature: str, a: str, b: str) -> float:
        if not ((stage == a).any() and (stage == b).any()):
            return float("nan")
        return float(np.median(features[feature][stage == a]) / np.median(features[feature][stage == b]))

    physiology = (
        f"{subject:<11}{ratio('chin', 'rem', 'nonrem2'):>14.2f}{ratio('chin', 'wake', 'nonrem2'):>12.2f}"
        f"{ratio('eog', 'rem', 'nonrem2'):>12.2f}{ratio('alpha', 'wake', 'nonrem2'):>11.2f}"
        f"{ratio('alpha', 'rem', 'nonrem2'):>13.2f}"
    )
    return quality, physiology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/physionet2018/training"))
    args = parser.parse_args()
    reports = [inspect(args.raw_dir, s) for s in args.subjects]
    print("1. Qualité des signaux")
    for quality, _ in reports:
        print(f"  {quality}")
    print("2. Physiologie par stade, en rapport à N2 de la même nuit (médianes)")
    print(f"  {'nuit':<11}{'menton REM':>14}{'menton W':>12}{'yeux REM':>12}{'alpha W':>11}{'alpha REM':>13}")
    for _, physiology in reports:
        print(f"  {physiology}")


if __name__ == "__main__":
    main()
