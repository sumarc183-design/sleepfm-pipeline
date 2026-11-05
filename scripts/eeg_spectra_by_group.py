"""Spectre de l'EEG par groupe d'époques (stade de l'expert / stade prédit).

Pour comprendre un désaccord entre l'expert et le modèle : dans chaque nuit,
les époques sont groupées par (stade de l'expert, stade prédit par le BiGRU),
et l'on compare, en médiane par groupe :
- la part de la puissance 0,5-30 Hz de C3-M2 par bande (delta, thêta,
  alpha, sigma, bêta) : indépendante du gain d'acquisition ;
- la part d'alpha en O2-M1 (élevée à l'éveil, yeux fermés) ;
- le tonus du menton (RMS au-dessus de 10 Hz), en rapport à N2 de la nuit
  (à la médiane de la nuit s'il y a moins de 20 époques de N2).
Époques repérées par leur temps d'origine (onset_s) dans les prédictions ;
signaux bruts PhysioNet sous --raw-dir. Groupes de moins de 30 époques omis.

    python scripts/eeg_spectra_by_group.py tr03-1292 tr04-0133 tr03-0167 \
        --predictions results/results_physionet2018_200.predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.extract_physionet2018 import import_signal_names  # noqa: E402

BANDS = {"delta": (0.5, 2), "thêta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "bêta": (15, 30)}
STAGES = ["W", "N1", "N2", "N3", "REM"]
MIN_EPOCHS = 30


def relative_band_power(epochs: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    freqs, power = ss.welch(epochs, sr, nperseg=sr * 4, axis=-1)
    total = power[:, (freqs >= 0.5) & (freqs < 30)].sum(1)
    total[total == 0] = np.nan  # époques plates : exclues des médianes
    return {band: power[:, (freqs >= lo) & (freqs < hi)].sum(1) / total for band, (lo, hi) in BANDS.items()}


def night_features(raw_dir: Path, subject: str, onsets: np.ndarray) -> dict[str, np.ndarray]:
    folder = raw_dir / subject
    channels, sr, _ = import_signal_names(str(folder / f"{subject}.hea"))
    signal = loadmat(str(folder / f"{subject}.mat"))["val"].astype(float)

    def epochs(name: str) -> np.ndarray:
        row = signal[channels.index(name)]
        return np.stack([row[o * sr : (o + 30) * sr] for o in onsets])

    features = relative_band_power(epochs("C3-M2"), sr)
    features["alphaO2"] = relative_band_power(epochs("O2-M1"), sr)["alpha"]
    chin = ss.sosfiltfilt(ss.butter(4, 10, "hp", fs=sr, output="sos"), epochs("Chin1-Chin2"), axis=1)
    features["menton"] = np.sqrt((chin**2).mean(1))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/physionet2018/training"))
    args = parser.parse_args()

    saved = np.load(args.predictions)
    if not np.isfinite(saved["onset_s"]).all():
        raise SystemExit("Prédictions sans temps d'origine (onset_s) : cache v1, époques non repérables")
    columns = [*BANDS, "alphaO2", "menton"]
    print(f"  {'groupe (C3 : part de puissance)':<34}{'n':>5}" + "".join(f"{c:>9}" for c in columns))
    for subject in args.subjects:
        rows = saved["subject"] == subject
        if not rows.any():
            raise SystemExit(f"{subject} absent des prédictions")
        y, pred = saved["y_true"][rows], saved["pred_sequence"][rows]
        features = night_features(args.raw_dir, subject, saved["onset_s"][rows].astype(int))
        enough_n2 = (y == 2).sum() >= 20
        chin_ref = np.median(features["menton"][y == 2] if enough_n2 else features["menton"])
        print(f"{subject} (menton : rapport à la médiane {'N2' if enough_n2 else 'de la nuit'})")
        for true_stage in range(5):
            for predicted in range(5):
                group = (y == true_stage) & (pred == predicted)
                if group.sum() < MIN_EPOCHS:
                    continue
                values = [np.nanmedian(features[c][group]) for c in columns[:-1]]
                values.append(np.median(features["menton"][group]) / chin_ref)
                label = f"expert {STAGES[true_stage]} / BiGRU {STAGES[predicted]}"
                print(f"  {label:<34}{group.sum():>5}" + "".join(f"{v:>9.2f}" for v in values))


if __name__ == "__main__":
    main()
