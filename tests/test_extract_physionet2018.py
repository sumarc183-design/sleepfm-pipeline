"""Vérifie le parsing des fichiers PhysioNet 2018 (.hea) et la conversion
des labels one-hot en événements."""

import numpy as np

from src.data.extract_physionet2018 import ALL_CHANNELS, import_signal_names, process_labels_to_events


def test_import_signal_names(tmp_path):
    lines = [f"tr03-0005 {len(ALL_CHANNELS)} 200 2567000"]
    for ch in ALL_CHANNELS:
        lines.append(f"tr03-0005.mat 16 100/mV 16 0 0 0 0 {ch}")
    hea_path = tmp_path / "tr03-0005.hea"
    hea_path.write_text("\n".join(lines) + "\n")

    channels, sample_rate, n_samples = import_signal_names(str(hea_path))
    assert channels == ALL_CHANNELS
    assert sample_rate == 200
    assert n_samples == 2567000


def test_process_labels_to_events_faithful_to_upstream():
    # 3 runs of 100 samples : stage 0, 1, 0 -- vérifie qu'on reproduit
    # fidèlement le comportement (avec son off-by-one hérité) de
    # l'algorithme officiel, pas une version "propre" qui diverge.
    labels = np.zeros((300, 3))
    labels[0:100, 0] = 1
    labels[100:200, 1] = 1
    labels[200:300, 0] = 1
    events = process_labels_to_events(labels)
    assert events == [[0, 101, 0], [101, 100, 1], [201, 99, 0]]
