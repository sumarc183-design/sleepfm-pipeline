"""Extrait les époques de 30 s et les stades de sommeil des fichiers bruts du
PhysioNet/CinC Challenge 2018, au format exact attendu par le prétraitement
officiel de SleepFM.

Adapté de 0_extract_pretraining_data.py (rthapa84/sleepfm-codebase, licence
MIT, voir THIRD_PARTY_NOTICES.md). Chaque dossier de sujet sous --data_path
contient trois fichiers (tels que distribués par PhysioNet) :
  <id>.hea            en-tête : noms des canaux, fréquence d'échantillonnage
  <id>.mat             signal PSG brut (clé "val" de scipy.io.loadmat)
  <id>-arousal.mat     stades de sommeil + arousals (h5py, one-hot par échantillon)

Sortie sous --save_path :
  X/<id>/<id>_<epoch_idx>.npy   (n_canaux, n_échantillons) float, une époque de 30 s
  Y/<id>.pickle                  {nom_de_fichier: nom_du_stade}
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import pickle
import warnings
from typing import List, Tuple

import h5py
import mne
import numpy as np
import scipy.signal
from scipy.io import loadmat
from tqdm import tqdm

warnings.filterwarnings("ignore")

ALL_CHANNELS = [
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2",
    "Chin1-Chin2", "ABD", "CHEST", "AIRFLOW", "SaO2", "ECG",
]

EVENT_TO_ID = {"wake": 1, "nonrem1": 2, "nonrem2": 3, "nonrem3": 4, "rem": 5}
ID_TO_EVENT = {v: k for k, v in EVENT_TO_ID.items()}


def import_signal_names(hea_path: str) -> Tuple[List[str], int, int]:
    with open(hea_path, "r") as f:
        lines = [line.split() for line in f.read().split("\n")]
    n_signals = int(lines[0][1])
    sample_rate = int(lines[0][2])
    n_samples = int(lines[0][3])
    channel_names = [lines[i][8] for i in range(1, 1 + n_signals)]
    return channel_names, sample_rate, n_samples


def extract_labels(arousal_mat_path: str) -> Tuple[np.ndarray, List[str]]:
    """One-hot sleep-stage labels, sampled at the original PSG rate."""
    data = h5py.File(arousal_mat_path, "r")
    stage_names = list(data["data"]["sleep_stages"].keys())
    length = data["data"]["sleep_stages"][stage_names[0]].shape[1]
    labels = np.zeros((length, len(stage_names)))
    for i, name in enumerate(stage_names):
        labels[:, i] = data["data"]["sleep_stages"][name][:]
    return labels, stage_names


def process_labels_to_events(labels: np.ndarray) -> List[list]:
    """Regroupe les labels one-hot par échantillon en segments
    (début, durée, id_stade).

    NOTE : porté fidèlement depuis l'implémentation officielle, y compris son
    décalage d'un échantillon aux frontières (la `durée` de chaque segment
    inclut le premier échantillon du suivant : des segments de 100/100/100
    échantillons sortent en 101/100/99). Ce flottement d'environ 1 échantillon
    (quelques ms) vient du code amont, pas d'une erreur ici -- gardé tel quel
    pour rester comparable aux chiffres publiés qu'on cherche à reproduire.
    """
    stage_ids = np.argmax(labels, axis=1)
    events, start, current = [], 0, stage_ids[0]
    i = 0
    while i < len(stage_ids) - 1:
        while i < len(stage_ids) - 1 and stage_ids[i] == current:
            i += 1
        events.append([start, i + 1 - start, current])
        current = stage_ids[i]
        start = i + 1
    return events


def process_one_subject(
    subject_files: dict,
    path_to_save: str,
    chunk_duration: float,
    target_sample_rate: int,
) -> None:
    hea_path, arousal_path, mat_path = subject_files["hea"], subject_files["arousal_mat"], subject_files["mat"]
    subject_id = os.path.basename(mat_path).split(".")[0]

    path_to_x = os.path.join(path_to_save, "X", subject_id)
    path_to_y = os.path.join(path_to_save, "Y", f"{subject_id}.pickle")
    if os.path.exists(path_to_x) and os.listdir(path_to_x):
        return  # already processed

    try:
        channel_names, sample_rate, _ = import_signal_names(hea_path)
        labels, stage_names = extract_labels(arousal_path)
        signal = loadmat(mat_path)["val"]
    except Exception as exc:
        print(f"[{subject_id}] skipped: failed to load raw files ({exc})")
        return

    try:
        # Drop the leading segment recorded before the first stage transition
        # (undefined/lead-in period), matching official preprocessing.
        diff = np.diff(labels, axis=0)
        cutoff = np.where(diff[:, -1] != 0)[0]
        if len(cutoff) == 0:
            print(f"[{subject_id}] skipped: no stage transitions found")
            return
        start = cutoff[0] + 1
        signal, labels = signal[:, start:], labels[start:, :]

        info = mne.create_info(channel_names, sample_rate, ch_types="eeg")
        raw = mne.io.RawArray(signal, info, verbose=False)

        label_id_by_index = dict(enumerate(stage_names))
        events = np.array(process_labels_to_events(labels))
        descriptions = [label_id_by_index[i] for i in events[:, 2]]
        raw.set_annotations(
            mne.Annotations(onset=events[:, 0] / sample_rate, duration=events[:, 1] / sample_rate, description=descriptions),
            emit_warning=False,
        )
        raw.resample(sfreq=256, verbose=False)

        events_raw, _ = mne.events_from_annotations(raw, event_id=EVENT_TO_ID, chunk_duration=chunk_duration, verbose=False)
        present_ids = {ID_TO_EVENT[i]: i for i in set(events_raw[:, 2])}
        epochs = mne.Epochs(
            raw=raw, events=events_raw, event_id=present_ids,
            tmin=0.0, tmax=chunk_duration - 1.0 / raw.info["sfreq"],
            baseline=None, event_repeated="drop", verbose=False,
        ).drop_bad(verbose=False)
    except Exception as exc:
        print(f"[{subject_id}] skipped: preprocessing failed ({exc})")
        return

    channel_indices = [raw.ch_names.index(ch) for ch in ALL_CHANNELS if ch in raw.ch_names]
    if len(channel_indices) != len(ALL_CHANNELS):
        print(f"[{subject_id}] skipped: missing required channels")
        return

    os.makedirs(path_to_x, exist_ok=True)
    labels_out = {}
    for idx, epoch in enumerate(epochs):
        epoch_data = epoch[channel_indices, :]
        if epoch_data.shape[1] == 0:
            continue
        n_target = int(epoch_data.shape[1] * target_sample_rate / raw.info["sfreq"])
        resampled = scipy.signal.resample(epoch_data, n_target, axis=1)

        file_name = f"{subject_id}_{idx}.npy"
        np.save(os.path.join(path_to_x, file_name), resampled)
        labels_out[file_name] = epochs[idx].event_id

    with open(path_to_y, "wb") as f:
        pickle.dump(labels_out, f)


def _worker(args: tuple) -> None:
    subjects, path_to_save, chunk_duration, target_sample_rate = args
    for subject_files in tqdm(subjects):
        process_one_subject(subject_files, path_to_save, chunk_duration, target_sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Directory of downloaded PhysioNet 2018 subject folders")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_files", type=int, default=-1, help="-1 = all downloaded subjects")
    parser.add_argument("--chunk_duration", type=float, default=30.0)
    parser.add_argument("--target_sample_rate", type=int, default=256)
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_path, "X"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "Y"), exist_ok=True)

    subjects = []
    for subject_dir in sorted(os.listdir(args.data_path)):
        subject_path = os.path.join(args.data_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        files = glob.glob(f"{subject_path}/*")
        entry = {}
        for file in files:
            if file.endswith(".hea"):
                entry["hea"] = file
            elif file.endswith("-arousal.mat"):
                entry["arousal_mat"] = file
            elif file.endswith(".mat"):
                entry["mat"] = file
        if {"hea", "arousal_mat", "mat"} <= entry.keys():
            subjects.append(entry)

    if args.num_files != -1:
        subjects = subjects[: args.num_files]
    print(f"{len(subjects)} subject(s) to process")

    chunks = np.array_split(subjects, max(1, args.num_threads))
    tasks = [(list(chunk), args.save_path, args.chunk_duration, args.target_sample_rate) for chunk in chunks if len(chunk)]
    with multiprocessing.Pool(args.num_threads) as pool:
        pool.map(_worker, tasks)


if __name__ == "__main__":
    main()
