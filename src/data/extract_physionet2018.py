"""Extracts 30s epochs + sleep-stage labels from PhysioNet/CinC Challenge 2018
raw files, in the exact format the official SleepFM preprocessing expects.

Adapted from 0_extract_pretraining_data.py in rthapa84/sleepfm-codebase (MIT
license, see THIRD_PARTY_NOTICES.md). Each subject folder under --data_path
must contain three files (as distributed by PhysioNet):
  <id>.hea            header: channel names, sampling rate
  <id>.mat             raw PSG signal (scipy.io.loadmat key "val")
  <id>-arousal.mat     sleep-stage + arousal labels (h5py, one-hot per sample)

Output layout under --save_path:
  X/<id>/<id>_<epoch_idx>.npy   (n_channels, n_samples) float, one 30s epoch
  Y/<id>.pickle                  {filename: stage_label_name}
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
    """Collapses per-sample one-hot labels into (start, duration, stage_id) runs.

    NOTE: faithfully ported from the official implementation, including its
    off-by-one run boundaries (each run's `duration` includes the first
    sample of the next run, i.e. runs of 100/100/100 samples come out as
    101/100/99). This is a ~1-sample (a few ms) boundary jitter inherited
    from upstream, not a bug here -- left as-is so results stay comparable
    to the published numbers we're trying to reproduce.
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
