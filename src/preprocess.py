import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml  # type: ignore


@dataclass
class PreprocessConfig:
    dataset_root: str
    processed_root: str
    channels: List[str]
    window_seconds: float
    sample_rate: int
    overlap: float
    filtering: str  # 'raw' or 'filtered'
    split_csv: str
    class_map_csv: str
    store_minmax: bool = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def load_cfg(path: str) -> PreprocessConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    d = y["data"]
    return PreprocessConfig(
        dataset_root=d["dataset_root"],
        processed_root=d["processed_root"],
        channels=d["channels"],
        window_seconds=float(d["window_seconds"]),
        sample_rate=int(d["sample_rate"]),
        overlap=float(d["overlap"]),
        filtering=str(d["filtering"]),
        split_csv=str(d["split_csv"]),
        class_map_csv=str(d["class_map_csv"]),
        store_minmax=bool(d.get("store_minmax", True)),
    )


def bandpass_notch(x: np.ndarray, fs: int) -> np.ndarray:
    try:
        from scipy import signal  # type: ignore
    except ImportError:
        return x
    # 0.5–45 Hz band-pass (4th order Butterworth) + 60 Hz notch (Q=30)
    b_bp, a_bp = signal.butter(4, [0.5 / (fs / 2), 45.0 / (fs / 2)], btype="band")
    y = signal.filtfilt(b_bp, a_bp, x, axis=-1)
    b_n, a_n = signal.iirnotch(w0=60.0 / (fs / 2), Q=30.0)
    y = signal.filtfilt(b_n, a_n, y, axis=-1)
    return y


def subjectwise_splits(split_csv: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    p = Path(split_csv)
    if not p.exists():
        return mapping
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and ("subject_id" in line and "," in line):
                # header
                continue
            parts = [s.strip() for s in line.strip().split(",")]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def class_map(class_map_csv: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    p = Path(class_map_csv)
    if not p.exists():
        return mapping
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and ("short" in line and "," in line):
                continue
            parts = [s.strip() for s in line.strip().split(",")]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def iterate_raw_events(_dataset_root: str, _channels: List[str]) -> List[Dict]:
    # Placeholder: implement TUAR iteration with MNE in a follow-up.
    # Should yield dicts: subject_id, recording_id, start_s, end_s, label, array (C x T)
    return []


def window_signal(arr: np.ndarray, fs: int, win_s: float, overlap: float) -> List[Tuple[int, np.ndarray]]:
    step = int(win_s * fs * (1 - overlap))
    length = int(win_s * fs)
    windows = []
    t_total = arr.shape[-1]
    start = 0
    while start + length <= t_total:
        windows.append((start, arr[:, start:start + length]))
        start += step
    return windows


def deduplicate_near_identical(windows: List[np.ndarray], tol: float = 1e-6) -> List[int]:
    # Return indices to keep; simple L2 thresholding placeholder
    keep = []
    prev = None
    for i, w in enumerate(windows):
        if prev is None:
            keep.append(i)
            prev = w
            continue
        if np.linalg.norm((w - prev).ravel(), ord=2) > tol:
            keep.append(i)
            prev = w
    return keep


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    os.makedirs(cfg.processed_root, exist_ok=True)
    meta_dir = Path(cfg.processed_root) / "metadata"
    os.makedirs(meta_dir, exist_ok=True)

    splits = subjectwise_splits(cfg.split_csv)
    cmap = class_map(cfg.class_map_csv)

    fs = cfg.sample_rate
    win_s = cfg.window_seconds

    manifest = []
    for ev in iterate_raw_events(cfg.dataset_root, cfg.channels):
        subj = ev["subject_id"]
        split = splits.get(subj, "train")
        label = cmap.get(ev["label"], ev["label"])  # map to canonical name if available
        arr = ev["array"]  # (C, T)

        if cfg.filtering == "filtered":
            arr = bandpass_notch(arr, fs)

        win_list = window_signal(arr, fs, win_s, cfg.overlap)
        _, windows = zip(*win_list) if win_list else ([], [])
        windows = list(windows)
        keep_idx = deduplicate_near_identical(windows)

        out_class_dir = Path(cfg.processed_root) / split / label
        os.makedirs(out_class_dir, exist_ok=True)

        for j, idx in enumerate(keep_idx):
            w = windows[idx]
            # Normalization placeholder: store min/max for WGAN, z-score handled later for DDPM
            meta: Dict = {
                "subject_id": subj,
                "recording_id": ev.get("recording_id"),
                "label": label,
                "channels": cfg.channels,
                "fs": fs,
                "filtering": cfg.filtering,
            }
            if cfg.store_minmax:
                meta["min"] = float(w.min())
                meta["max"] = float(w.max())

            base = f"{subj}_{ev.get('recording_id','rec')}_{j:05d}"
            npy_path = out_class_dir / f"{base}.npy"
            meta_path = meta_dir / f"{base}.json"
            np.save(npy_path, w.astype(np.float32))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            manifest.append({
                "path": str(npy_path),
                "meta": str(meta_path),
                "split": split,
                "label": label,
            })

    # Append to global manifest file if present in configs
    try:
        with open("results/manifest.json", "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(manifest)
    os.makedirs("results", exist_ok=True)
    with open("results/manifest.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved {len(manifest)} windows to {cfg.processed_root}")


if __name__ == "__main__":
    main()
