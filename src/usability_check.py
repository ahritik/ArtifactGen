#!/usr/bin/env python3
"""Reusable function to check if a subject is usable for preprocessing."""
import csv
from pathlib import Path
from typing import List, Tuple, Dict
import mne
from src.merge_map import remap_label


def normalize_ch_names(orig_chs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Normalize TUAR/MNE channel name variants to canonical names."""
    heur = [
        (['fp1', 'fp1f3', 'fp1f7'], 'Fp1'),
        (['fp2', 'fp2f4', 'fp2f8'], 'Fp2'),
        (['c3', 'f3c3', 'c3p3'], 'C3'),
        (['c4', 'f4c4', 'c4p4'], 'C4'),
        (['o1', 'o1o2'], 'O1'),
        (['o2', 'o2o1'], 'O2'),
        (['t3', 'f7t3', 't3t5'], 'T3'),
        (['t4', 'f8t4', 't4t6'], 'T4'),
    ]
    name_map: Dict[str, str] = {}
    normalized = []
    for ch in orig_chs:
        ch_l = ch.lower().replace(' ', '').replace('-', '').replace('_', '')
        mapped = None
        for keys, canon in heur:
            for k in keys:
                if k in ch_l:
                    mapped = canon
                    break
            if mapped:
                break
        if mapped is None:
            mapped = ch
        name_map[ch] = mapped
        normalized.append(mapped)
    return normalized, name_map


def is_subject_usable(dataset_root: str, subject: str, required_channels: List[str]) -> bool:
    """
    Check if a subject is usable by simulating the full preprocessing pipeline.
    """
    edf_root = Path(dataset_root) / 'edf'
    for subdir in edf_root.iterdir():
        if not subdir.is_dir():
            continue
        for edf_path in subdir.glob(f"{subject}*.edf"):
            csv_path = edf_path.with_suffix('.csv')
            if not csv_path.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception:
                continue
            if raw.times[-1] < 1.0:
                continue
            orig_chs = raw.ch_names
            normalized_chs, ch_name_map = normalize_ch_names(orig_chs)
            present = set(normalized_chs)
            if not all(r in present for r in required_channels):
                continue
            # Pick channels as in preprocessing
            pick_names = []
            for rc in required_channels:
                for orig, mapped in ch_name_map.items():
                    if mapped == rc:
                        pick_names.append(orig)
                        break
            if len(pick_names) != len(required_channels):
                continue
            raw.pick(pick_names)
            data = raw.get_data()
            fs = int(raw.info['sfreq'])
            # Parse annotations and check for valid events
            events = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5 or row[0].startswith('#') or row[0].lower() == 'channel':
                        continue
                    ch, start, stop, label, conf = row[:5]
                    try:
                        start = float(start)
                        stop = float(stop)
                    except ValueError:
                        continue
                    label_mapped = remap_label(label)
                    if label_mapped in ['musc', 'eyem', 'elec', 'chew', 'shiv']:
                        events.append({'start': start, 'stop': stop, 'label': label_mapped})
            if not events:
                continue
            # Simulate window extraction
            for ev in events:
                start = ev['start']
                window_start = max(0, start - 0.5)
                window_end = window_start + 1.0
                if window_end > raw.times[-1]:
                    continue
                start_idx = int(window_start * fs)
                end_idx = int(window_end * fs)
                if end_idx - start_idx == int(1.0 * fs):
                    # Can extract at least one window
                    return True
    return False
