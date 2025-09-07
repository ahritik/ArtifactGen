"""Validate processed data and splits against model expectations.
Checks:
 - processed windows (.npz) exist in data/processed/train
 - each window has expected shape: (channels, length)
 - sample rate / window length vs config
 - splits CSV subjects map to processed subjects
 - class_map vs config.model.num_classes

Usage: python src/validate_splits.py --config configs/wgan_raw.yaml
"""
import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import yaml
except Exception:
    yaml = None


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the config (install pyyaml)")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def inspect_npz(npz_path: Path):
    info = {'path': str(npz_path)}
    try:
        with np.load(str(npz_path), allow_pickle=True) as data:
            keys = list(data.keys())
            info['keys'] = keys
            # Heuristics: find first array that looks like signal
            arr = None
            for k in ['data', 'x', 'window', 'signal']:
                if k in data:
                    arr = data[k]
                    info['signal_key'] = k
                    break
            if arr is None and keys:
                arr = data[keys[0]]
                info['signal_key'] = keys[0]
            if arr is None:
                info['shape'] = None
            else:
                info['shape'] = arr.shape
            # try metadata
            md = {}
            for mkey in ['subject', 'subject_id', 'meta', 'info']:
                if mkey in data:
                    md[mkey] = data[mkey].tolist() if hasattr(data[mkey], 'tolist') else data[mkey]
            info['meta'] = md
    except Exception as e:
        info['error'] = repr(e)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='configs/wgan_raw.yaml')
    args = p.parse_args()

    cfg_path = Path(args.config)
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print('Failed to load config:', e)
        sys.exit(2)

    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})

    processed_root = Path(data_cfg.get('processed_root', 'data/processed')).resolve()
    train_dir = processed_root / 'train'

    # Helper: resolve configured CSV path with sensible fallbacks so users
    # don't hit a 'not found' when the config uses a relative path or the
    # file exists under the processed root with a slightly different name.
    def resolve_csv(cfg_key: str, default_name: str, alternates=None):
        alternates = alternates or []
        cfg_val = data_cfg.get(cfg_key)
        tried = []
        # 1) If config explicitly set a path, try it first
        if cfg_val:
            p = Path(cfg_val)
            tried.append(p)
            if p.exists():
                return p.resolve(), tried
            # try relative to processed_root using only the filename
            p2 = processed_root / p.name
            tried.append(p2)
            if p2.exists():
                return p2.resolve(), tried
        # 2) Try the default name under processed_root
        p3 = processed_root / default_name
        tried.append(p3)
        if p3.exists():
            return p3.resolve(), tried
        # 3) Try any alternates under processed_root
        for a in alternates:
            pa = processed_root / a
            tried.append(pa)
            if pa.exists():
                return pa.resolve(), tried
        # 4) Not found; return the most likely path (default under processed_root)
        return p3.resolve(), tried

    split_csv, split_tried = resolve_csv('split_csv', 'suggested_splits_subjectwise_multilabel.csv', alternates=['suggested_splits_subjectwise.csv'])
    class_map_csv, class_tried = resolve_csv('class_map_csv', 'class_map.csv')

    print(f"Config: {cfg_path}")
    print(f"Processed root: {processed_root}")
    print(f"Train dir: {train_dir}")
    print(f"Split CSV (resolved): {split_csv}")
    if split_csv.exists():
        print('  Found split CSV at:', split_csv)
    else:
        print('  Split CSV not found. Tried:')
        for t in split_tried:
            print('   -', t)

    print(f"Class map CSV (resolved): {class_map_csv}")
    if class_map_csv.exists():
        print('  Found class map CSV at:', class_map_csv)
    else:
        print('  Class map CSV not found. Tried:')
        for t in class_tried:
            print('   -', t)

    npz_files = list(train_dir.rglob('*.npz')) if train_dir.exists() else []
    print(f"Found {len(npz_files)} .npz window files under {train_dir}")

    # quick checks
    expected_channels = model_cfg.get('channels')
    expected_length = model_cfg.get('length')
    # model_cfg may store channels as an int (count) or a list of names.
    if isinstance(expected_channels, int):
        expected_n_channels = expected_channels
    else:
        try:
            expected_n_channels = len(expected_channels) if expected_channels is not None else None
        except Exception:
            expected_n_channels = None
    model_num_classes = model_cfg.get('num_classes')

    sample_info = None
    shapes = {}
    subjects_found = set()
    labels_found = set()

    for npz in npz_files[:50]:  # inspect up to 50 for speed
        info = inspect_npz(npz)
        shapes.setdefault(str(info.get('shape')), 0)
        shapes[str(info.get('shape'))] += 1
        # try to extract subject
        md = info.get('meta', {})
        subj = None
        if 'subject' in md:
            subj = md['subject']
        elif 'subject_id' in md:
            subj = md['subject_id']
        else:
            # filename heuristic: take part before first underscore
            name = Path(info['path']).stem
            if '_' in name:
                subj = name.split('_')[0]
            else:
                subj = None
        if subj:
            subjects_found.add(str(subj))
        # try label
        # look for 'label' or 'y' in meta
        if 'label' in md:
            labels_found.add(str(md['label']))
        sample_info = info

    if shapes:
        print('Observed window shapes (sample):')
        for s,c in shapes.items():
            print(' ', s, 'x', c)
    else:
        print('No shapes observed (no .npz files inspected)')

    # check expected channels/length
    if expected_channels is not None:
        # Print either the list or the inferred count
        if isinstance(expected_channels, int):
            print('Model expects channels (count):', expected_channels)
        else:
            print('Model expects channels:', expected_channels)
    if expected_length is not None:
        print('Model expects length (samples):', expected_length)

    if sample_info is None:
        print('\nNo processed windows found to validate shape. Run preprocessing to generate windows.\n')
    else:
        print('\nSample inspected NPZ info:')
        for k,v in sample_info.items():
            print(' ', k+':', v)
        # shape check
        shp = sample_info.get('shape')
        if shp and expected_n_channels and expected_length:
            # possible layouts
            ok = False
            if len(shp) == 2:
                if shp[0] == expected_n_channels and shp[1] == expected_length:
                    ok = True
                if shp[1] == expected_n_channels and shp[0] == expected_length:
                    ok = True
            print('Shape matches model expected (channels x length)?', ok)

    # splits exist?
    if split_csv.exists():
        import pandas as pd
        try:
            df = pd.read_csv(split_csv)
            print(f"Split CSV rows: {len(df)}")
            # extract subjects
            if 'subject_id' in df.columns:
                split_subjects = set(df['subject_id'].astype(str).unique())
            elif 'subject' in df.columns:
                split_subjects = set(df['subject'].astype(str).unique())
            else:
                split_subjects = set()
            print('Subjects in split CSV:', len(split_subjects))
            if subjects_found:
                missing = split_subjects - subjects_found
                if missing:
                    print('WARNING: The following subjects are present in split CSV but have NO processed windows (sample):')
                    print('  ', list(missing)[:20])
                else:
                    print('All split subjects appear in processed windows (sample check).')
        except Exception as e:
            print('Failed to read split CSV:', e)
    else:
        print('Split CSV not found:', split_csv)

    # class map check
    if class_map_csv.exists():
        import csv
        with open(class_map_csv, 'r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            classes = [r for r in rdr]
        print('Class map entries:', len(classes))
        if model_num_classes is not None:
            if len(classes) != model_num_classes:
                print(f"WARNING: class_map has {len(classes)} entries but model expects {model_num_classes} classes")
            else:
                print('Class map size matches model.num_classes')
    else:
        print('Class map CSV not found:', class_map_csv)

    print('\nValidation complete.')

if __name__ == '__main__':
    main()
