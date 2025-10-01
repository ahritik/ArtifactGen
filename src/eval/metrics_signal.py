from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np  # type: ignore
import yaml  # type: ignore
import torch
from scipy.signal import welch
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help='Path to model checkpoint to sample for evaluation')
    ap.add_argument("--model-kind", type=str, default='ddpm', help='Model kind: ddpm or wgan_gp')
    ap.add_argument("--n", type=int, default=128, help='Number of samples to generate')
    return ap.parse_args()


def _band_edges() -> Dict[str, Tuple[float, float]]:
    return {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),  # extended high-frequency band for artifact energy
    }


def power_bands(x: np.ndarray, fs: int) -> Dict[str, float]:
    """Compute relative band power for an array shaped (C, T)."""
    bands = _band_edges()
    freqs, psd = welch(x, fs=fs, axis=-1, nperseg=min(256, x.shape[-1]))
    psd = psd.sum(axis=0)  # sum channels
    total = float(np.trapz(psd, freqs)) + 1e-12
    out: Dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        band_power = float(np.trapz(psd[m], freqs[m])) / total
        out[name] = band_power
    return out


def channel_correlation(x_real: np.ndarray, x_gen: np.ndarray) -> float:
    """Mean Pearson correlation across channels for (C,T) arrays."""
    C = min(x_real.shape[0], x_gen.shape[0])
    xr = x_real[:C]
    xg = x_gen[:C]
    xr = (xr - xr.mean(axis=-1, keepdims=True)) / (xr.std(axis=-1, keepdims=True) + 1e-8)
    xg = (xg - xg.mean(axis=-1, keepdims=True)) / (xg.std(axis=-1, keepdims=True) + 1e-8)
    num = (xr * xg).sum(axis=-1)
    den = np.sqrt((xr**2).sum(axis=-1) * (xg**2).sum(axis=-1)) + 1e-8
    return float((num / den).mean())


def psd_l2_distance(x_real: np.ndarray, x_gen: np.ndarray, fs: int) -> float:
    fr, pr = welch(x_real, fs=fs, axis=-1, nperseg=min(256, x_real.shape[-1]))
    fg, pg = welch(x_gen, fs=fs, axis=-1, nperseg=min(256, x_gen.shape[-1]))
    pr = pr.sum(axis=0)
    pg = pg.sum(axis=0)
    if not np.array_equal(fr, fg):
        pg = np.interp(fr, fg, pg)
    return float(np.linalg.norm(pr - pg) / (np.linalg.norm(pr) + 1e-12))


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fs = int(cfg["data"]["sample_rate"])  # noqa: F841
    manifest = Path("results/manifest.json")
    if not manifest.exists():
        print("No manifest found.")
        return
    with open(manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items; computing signal metrics…")

    # Load a batch of real samples from test split
    test_paths = [r["path"] for r in items if r.get("split") == "test"]
    if not test_paths:
        print("Manifest has no test items; cannot compute real-vs-generated metrics.")
        return
    K = min(256, len(test_paths))
    real_arrs = []
    for p in test_paths[:K]:
        with np.load(p) as d:
            real_arrs.append(d['array'])
    x_real = np.stack(real_arrs, axis=0).mean(axis=0)  # (C,T)

    # Find latest generated samples
    gen_dir = Path("results/generated")
    gen_files = sorted(gen_dir.glob("*samples*.npy")) if gen_dir.exists() else []
    if not gen_files and (args.ckpt is None):
        print("No generated samples found; provide --ckpt to generate on the fly.")

    # If checkpoint provided, generate fresh samples
    if args.ckpt is not None:
        from src.eval.generate import generate_samples
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Generating {args.n} samples from checkpoint {args.ckpt} using {args.model_kind}...')
        xs, _ = generate_samples(cfg, args.ckpt, device, args.model_kind, n=args.n)
        x_gen = xs.mean(axis=0)
    else:
        if not gen_files:
            return
        xs = np.load(gen_files[-1])
        x_gen = xs.mean(axis=0)

    # Compute metrics on means (robust to misalignment)
    bp_real = power_bands(x_real, fs)
    bp_gen = power_bands(x_gen, fs)
    bands = list(_band_edges().keys())
    bp_err = {f"bp_rel_err_{b}": abs(bp_gen[b] - bp_real[b])/(bp_real[b] + 1e-12) for b in bands}
    ch_corr = channel_correlation(x_real, x_gen)
    psd_dist = psd_l2_distance(x_real, x_gen, fs)

    df = pd.DataFrame([{**{"ch_corr": ch_corr, "psd_l2": psd_dist}, **bp_err}])
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "signal_metrics.csv", index=False)
    with open(out_dir / "table_bandpower.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Wrote results/signal_metrics.csv and results/table_bandpower.tex")



if __name__ == "__main__":
    main()
