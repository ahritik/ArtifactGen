from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # type: ignore
import yaml  # type: ignore
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help='Path to model checkpoint to sample for evaluation')
    ap.add_argument("--model-kind", type=str, default='ddpm', help='Model kind: ddpm or wgan_gp')
    ap.add_argument("--n", type=int, default=128, help='Number of samples to generate')
    return ap.parse_args()


def power_bands(_x: np.ndarray, _fs: int) -> dict:
    # Placeholder: compute band powers (delta/theta/alpha/beta)
    return {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0}


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _fs = int(cfg["data"]["sample_rate"])  # noqa: F841
    manifest = Path("results/manifest.json")
    if not manifest.exists():
        print("No manifest found.")
        return
    with open(manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items; compute signal metrics here")

    # If checkpoint provided, generate samples for evaluation
    if args.ckpt is not None:
        from src.eval.generate import generate_samples
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Generating {args.n} samples from checkpoint {args.ckpt} using {args.model_kind}...')
        generate_samples(cfg, args.ckpt, device, args.model_kind, n=args.n)


if __name__ == "__main__":
    main()
