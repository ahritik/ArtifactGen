from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help='Path to model checkpoint to sample for evaluation')
    ap.add_argument("--model-kind", type=str, default='ddpm', help='Model kind: ddpm or wgan_gp')
    ap.add_argument("--n", type=int, default=128, help='Number of samples to generate')
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) if yaml is not None else {}
    manifest = Path("results/manifest.json")
    if not manifest.exists():
        print("No manifest found.")
        return
    with open(manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items; compute KID/MMD/PRD via encoder features here…")

    if args.ckpt is not None:
        from src.eval.generate import generate_samples
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Generating {args.n} samples from checkpoint {args.ckpt} using {args.model_kind}...')
        generate_samples(_cfg, args.ckpt, device, args.model_kind, n=args.n)


if __name__ == "__main__":
    main()
