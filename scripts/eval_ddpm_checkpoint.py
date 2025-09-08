"""
Evaluate latest DDPM checkpoint on the validation split without interrupting training.
Writes an epoch-averaged MSE to results/val_mse.log and prints to console.

Usage (in venv):
& .\venv\Scripts\Activate.ps1
python scripts/eval_ddpm_checkpoint.py --config configs/ddpm_raw.yaml

Add --repeat SECONDS to run periodically (useful while training runs in another terminal).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.dataset import EEGWindowDataset
from src.models import UNet1D


def latest_checkpoint(checkpoints_dir: str) -> str | None:
    p = Path(checkpoints_dir)
    if not p.exists():
        return None
    files = list(p.glob('*.pth'))
    if not files:
        return None
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(files[0])


def eval_ckpt(cfg: dict, ckpt_path: str, device: torch.device) -> float:
    data_root = cfg['data']['processed_root']
    length = int(cfg['model']['length'])
    num_classes = int(cfg['model']['num_classes'])

    ds = EEGWindowDataset(data_root, split='val', normalization='zscore', length=length)
    if len(ds) == 0:
        raise RuntimeError('Validation dataset is empty; make sure manifest contains val split')
    dl = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False, num_workers=4)

    unet_cfg = cfg['model'].get('unet', {})
    net = UNet1D(
        in_ch=int(cfg['model']['channels']),
        base_channels=unet_cfg.get('base_channels', 64),
        channel_mults=tuple(unet_cfg.get('channel_mults', [1, 2, 4])),
        num_res_blocks=unet_cfg.get('num_res_blocks', 2),
        num_classes=num_classes,
        dropout=unet_cfg.get('dropout', 0.1),
        use_dilations=bool(unet_cfg.get('use_dilations', True)),
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    try:
        net.load_state_dict(ck)
    except Exception:
        # checkpoint could be {'model': state_dict} or raw
        if isinstance(ck, dict) and 'model' in ck:
            net.load_state_dict(ck['model'])
        else:
            # try direct
            net.load_state_dict(ck)

    net.eval()

    # DDPM fixed schedule used in training
    T = 1000
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            bs = x.size(0)
            t = torch.randint(0, T, (bs,), device=device)
            a_bar = alphas_cumprod[t].view(bs, 1, 1)
            noise = torch.randn_like(x)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise
            pred = net(x_t, t, y)
            loss = F.mse_loss(pred, noise, reduction='sum')
            total_loss += loss.item()
            total_count += bs * x.numel() // bs  # equals bs * C * L

    # compute mean MSE per element
    mean_mse = total_loss / (len(dl.dataset) * x.numel() // bs)
    return mean_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoints', default='results/checkpoints')
    ap.add_argument('--repeat', type=int, default=0, help='If >0, re-run every N seconds')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_path = Path('results/val_mse.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ckpt = latest_checkpoint(args.checkpoints)
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        if ckpt is None:
            print(f'[{ts}] No checkpoint found in {args.checkpoints}')
        else:
            try:
                mse = eval_ckpt(cfg, ckpt, device)
                line = f'{ts}\t{Path(ckpt).name}\t{mse:.6e}\n'
                print(line.strip())
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(line)
            except Exception as e:
                print(f'[{ts}] Evaluation failed: {e}')

        if args.repeat > 0:
            time.sleep(args.repeat)
        else:
            break


if __name__ == '__main__':
    main()
