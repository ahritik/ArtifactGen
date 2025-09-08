from __future__ import annotations

import os
from pathlib import Path
import math
import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple

from src.models import UNet1D, WGANGPGenerator


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_wgan_generator(cfg: dict, ckpt_path: str, device: torch.device) -> WGANGPGenerator:
    channels = int(cfg['model']['channels'])
    length = int(cfg['model']['length'])
    num_classes = int(cfg['model']['num_classes'])
    z_dim = int(cfg['model']['z_dim'])
    G = WGANGPGenerator(z_dim, channels, length, num_classes).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    # support wrapper dicts
    if isinstance(ck, dict) and 'model' in ck:
        G.load_state_dict(ck['model'])
    else:
        try:
            G.load_state_dict(ck)
        except Exception:
            # maybe generator key present
            if isinstance(ck, dict) and 'G' in ck:
                G.load_state_dict(ck['G'])
    G.eval()
    return G


def load_ddpm_unet(cfg: dict, ckpt_path: str, device: torch.device) -> UNet1D:
    num_classes = int(cfg['model']['num_classes'])
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
        if isinstance(ck, dict) and 'model' in ck:
            net.load_state_dict(ck['model'])
        else:
            net.load_state_dict(ck)
    except Exception:
        # best-effort: if names differ, try to find 'net' or 'unet' keys
        if isinstance(ck, dict):
            for k in ('net', 'model_state', 'unet'):
                if k in ck and isinstance(ck[k], dict):
                    net.load_state_dict(ck[k])
                    break
    net.eval()
    return net


def sample_wgan(G: WGANGPGenerator, cfg: dict, device: torch.device, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    z_dim = int(cfg['model']['z_dim'])
    num_classes = int(cfg['model']['num_classes'])
    bs = min(n, 256)
    samples = []
    labels = []
    with torch.no_grad():
        while len(samples) * bs < n:
            cur = min(bs, n - len(samples) * bs)
            z = torch.randn(cur, z_dim, device=device)
            y = torch.randint(0, num_classes, (cur,), device=device)
            x = G(z, y)
            samples.append(x.cpu())
            labels.append(y.cpu())
    return torch.cat(samples, dim=0)[:n], torch.cat(labels, dim=0)[:n]


def sample_ddpm(net: UNet1D, cfg: dict, device: torch.device, n: int, steps: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Basic ancestral DDPM sampling using training betas
    channels = int(cfg['model']['channels'])
    length = int(cfg['model']['length'])
    num_classes = int(cfg['model']['num_classes'])
    T = 1000
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # If user requested fewer steps, create a coarse schedule by selecting indices
    if steps is None:
        steps = int(cfg.get('sampling', {}).get('steps', 50))
    if steps < T:
        indices = torch.linspace(0, T - 1, steps, dtype=torch.long, device=device)
    else:
        indices = torch.arange(0, T, device=device, dtype=torch.long)

    samples = []
    labels = []
    bs = min(n, 64)
    with torch.no_grad():
        while len(samples) * bs < n:
            cur = min(bs, n - len(samples) * bs)
            x = torch.randn(cur, channels, length, device=device)
            y = torch.randint(0, num_classes, (cur,), device=device)
            # loop from high t -> low t
            for t_idx in reversed(indices.tolist()):
                t = torch.full((cur,), t_idx, device=device, dtype=torch.long)
                # model predicts noise
                eps = net(x, t, y)
                # coefficients per DDPM posterior
                a_t = alphas[t_idx]
                a_cum = alphas_cumprod[t_idx]
                if t_idx > 0:
                    beta_t = betas[t_idx]
                    coef1 = 1.0 / math.sqrt(a_t)
                    coef2 = beta_t / math.sqrt(1.0 - a_cum)
                    mean = coef1 * (x - coef2 * eps)
                    noise = torch.randn_like(x)
                    x = mean + math.sqrt(beta_t) * noise
                else:
                    # final step: take the mean (no noise)
                    coef1 = 1.0 / math.sqrt(a_t)
                    coef2 = beta_t = float(betas[0]) if len(betas) > 0 else 0.0
                    coef2 = beta_t / math.sqrt(1.0 - a_cum) if a_cum < 1.0 else 0.0
                    x = (1.0 / math.sqrt(a_t)) * (x - coef2 * eps)
            samples.append(x.cpu())
            labels.append(y.cpu())

    return torch.cat(samples, dim=0)[:n], torch.cat(labels, dim=0)[:n]


def generate_samples(cfg: dict, ckpt_path: str, device: torch.device, model_kind: str, n: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n samples using the checkpoint and return numpy arrays (samples, labels).

    samples shape: (n, channels, length)
    labels shape: (n,)
    """
    model_kind = model_kind.lower()
    if model_kind.startswith('wgan'):
        G = load_wgan_generator(cfg, ckpt_path, device)
        x, y = sample_wgan(G, cfg, device, n)
    elif model_kind.startswith('ddpm'):
        net = load_ddpm_unet(cfg, ckpt_path, device)
        x, y = sample_ddpm(net, cfg, device, n)
    else:
        raise ValueError(f'Unknown model kind: {model_kind}')

    x_np = x.numpy()
    y_np = y.numpy()
    out_dir = Path('results/generated')
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(ckpt_path).stem
    np.save(out_dir / f'{base}_samples.npy', x_np)
    np.save(out_dir / f'{base}_labels.npy', y_np)
    return x_np, y_np
