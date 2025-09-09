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


def sample_wgan(G: WGANGPGenerator, cfg: dict, device: torch.device, n: int, class_id: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    z_dim = int(cfg['model']['z_dim'])
    num_classes = int(cfg['model']['num_classes'])
    bs = min(n, 256)
    samples = []
    labels = []
    with torch.no_grad():
        while len(samples) * bs < n:
            cur = min(bs, n - len(samples) * bs)
            z = torch.randn(cur, z_dim, device=device)
            if class_id is not None:
                y = torch.full((cur,), class_id, device=device)
            else:
                y = torch.randint(0, num_classes, (cur,), device=device)
            x = G(z, y)
            samples.append(x.cpu())
            labels.append(y.cpu())
    return torch.cat(samples, dim=0)[:n], torch.cat(labels, dim=0)[:n]


def sample_ddpm(net: UNet1D, cfg: dict, device: torch.device, n: int, steps: int | None = None, class_id: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Basic ancestral DDPM sampling using training betas
    channels = int(cfg['model']['channels'])
    length = int(cfg['model']['length'])
    num_classes = int(cfg['model']['num_classes'])
    T = 1000
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64, device=device)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    betas = cosine_beta_schedule(T, s=0.008)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    objective = cfg.get('sampling', {}).get('objective', 'noise')
    sampler = cfg.get('sampling', {}).get('sampler', 'ancestral')

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
            if class_id is not None:
                y = torch.full((cur,), class_id, device=device)
            else:
                y = torch.randint(0, num_classes, (cur,), device=device)
            # loop from high t -> low t
            for t_idx in reversed(indices.tolist()):
                t = torch.full((cur,), t_idx, device=device, dtype=torch.long)
                # coefficients per DDPM posterior
                a_t = alphas[t_idx]
                # Always assign a_cum and a_cum_prev for all t_idx
                if t_idx == 0:
                    a_cum = torch.tensor(1.0, device=device)
                    a_cum_prev = torch.tensor(1.0, device=device)
                else:
                    a_cum = alphas_cumprod[t_idx]
                    a_cum_prev = alphas_cumprod[t_idx - 1]
                eps = None
                if t_idx == 0:
                    eps = torch.zeros_like(x)
                else:
                    if objective == 'v_prediction':
                        v = net(x, t, y)
                        eps = (torch.sqrt(a_cum) * v + torch.sqrt(1 - a_cum) * x) / torch.sqrt(1 - a_cum)
                    else:
                        eps = net(x, t, y)
                if sampler == 'ancestral':
                    if t_idx > 0:
                        beta_t = betas[t_idx]
                        coef1 = 1.0 / torch.sqrt(a_t)
                        coef2 = beta_t / torch.sqrt(1.0 - a_cum)
                        mean = coef1 * (x - coef2 * eps)
                        noise = torch.randn_like(x)
                        x = mean + torch.sqrt(beta_t) * noise
                else: # ddim
                    x = torch.sqrt(a_cum_prev) * (x - torch.sqrt(1 - a_cum) * eps) / torch.sqrt(a_cum) + torch.sqrt(1 - a_cum_prev) * eps
            samples.append(x.cpu())
            labels.append(y.cpu())

    return torch.cat(samples, dim=0)[:n], torch.cat(labels, dim=0)[:n]


def generate_samples(cfg: dict, ckpt_path: str, device: torch.device, model_kind: str, n: int = 128, class_id: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n samples using the checkpoint and return numpy arrays (samples, labels).

    samples shape: (n, channels, length)
    labels shape: (n,)
    """
    model_kind = model_kind.lower()
    try:
        if model_kind.startswith('wgan'):
            G = load_wgan_generator(cfg, ckpt_path, device)
            x, y = sample_wgan(G, cfg, device, n, class_id)
        elif model_kind.startswith('ddpm'):
            net = load_ddpm_unet(cfg, ckpt_path, device)
            x, y = sample_ddpm(net, cfg, device, n, class_id=class_id)
        else:
            raise ValueError(f'Unknown model kind: {model_kind}')
    except Exception as e:
        print(f"Error during generation for {model_kind}: {e}")
        import traceback
        traceback.print_exc()
        return np.empty((0,)), np.empty((0,))

    x_np = x.numpy()
    y_np = y.numpy()
    out_dir = Path('C:/works/ArtifactGen/results/generated')
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(ckpt_path).stem
    suffix = f'_class{class_id}' if class_id is not None else ''
    np.save(out_dir / f'{base}_samples{suffix}.npy', x_np)
    np.save(out_dir / f'{base}_labels{suffix}.npy', y_np)
    return x_np, y_np
