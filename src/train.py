from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm

from .dataset import EEGWindowDataset
from .models import WGANGPGenerator, WGANGPCritic, UNet1D
from .models.wgan import gradient_penalty


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    return ap.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_wgan(cfg, device: torch.device):
    data_root = cfg['data']['processed_root']
    num_classes = int(cfg['model']['num_classes'])
    channels = int(cfg['model']['channels'])
    length = int(cfg['model']['length'])

    # Create dataset and dataloader for training
    ds = EEGWindowDataset(data_root, split='train', normalization='wgan_minmax')
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training'].get('num_workers', 4), pin_memory=True)

    # Initialize generator and critic models
    G = WGANGPGenerator(cfg['model']['z_dim'], channels, length, num_classes).to(device)
    D = WGANGPCritic(channels, num_classes).to(device)

    # Optimizers for generator and critic
    opt_g = optim.Adam(G.parameters(), lr=float(cfg['training']['lr']), betas=tuple(cfg['training']['betas']), weight_decay=0.0)
    opt_d = optim.Adam(D.parameters(), lr=float(cfg['training']['lr']), betas=tuple(cfg['training']['betas']), weight_decay=0.0)

    # Training hyperparameters
    n_critic = int(cfg['training']['n_critic']) 
    gp_lambda = float(cfg['training']['grad_penalty_lambda']) 
    spec_w = float(cfg['training']['spectral_loss_weight']) 
    patience = int(cfg['training'].get('patience', 20))  # Early stopping patience

    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir='results/tensorboard/wgan')

    # Early stopping variables
    best_loss_g = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in tqdm.tqdm(range(cfg['training']['epochs']), desc='WGAN Training'):
        epoch_loss_g = 0.0
        for _i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            bs = x.size(0)

            # Train critic multiple times per generator step
            for _ in range(n_critic):
                z = torch.randn(bs, cfg['model']['z_dim'], device=device)
                x_fake = G(z, y)
                d_real = D(x, y).mean()
                d_fake = D(x_fake.detach(), y).mean()
                gp = gradient_penalty(D, x, x_fake.detach(), y)
                loss_d = -(d_real - d_fake) + gp_lambda * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # Train generator
            z = torch.randn(bs, cfg['model']['z_dim'], device=device)
            x_fake = G(z, y)
            d_fake = D(x_fake, y).mean()

            # Optional spectral loss for better frequency domain matching
            spec_loss = 0.0
            if spec_w > 0:
                window = torch.hann_window(64, device=x.device)
                xr = torch.view_as_real(torch.stft(x.view(bs * x.size(1), x.size(2)), n_fft=64, window=window, return_complex=True))
                xf = torch.view_as_real(torch.stft(x_fake.view(bs * x_fake.size(1), x_fake.size(2)), n_fft=64, window=window, return_complex=True))
                spec_loss = F.l1_loss(xf, xr)

            loss_g = -d_fake + spec_w * spec_loss
            epoch_loss_g += loss_g.item()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            # Log losses to TensorBoard
            writer.add_scalar('Loss/D', loss_d.item(), epoch)
            writer.add_scalar('Loss/G', loss_g.item(), epoch)
            if spec_w > 0:
                writer.add_scalar('Loss/Spectral', spec_loss.item(), epoch)

        # Average generator loss for the epoch
        epoch_loss_g /= len(dl)  # Average loss per epoch

        # Early stopping check based on generator loss
        if epoch_loss_g < best_loss_g:
            best_loss_g = epoch_loss_g
            patience_counter = 0
            # Save best model
            torch.save(G.state_dict(), 'results/checkpoints/wgan_generator_best.pth')
            torch.save(D.state_dict(), 'results/checkpoints/wgan_critic_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in generator loss.')
                break

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'[WGAN] Epoch {epoch+1} | D: {loss_d.item():.4f} G: {loss_g.item():.4f}')

    writer.close()


def train_ddpm(cfg, device: torch.device):
    data_root = cfg['data']['processed_root']
    num_classes = int(cfg['model']['num_classes'])
    channels = int(cfg['model']['channels'])

    ds = EEGWindowDataset(data_root, split='train', normalization='zscore')
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training'].get('num_workers', 4), pin_memory=True)

    unet_cfg = cfg['model'].get('unet', {})
    net = UNet1D(
        in_ch=channels,
        base_channels=unet_cfg.get('base_channels', 64),
        channel_mults=tuple(unet_cfg.get('channel_mults', [1, 2, 4])),
        num_res_blocks=unet_cfg.get('num_res_blocks', 2),
        num_classes=num_classes,
        dropout=unet_cfg.get('dropout', 0.1),
        use_dilations=bool(unet_cfg.get('use_dilations', True)),
    ).to(device)

    opt = optim.AdamW(net.parameters(), lr=float(cfg['training']['lr']), weight_decay=0.0)
    T = 1000
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    patience = int(cfg['training'].get('patience', 20))  # Early stopping patience

    writer = SummaryWriter(log_dir='results/tensorboard/ddpm')

    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm.tqdm(range(cfg['training']['epochs']), desc='DDPM Training'):
        epoch_loss = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            bs = x.size(0)

            t = torch.randint(0, T, (bs,), device=device)
            a_bar = alphas_cumprod[t].view(bs, 1, 1)
            noise = torch.randn_like(x)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise

            pred = net(x_t, t, y)
            loss = F.mse_loss(pred, noise)
            epoch_loss += loss.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            writer.add_scalar('Loss/MSE', loss.item(), epoch)

        epoch_loss /= len(dl)  # Average loss per epoch

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # Save best model
            torch.save(net.state_dict(), 'results/checkpoints/ddpm_unet_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in MSE loss.')
                break

        if (epoch + 1) % 10 == 0:
            print(f'[DDPM] Epoch {epoch+1} | MSE: {loss.item():.4f}')

    writer.close()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('experiment', {}).get('seed', 42))
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model_kind = cfg['training']['model']
    os.makedirs('results/checkpoints', exist_ok=True)

    if model_kind == 'wgan_gp':
        train_wgan(cfg, device)
    elif model_kind == 'ddpm':
        train_ddpm(cfg, device)
    else:
        raise ValueError(f'Unknown model: {model_kind}')


if __name__ == '__main__':
    main()
