from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WGANGPGenerator(nn.Module):
    def __init__(self, z_dim: int, channels: int, length: int, num_classes: int):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.length = length
        self.num_classes = num_classes

        d = 256
        out_len = length // 8
        self.fc = nn.Linear(z_dim + num_classes, d * out_len)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(d, d // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(d // 2, d // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(d // 4, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y is class index -> one-hot
        y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        x = torch.cat([z, y_oh], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.length // 8)
        x = self.net(x)
        return x  # (B, C, T)


class ProjectionHead(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, feat_dim)

    def forward(self, feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # feat: (B, D), y: (B,)
        w_y = self.embed(y)  # (B, D)
        return torch.sum(feat * w_y, dim=1)  # (B,)


class WGANGPCritic(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        d = 64
        self.conv = nn.Sequential(
            nn.Conv1d(channels, d, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(d, d * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(d * 2, d * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d * 4, 1)
        self.proj = ProjectionHead(num_classes=num_classes, feat_dim=d * 4)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (B, D, T')
        h_pool = self.gap(h).squeeze(-1)  # (B, D)
        score = self.fc(h_pool).squeeze(-1)  # (B,)
        score += self.proj(h_pool, y)
        return score


def gradient_penalty(critic: WGANGPCritic, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, device=real.device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    scores = critic(interp, y)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp
