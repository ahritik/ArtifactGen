from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMLayer(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, in_ch)
        self.to_shift = nn.Linear(cond_dim, in_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s = self.to_scale(cond)[:, :, None]
        b = self.to_shift(cond)[:, :, None]
        return x * (1 + s) + b


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film1 = FiLMLayer(out_ch, cond_dim)
        self.film2 = FiLMLayer(out_ch, cond_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        h = self.film1(h, cond)
        h = self.dropout(F.silu(self.conv2(h)))
        h = self.film2(h, cond)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 8,
        base_channels: int = 64,
        channel_mults=(1, 2, 4),
        num_res_blocks: int = 2,
        num_classes: int = 6,
        cond_dim: int = 128,
        dropout: float = 0.1,
        use_dilations: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(128, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, 128)  # last index is null token for cfg

        # Down
        self.downs = nn.ModuleList()
        ch = base_channels
        self.in_conv = nn.Conv1d(in_ch, ch, kernel_size=3, padding=1)
        downs_channels = [ch]
        for i, m in enumerate(channel_mults):
            out_ch = base_channels * m
            blocks = nn.ModuleList()
            current_ch = ch
            for _ in range(num_res_blocks):
                rb = ResBlock(current_ch, out_ch, cond_dim, dropout, dilation=(2 ** i if use_dilations else 1))
                blocks.append(rb)
                current_ch = out_ch
            self.downs.append(blocks)
            ch = out_ch
            downs_channels.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)]))

        # Middle
        self.mid = ResBlock(ch, ch, cond_dim, dropout)

        # Up
        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * m
            blocks = nn.ModuleList()
            current_ch = ch
            for _ in range(num_res_blocks):
                rb = ResBlock(current_ch, out_ch, cond_dim, dropout)
                blocks.append(rb)
                current_ch = out_ch
            self.ups.append(blocks)
            if i != 0:
                self.ups.append(nn.ModuleList([nn.ConvTranspose1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)]))
            ch = out_ch

        self.out_conv = nn.Conv1d(ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None, guidance_p: float | None = None) -> torch.Tensor:
        # classifier-free guidance via random nulling of labels at train-time should be handled in the training loop
        if y is None:
            y_emb = torch.zeros((x.size(0), 128), device=x.device)
        else:
            y_emb = self.class_embed(y)
        t_emb = sinusoidal_time_embedding(t, 128)
        cond = self.time_mlp(t_emb + y_emb)

        hs = []
        h = self.in_conv(x)
        for block in self.downs:
            if isinstance(block, nn.ModuleList) and any(isinstance(m, ResBlock) for m in block):
                for rb in block:  # type: ignore[assignment]
                    h = rb(h, cond)
                    hs.append(h)
            else:
                # downsample
                h = block[0](h)  # type: ignore[index]
                hs.append(h)

        h = self.mid(h, cond)

        for block in self.ups:
            if isinstance(block, nn.ModuleList) and any(isinstance(m, ResBlock) for m in block):
                for rb in block:  # type: ignore[assignment]
                    h = rb(h, cond)
            else:
                h = block[0](h)  # upsample  # type: ignore[index]

        return self.out_conv(h)
