from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WGANGPGenerator(nn.Module):
    """
    Generator network for WGAN-GP that produces synthetic EEG windows from noise and class labels.

    The generator takes a latent vector z concatenated with a one-hot class label, upsamples
    through transposed convolutions to produce an EEG signal window of shape (channels, length).

    Args:
        z_dim (int): Dimensionality of the latent noise vector.
        channels (int): Number of EEG channels in the output.
        length (int): Length of the output signal in samples.
        num_classes (int): Number of artifact classes.
    """

    def __init__(self, z_dim: int, channels: int, length: int, num_classes: int):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.length = length
        self.num_classes = num_classes

        d = 256
        out_len = length // 8  # After 3 upsampling layers (4x each)
        self.fc = nn.Linear(z_dim + num_classes, d * out_len)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(d, d // 2, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.ReLU(True),
            nn.ConvTranspose1d(d // 2, d // 4, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.ReLU(True),
            nn.ConvTranspose1d(d // 4, channels, kernel_size=4, stride=2, padding=1),  # Upsample x2, output channels
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Concatenate noise z with one-hot class label y
        y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        x = torch.cat([z, y_oh], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.length // 8)
        x = self.net(x)
        return x  # Shape: (batch, channels, length)


class ProjectionHead(nn.Module):
    """
    Projection head for conditional critic to incorporate class information.

    This module embeds class labels and computes a projection score to condition
    the critic on the artifact type.

    Args:
        num_classes (int): Number of artifact classes.
        feat_dim (int): Dimensionality of the feature vector to project.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, feat_dim)

    def forward(self, feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Embed class label and compute dot product with features
        w_y = self.embed(y)  # (batch, feat_dim)
        return torch.sum(feat * w_y, dim=1)  # (batch,)


class WGANGPCritic(nn.Module):
    """
    Critic (discriminator) network for WGAN-GP that scores real/fake EEG windows.

    The critic uses convolutional layers to extract features, followed by global average pooling
    and a linear layer with conditional projection for class-aware scoring.

    Args:
        channels (int): Number of EEG channels in the input.
        num_classes (int): Number of artifact classes.
    """

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        d = 64
        self.conv = nn.Sequential(
            nn.Conv1d(channels, d, kernel_size=5, stride=2, padding=2),  # Downsample x2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(d, d * 2, kernel_size=5, stride=2, padding=2),  # Downsample x2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(d * 2, d * 4, kernel_size=5, stride=2, padding=2),  # Downsample x2
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling to (batch, d*4, 1)
        self.fc = nn.Linear(d * 4, 1)  # Output scalar score
        self.proj = ProjectionHead(num_classes=num_classes, feat_dim=d * 4)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (batch, d*4, length')
        h_pool = self.gap(h).squeeze(-1)  # (batch, d*4)
        score = self.fc(h_pool).squeeze(-1)  # (batch,)
        score += self.proj(h_pool, y)  # Add conditional projection
        return score  # (batch,)


def gradient_penalty(critic: WGANGPCritic, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient penalty for WGAN-GP to enforce Lipschitz continuity.

    Interpolates between real and fake samples, computes critic gradients w.r.t. interpolation,
    and penalizes gradients with norm != 1.

    Args:
        critic: The critic network.
        real: Real EEG samples.
        fake: Fake EEG samples.
        y: Class labels.

    Returns:
        Gradient penalty loss.
    """
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, device=real.device)  # Random interpolation weights
    interp = eps * real + (1 - eps) * fake  # Interpolated samples
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
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()  # Penalize ||grad|| != 1
    return gp
