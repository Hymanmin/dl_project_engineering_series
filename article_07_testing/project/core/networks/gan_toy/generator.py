from __future__ import annotations

from torch import nn


class ToyGenerator(nn.Module):
    def __init__(self, noise_dim: int = 16, output_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_dim, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, z):
        return self.net(z)
