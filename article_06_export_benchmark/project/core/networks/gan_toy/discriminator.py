from __future__ import annotations

from torch import nn


class ToyDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)
