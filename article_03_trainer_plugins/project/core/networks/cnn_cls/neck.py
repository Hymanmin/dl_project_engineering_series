from __future__ import annotations

from torch import nn


class CnnNeck(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


