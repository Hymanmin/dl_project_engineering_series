from __future__ import annotations

from torch import nn


class CnnBackbone(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


