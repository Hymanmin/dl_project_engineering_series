from __future__ import annotations

import torch
from torch import nn

from core.registry import NETWORKS


@NETWORKS.register("toy_denoiser")
class ToyDenoiser(nn.Module):
    def __init__(self, data_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(1000, hidden_dim)
        self.net = nn.Sequential(nn.Linear(data_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, data_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t)
        return self.net(torch.cat([x, emb], dim=1))
