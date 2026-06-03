from __future__ import annotations

from torch import nn


class CnnHead(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


