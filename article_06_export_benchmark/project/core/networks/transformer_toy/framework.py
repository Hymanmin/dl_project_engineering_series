from __future__ import annotations

import torch
from torch import nn

from core.registry import NETWORKS


@NETWORKS.register("toy_transformer_classifier")
class ToyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32, num_heads: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.head(x[:, 0])
