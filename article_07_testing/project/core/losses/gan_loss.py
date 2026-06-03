from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import LOSSES


@LOSSES.register("toy_gan_loss")
class ToyGANLoss(nn.Module):
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real_loss + fake_loss

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use d_loss() and g_loss() for GAN training.")
