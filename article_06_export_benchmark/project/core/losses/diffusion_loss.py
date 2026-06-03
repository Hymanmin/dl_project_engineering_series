from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from core.registry import LOSSES


@LOSSES.register("toy_diffusion_loss")
class ToyDiffusionLoss(nn.Module):
    def forward(self, outputs: dict):
        return F.mse_loss(outputs["pred"], outputs["target"])
