from __future__ import annotations

import torch

from core.models.base_model import BaseModel
from core.networks.diffusion_toy.unet import ToyDenoiser
from core.registry import MODELS, build_network

_ = ToyDenoiser


@MODELS.register("toy_diffusion_model")
class ToyDiffusionModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.denoiser = build_network(cfg.get("network", {}))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x0 = batch["input"]
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (x0.shape[0],), device=x0.device)
        xt = x0 + 0.1 * noise
        pred = self.denoiser(xt, t)
        return {"pred": pred, "target": noise, "timestep": t}
