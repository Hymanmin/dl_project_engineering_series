from __future__ import annotations

from core.models.base_model import BaseModel
from core.networks.gan_toy.framework import ToyGANNetwork
from core.registry import MODELS, build_network

_ = ToyGANNetwork


@MODELS.register("toy_gan_model")
class ToyGANModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.network = build_network(cfg.get("network", {}))

    @property
    def generator(self):
        return self.network.generator

    @property
    def discriminator(self):
        return self.network.discriminator

    def forward(self, z):
        return self.generator(z)
