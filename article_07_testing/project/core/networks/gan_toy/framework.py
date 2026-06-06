from __future__ import annotations

from torch import nn

from core.networks.gan_toy.discriminator import ToyDiscriminator
from core.networks.gan_toy.generator import ToyGenerator
from core.registry import NETWORKS


@NETWORKS.register("toy_gan_network")
class ToyGANNetwork(nn.Module):
    def __init__(self, noise_dim: int = 16, data_dim: int = 16) -> None:
        super().__init__()
        self.generator = ToyGenerator(noise_dim=noise_dim, output_dim=data_dim)
        self.discriminator = ToyDiscriminator(input_dim=data_dim)
