from __future__ import annotations

import torch

from core.models.base_model import BaseModel
from core.networks.cnn_cls.framework import CnnClsNetwork
from core.registry import MODELS, build_network

_ = CnnClsNetwork


@MODELS.register("cnn_cls_model")
class CnnClsModel(BaseModel):
    def __init__(self, cfg: dict, num_classes: int = 2) -> None:
        super().__init__()
        network_cfg = dict(cfg.get("network", {}))
        network_cfg["num_classes"] = num_classes
        self.network = build_network(network_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


