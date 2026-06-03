from __future__ import annotations

from torch import nn

from core.networks.cnn_cls.backbone import CnnBackbone
from core.networks.cnn_cls.head import CnnHead
from core.networks.cnn_cls.neck import CnnNeck
from core.registry import NETWORKS


@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = CnnBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
        self.neck = CnnNeck(hidden_dim=hidden_dim)
        self.head = CnnHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)


@NETWORKS.register("cnn_cls_tiny_network")
class CnnClsTinyNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = CnnBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
        self.head = CnnHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


