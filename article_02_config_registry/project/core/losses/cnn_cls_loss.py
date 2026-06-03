from __future__ import annotations

from torch import nn

from core.registry import LOSSES


@LOSSES.register("cnn_cls_loss")
class CnnClsLoss(nn.CrossEntropyLoss):
    pass


