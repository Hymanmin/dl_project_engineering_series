from __future__ import annotations

import unittest

import torch

from core.imports import import_modules
from core.registry import build_model


class ModelForwardTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        cfg = {
            "imports": ["core.networks.cnn_cls.framework", "core.models.cnn_cls.cnn_cls_model"],
            "network": {"name": "cnn_cls_network", "input_dim": 16, "hidden_dim": 32, "num_classes": 2},
        }
        import_modules(cfg["imports"])
        model = build_model({"name": "cnn_cls_model", "num_classes": 2}, cfg)
        logits = model(torch.randn(4, 16))
        self.assertEqual(tuple(logits.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
