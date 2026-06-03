from __future__ import annotations

import unittest

from core.imports import import_modules
from core.registry import build_loss, build_model


class RegistryTest(unittest.TestCase):
    def test_build_registered_components(self) -> None:
        cfg = {
            "imports": [
                "core.dataset.cnn_cls.cnn_cls_dataset",
                "core.networks.cnn_cls.framework",
                "core.losses.cnn_cls_loss",
                "core.models.cnn_cls.cnn_cls_model",
            ],
            "network": {"name": "cnn_cls_network", "input_dim": 16, "hidden_dim": 32, "num_classes": 2},
        }
        import_modules(cfg["imports"])
        model = build_model({"name": "cnn_cls_model", "num_classes": 2}, cfg)
        loss = build_loss({"name": "cnn_cls_loss"})
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
