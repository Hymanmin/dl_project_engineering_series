from __future__ import annotations

import unittest
from pathlib import Path

from config.loader import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_base_inheritance_and_override(self) -> None:
        cfg = load_config(Path("config/cnn_cls_train.yaml"), ["train.epochs=1", "optimizer.lr=0.01"])
        self.assertEqual(cfg["task"], "cnn_cls")
        self.assertEqual(cfg["train"]["epochs"], 1)
        self.assertAlmostEqual(cfg["optimizer"]["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
