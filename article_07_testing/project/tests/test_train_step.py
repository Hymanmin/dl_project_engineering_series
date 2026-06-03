from __future__ import annotations

import unittest
from pathlib import Path

from config.loader import load_config
from core.imports import import_modules
from core.registry import build_trainer
from utils.logger import setup_logger


class TrainStepTest(unittest.TestCase):
    def test_one_epoch_smoke(self) -> None:
        cfg = load_config(Path("config/cnn_cls_train.yaml"), ["train.epochs=1", "train.print_freq=100"])
        cfg["runtime"]["checkpoint_dir"] = "runs/test_checkpoints"
        cfg["runtime"]["log_dir"] = "runs/test_logs"
        import_modules(cfg.get("imports", []))
        trainer = build_trainer(cfg["trainer"], cfg, setup_logger(cfg["runtime"]["log_dir"]))
        trainer.train()
        self.assertTrue(Path("runs/test_checkpoints/cnn_cls.pt").exists())


if __name__ == "__main__":
    unittest.main()
