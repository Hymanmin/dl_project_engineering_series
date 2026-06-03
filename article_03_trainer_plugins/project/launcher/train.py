from __future__ import annotations

from core.registry import build_trainer


def run_train(cfg: dict, logger) -> None:
    trainer = build_trainer(cfg.get("trainer", {}), cfg, logger)
    trainer.train()
