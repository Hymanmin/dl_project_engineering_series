from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, cfg: dict, logger) -> None:
        self.cfg = cfg
        self.logger = logger

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
