from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from testing.config import TestResult


class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(self, data_source: str) -> list[Any]:
        raise NotImplementedError

    def filter_data(self, data: list[Any]) -> list[Any]:
        return data

    @abstractmethod
    def prepare_task_args(self, data: Any, global_context: dict[str, Any]) -> tuple[Any, ...]:
        raise NotImplementedError


class BaseTaskProcessor(ABC):
    def initialize_worker(self) -> None:
        pass

    @abstractmethod
    def process(self, *args: Any) -> TestResult:
        raise NotImplementedError

    def cleanup_worker(self) -> None:
        pass


class BaseResultCollector(ABC):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TestResult] = []

    def collect(self, result: TestResult) -> None:
        self.results.append(result)

    @abstractmethod
    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
