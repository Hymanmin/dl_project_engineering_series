from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TestConfig:
    max_workers: int | None = None
    progress_interval: int = 10
    timeout: int = 300
    output_dir: str = "runs/business_tests"


@dataclass
class TestResult:
    task_id: str
    status: str
    data: dict[str, Any]
    error_message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
