from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from typing import Any

from testing.base import BaseDataLoader, BaseResultCollector, BaseTaskProcessor
from testing.config import TestConfig, TestResult


def _worker_wrapper(
    processor_class: type[BaseTaskProcessor],
    task_args: tuple[Any, ...],
) -> TestResult:
    start = time.perf_counter()
    processor = processor_class()
    try:
        processor.initialize_worker()
        result = processor.process(*task_args)
        result.execution_time = time.perf_counter() - start
        return result
    except Exception as exc:  # noqa: BLE001 - convert child-process failures to data.
        return TestResult(
            task_id=str(task_args[0]) if task_args else "<unknown>",
            status="error",
            data={},
            error_message=f"{type(exc).__name__}: {exc}",
            execution_time=time.perf_counter() - start,
        )
    finally:
        processor.cleanup_worker()


class ParallelTestFramework:
    def __init__(
        self,
        data_loader: BaseDataLoader,
        task_processor_class: type[BaseTaskProcessor],
        result_collector: BaseResultCollector,
        config: TestConfig | None = None,
    ) -> None:
        self.data_loader = data_loader
        self.task_processor_class = task_processor_class
        self.result_collector = result_collector
        self.config = config or TestConfig()

    def run(self, data_source: str, global_context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = global_context or {}
        raw_data = self.data_loader.load_data(data_source)
        filtered_data = self.data_loader.filter_data(raw_data)
        task_args = [self.data_loader.prepare_task_args(item, context) for item in filtered_data]

        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(_worker_wrapper, self.task_processor_class, args)
                for args in task_args
            ]
            try:
                for index, future in enumerate(as_completed(futures, timeout=self.config.timeout), start=1):
                    result = future.result()
                    self.result_collector.collect(result)
                    if self.config.progress_interval and index % self.config.progress_interval == 0:
                        print(f"finished {index}/{len(futures)} tasks")
            except TimeoutError:
                for future in futures:
                    future.cancel()
                for index, future in enumerate(futures):
                    if not future.done():
                        self.result_collector.collect(
                            TestResult(
                                task_id=f"timeout_{index}",
                                status="error",
                                data={},
                                error_message=f"Global timeout exceeded: {self.config.timeout}s",
                            )
                        )

        summary = self._generate_summary(time.perf_counter() - start, len(task_args))
        return self.result_collector.finalize(summary)

    def _generate_summary(self, total_time: float, total_tasks: int) -> dict[str, Any]:
        results = self.result_collector.results
        success_count = sum(1 for result in results if result.status == "success")
        error_count = len(results) - success_count
        times = [result.execution_time for result in results]
        return {
            "total_tasks": total_tasks,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / max(total_tasks, 1),
            "total_time": total_time,
            "avg_time_per_task": sum(times) / max(len(times), 1),
            "max_time_per_task": max(times, default=0.0),
            "min_time_per_task": min(times, default=0.0),
            "tasks_per_second": total_tasks / max(total_time, 1e-12),
            "max_workers": self.config.max_workers,
        }
