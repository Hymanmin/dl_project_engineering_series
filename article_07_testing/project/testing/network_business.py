from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from config.loader import load_config
from core.dataset.cnn_cls.cnn_cls_dataset import CnnClsDataset
from core.imports import import_modules
from core.registry import build_model
from testing.base import BaseDataLoader, BaseResultCollector, BaseTaskProcessor
from testing.config import TestResult


class NetworkCaseLoader(BaseDataLoader):
    def __init__(self, chunk_size: int = 8, max_samples: int | None = None, split: str = "test") -> None:
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.split = split

    def load_data(self, data_source: str) -> list[dict[str, Any]]:
        cfg = load_config(Path(data_source), [])
        dataset = CnnClsDataset(cfg.get("dataset", {}), split=self.split)
        total = self.max_samples or len(dataset)
        chunks: list[dict[str, Any]] = []
        for start in range(0, total, self.chunk_size):
            indices = list(range(start, min(start + self.chunk_size, total)))
            chunks.append(
                {
                    "task_id": f"{self.split}_{start:06d}_{indices[-1]:06d}",
                    "config_path": data_source,
                    "indices": indices,
                    "split": self.split,
                }
            )
        return chunks

    def prepare_task_args(self, data: dict[str, Any], global_context: dict[str, Any]) -> tuple[Any, ...]:
        return (
            data["task_id"],
            data["config_path"],
            data["indices"],
            data["split"],
            global_context,
        )


class NetworkBatchProcessor(BaseTaskProcessor):
    def process(
        self,
        task_id: str,
        config_path: str,
        indices: list[int],
        split: str,
        global_context: dict[str, Any],
    ) -> TestResult:
        torch.set_num_threads(int(global_context.get("torch_num_threads", 1)))
        cfg = load_config(Path(config_path), list(global_context.get("opts", [])))
        cfg.setdefault("runtime", {})["device"] = global_context.get("device", "cpu")
        import_modules(cfg.get("imports", []))

        device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
        dataset = CnnClsDataset(cfg.get("dataset", {}), split=split)
        model = build_model(cfg.get("model", {}), cfg).to(device).eval()

        correct = 0
        total = 0
        inference_start = time.perf_counter()
        with torch.no_grad():
            for index in indices:
                sample = dataset[index % len(dataset)]
                x = sample["input"].unsqueeze(0).to(device)
                target = sample["target"].view(1)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu()
                correct += int((pred == target).sum().item())
                total += 1
        inference_time = time.perf_counter() - inference_start

        return TestResult(
            task_id=task_id,
            status="success",
            data={
                "samples": total,
                "correct": correct,
                "accuracy": correct / max(total, 1),
                "inference_time": inference_time,
                "samples_per_second": total / max(inference_time, 1e-12),
            },
        )


class JsonResultCollector(BaseResultCollector):
    def finalize(self, summary: dict[str, Any]) -> dict[str, Any]:
        total_samples = sum(int(result.data.get("samples", 0)) for result in self.results)
        total_correct = sum(int(result.data.get("correct", 0)) for result in self.results)
        inference_time = sum(float(result.data.get("inference_time", 0.0)) for result in self.results)
        report = {
            **summary,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "accuracy": total_correct / max(total_samples, 1),
            "inference_samples_per_second": total_samples / max(inference_time, 1e-12),
            "results": [result.to_dict() for result in self.results],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report
