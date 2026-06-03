from __future__ import annotations

import unittest
from pathlib import Path

from config.loader import load_config
from core.imports import import_modules
from launcher.benchmark import run_benchmark
from launcher.export import run_export
from utils.logger import setup_logger


class ExportBenchmarkTest(unittest.TestCase):
    def test_export_and_benchmark_smoke(self) -> None:
        cfg = load_config(Path("config/cnn_cls_export_torchscript.yaml"), [])
        cfg["export"]["output_path"] = "exports/test_model.pt"
        cfg["runtime"]["log_dir"] = "runs/test_logs"
        import_modules(cfg.get("imports", []))
        logger = setup_logger(cfg["runtime"]["log_dir"])
        run_export(cfg, logger)
        self.assertTrue(Path("exports/test_model.pt").exists())

        bench_cfg = load_config(Path("config/cnn_cls_benchmark.yaml"), ["benchmark.iters=2", "benchmark.warmup=1"])
        bench_cfg["benchmark"]["output_path"] = "runs/test_benchmark.json"
        import_modules(bench_cfg.get("imports", []))
        run_benchmark(bench_cfg, logger)
        self.assertTrue(Path("runs/test_benchmark.json").exists())


if __name__ == "__main__":
    unittest.main()
