from __future__ import annotations

import time

import torch

from core.registry import build_model


def run_benchmark(cfg: dict, logger) -> None:
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    model = build_model(cfg.get("model", {}), cfg).to(device).eval()
    input_dim = int(cfg.get("dataset", {}).get("input_dim", 16))
    x = torch.randn(1, input_dim, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(x)
    elapsed = time.perf_counter() - start
    logger.info("benchmark_latency_ms=%.4f", elapsed / 20 * 1000)
