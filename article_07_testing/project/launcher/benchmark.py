from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from core.registry import build_model


def run_benchmark(cfg: dict, logger) -> None:
    bench_cfg = cfg.get("benchmark", {})
    warmup = int(bench_cfg.get("warmup", 5))
    iters = int(bench_cfg.get("iters", 50))
    output_path = Path(bench_cfg.get("output_path", "runs/benchmark.json"))
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    input_dim = int(cfg.get("dataset", {}).get("input_dim", 16))

    model = build_model(cfg.get("model", {}), cfg).to(device).eval()
    x = torch.randn(1, input_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    result = {
        "device": str(device),
        "iters": iters,
        "latency_ms": elapsed / max(iters, 1) * 1000,
        "throughput_samples_per_s": iters / max(elapsed, 1e-12),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("benchmark=%s", result)
