from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from testing.config import TestConfig
from testing.framework import ParallelTestFramework
from testing.network_business import JsonResultCollector, NetworkBatchProcessor, NetworkCaseLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale network business tests in parallel.")
    parser.add_argument("--config", default="config/cnn_cls_predict.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/business_tests")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--opts", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = max((os.cpu_count() or 2) // 2, 1)

    framework = ParallelTestFramework(
        data_loader=NetworkCaseLoader(
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            split=args.split,
        ),
        task_processor_class=NetworkBatchProcessor,
        result_collector=JsonResultCollector(args.output_dir),
        config=TestConfig(
            max_workers=max_workers,
            timeout=args.timeout,
            output_dir=args.output_dir,
        ),
    )
    report = framework.run(
        args.config,
        global_context={
            "device": args.device,
            "torch_num_threads": args.torch_num_threads,
            "opts": args.opts,
        },
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
