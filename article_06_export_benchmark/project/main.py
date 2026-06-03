from __future__ import annotations

import argparse
from pathlib import Path

from config.loader import load_config
from core.imports import import_modules
from launcher.benchmark import run_benchmark
from launcher.eval import run_eval
from launcher.export import run_export
from launcher.predict import run_predict
from launcher.train import run_train
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular deep learning project template")
    parser.add_argument("--mode", choices=["train", "eval", "predict", "export", "benchmark"], required=True, help="Run mode")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Override config options, for example: train.epochs=20 model.num_classes=2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), args.opts)
    import_modules(cfg.get("imports", []))
    logger = setup_logger(cfg.get("runtime", {}).get("log_dir", "runs/logs"))
    logger.info("Start %s mode with task=%s", args.mode, cfg.get("task", "unknown"))

    if args.mode == "train":
        run_train(cfg, logger)
    elif args.mode == "eval":
        run_eval(cfg, logger)
    elif args.mode == "predict":
        run_predict(cfg, logger)
    elif args.mode == "export":
        run_export(cfg, logger)
    else:
        run_benchmark(cfg, logger)


if __name__ == "__main__":
    main()


