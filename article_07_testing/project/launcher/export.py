from __future__ import annotations

from pathlib import Path

import torch

from core.registry import build_model


def _load_checkpoint_if_exists(model, checkpoint: str, device: torch.device, logger) -> None:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found, export randomly initialized model: %s", ckpt_path)
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    logger.info("Loaded checkpoint from %s", ckpt_path)


def run_export(cfg: dict, logger) -> None:
    export_cfg = cfg.get("export", {})
    fmt = export_cfg.get("format", "torchscript")
    output_path = Path(export_cfg.get("output_path", "exports/cnn_cls.pt"))
    checkpoint = export_cfg.get("checkpoint", "runs/checkpoints/cnn_cls.pt")
    input_dim = int(cfg.get("dataset", {}).get("input_dim", 16))
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))

    model = build_model(cfg.get("model", {}), cfg).to(device).eval()
    _load_checkpoint_if_exists(model, checkpoint, device, logger)
    dummy = torch.randn(1, input_dim, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "torchscript":
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_path))
    elif fmt == "onnx":
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=int(export_cfg.get("opset_version", 17)),
        )
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    logger.info("Exported %s model to %s", fmt, output_path)
