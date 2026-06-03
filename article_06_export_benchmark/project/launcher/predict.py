from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.dataset.cnn_cls.cnn_cls_dataset import CnnClsDataset
from core.models.cnn_cls.cnn_cls_model import CnnClsModel
from core.registry import build_model

_ = CnnClsModel


def run_predict(cfg: dict, logger) -> None:
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    dataset = CnnClsDataset(cfg.get("dataset", {}), split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_model(cfg.get("model", {}), cfg).to(device)
    ckpt_path = Path(cfg.get("predict", {}).get("checkpoint", ""))
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning("Checkpoint not found, using randomly initialized model: %s", ckpt_path)

    model.eval()
    results: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(device))
            pred = logits.argmax(dim=1).cpu().item()
            results.append(str(pred))

    output_path = Path(cfg.get("predict", {}).get("output_path", "runs/predict_results.txt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(results), encoding="utf-8")
    logger.info("Saved predictions to %s", output_path)


