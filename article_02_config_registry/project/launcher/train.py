from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.dataset.cnn_cls.cnn_cls_dataset import CnnClsDataset
from core.losses.cnn_cls_loss import CnnClsLoss
from core.models.cnn_cls.cnn_cls_model import CnnClsModel
from core.registry import build_loss, build_model

_ = CnnClsLoss


def _build_optimizer(parameters, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_train(cfg: dict, logger) -> None:
    torch.manual_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))

    dataset = CnnClsDataset(cfg.get("dataset", {}), split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("dataset", {}).get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("dataset", {}).get("num_workers", 0)),
    )

    model: CnnClsModel = build_model(cfg.get("model", {}), cfg).to(device)
    criterion = build_loss(cfg.get("loss", {}))
    optimizer = _build_optimizer(model.parameters(), cfg)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    print_freq = int(train_cfg.get("print_freq", 10))

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader, start=1):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % print_freq == 0 or step == 1:
                logger.info("epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

    ckpt_dir = Path(cfg.get("runtime", {}).get("checkpoint_dir", "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / train_cfg.get("save_name", "model.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    logger.info("Saved checkpoint to %s", ckpt_path)


