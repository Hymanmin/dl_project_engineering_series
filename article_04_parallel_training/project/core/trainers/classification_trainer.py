from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.dataset.cnn_cls.cnn_cls_dataset import CnnClsDataset
from core.losses.cnn_cls_loss import CnnClsLoss
from core.models.cnn_cls.cnn_cls_model import CnnClsModel
from core.registry import TRAINERS, build_loss, build_model
from core.trainers.base_trainer import BaseTrainer
from utils.distributed import build_sampler, init_distributed, is_main_process
from utils.parallel import unwrap_model, wrap_model

_ = CnnClsDataset, CnnClsLoss, CnnClsModel


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


@TRAINERS.register("classification_trainer")
class ClassificationTrainer(BaseTrainer):
    def __init__(self, cfg: dict, logger, **kwargs) -> None:
        super().__init__(cfg, logger)
        torch.manual_seed(int(cfg.get("seed", 42)))
        self.dist_state = init_distributed(cfg)
        runtime_device = cfg.get("runtime", {}).get("device", "cpu")
        if runtime_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", self.dist_state["local_rank"])
        else:
            self.device = torch.device("cpu")
        self.model = build_model(cfg.get("model", {}), cfg).to(self.device)
        self.model = wrap_model(self.model, cfg, self.device)
        self.criterion = build_loss(cfg.get("loss", {}))
        self.optimizer = _build_optimizer(self.model.parameters(), cfg)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = CnnClsDataset(self.cfg.get("dataset", {}), split=split)
        sampler = build_sampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.get("dataset", {}).get("batch_size", 8)),
            shuffle=shuffle if sampler is None else False,
            num_workers=int(self.cfg.get("dataset", {}).get("num_workers", 0)),
            sampler=sampler,
        )

    def train(self) -> None:
        train_cfg = self.cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", 1))
        print_freq = int(train_cfg.get("print_freq", 10))
        loader = self._loader("train", shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(loader, start=1):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % print_freq == 0 or step == 1:
                    self.logger.info("epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

        if is_main_process():
            ckpt_dir = Path(self.cfg.get("runtime", {}).get("checkpoint_dir", "runs/checkpoints"))
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / train_cfg.get("save_name", "model.pt")
            torch.save({"model": unwrap_model(self.model).state_dict(), "cfg": self.cfg}, ckpt_path)
            self.logger.info("Saved checkpoint to %s", ckpt_path)

    def evaluate(self) -> dict[str, float]:
        loader = self._loader("val", shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch["input"].to(self.device))
                pred = logits.argmax(dim=1).cpu()
                target = batch["target"]
                correct += int((pred == target).sum().item())
                total += int(target.numel())
        acc = correct / max(total, 1)
        self.logger.info("eval_acc=%.4f", acc)
        return {"acc": acc}
