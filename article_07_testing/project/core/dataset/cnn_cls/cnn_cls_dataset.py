from __future__ import annotations

from pathlib import Path

import torch

from core.dataset.base_dataset import BaseDataset
from core.registry import DATASETS


@DATASETS.register("cnn_cls_dataset")
class CnnClsDataset(BaseDataset):
    def __init__(self, cfg: dict, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.input_dim = int(cfg.get("input_dim", 16))
        self.num_classes = int(cfg.get("num_classes", 2))
        self.samples = self._load_samples(cfg.get(f"{split}_list"))

    def _load_samples(self, list_path: str | None) -> list[tuple[str, int]]:
        if not list_path or not Path(list_path).exists():
            return [("synthetic", i % self.num_classes) for i in range(32)]

        samples: list[tuple[str, int]] = []
        for line in Path(list_path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            label = int(parts[1]) if len(parts) > 1 else 0
            samples.append((parts[0], label))
        return samples or [("synthetic", 0)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _, label = self.samples[index]
        generator = torch.Generator().manual_seed(index)
        return {
            "input": torch.randn(self.input_dim, generator=generator),
            "target": torch.tensor(label, dtype=torch.long),
        }


