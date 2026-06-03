from __future__ import annotations

import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils.distributed import is_dist_available_and_initialized


def wrap_model(model: nn.Module, cfg: dict, device: torch.device) -> nn.Module:
    parallel_cfg = cfg.get("runtime", {}).get("parallel", {})
    parallel_type = parallel_cfg.get("type", "none")

    if parallel_type == "dp" and torch.cuda.device_count() > 1:
        return DataParallel(model)

    if parallel_type == "ddp" and is_dist_available_and_initialized():
        if device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[device.index])
        return DistributedDataParallel(model)

    if parallel_type == "pipeline":
        raise NotImplementedError("Pipeline parallel should be integrated through a dedicated library.")

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
