from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(cfg: dict) -> dict[str, int]:
    parallel_cfg = cfg.get("runtime", {}).get("parallel", {})
    if parallel_cfg.get("type") != "ddp":
        return {"rank": 0, "local_rank": 0, "world_size": 1}

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = parallel_cfg.get("backend", "gloo")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def is_main_process() -> bool:
    return not is_dist_available_and_initialized() or dist.get_rank() == 0


def build_sampler(dataset, shuffle: bool):
    if is_dist_available_and_initialized():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()
