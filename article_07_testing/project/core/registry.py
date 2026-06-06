from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError("Registry key cannot be empty")
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._items[key] = obj  # type: ignore[assignment]
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"{name} is not registered in {self.name}. Available: {available}")
        return self._items[name]

    def build(self, cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if "name" not in cfg:
            raise KeyError(f"Missing 'name' in {self.name} config")
        target = self.get(str(cfg["name"]))
        params = {k: v for k, v in cfg.items() if k != "name"}
        return target(*args, **params, **kwargs)


DATASETS = Registry("dataset")
NETWORKS = Registry("network")
MODELS = Registry("model")


def build_dataset(cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    return DATASETS.build(cfg, *args, **kwargs)


def build_network(cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    return NETWORKS.build(cfg, *args, **kwargs)


def build_model(cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    return MODELS.build(cfg, *args, **kwargs)


