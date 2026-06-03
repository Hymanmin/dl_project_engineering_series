from __future__ import annotations

import importlib


def import_modules(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)
