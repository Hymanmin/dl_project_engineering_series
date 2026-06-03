from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_value(raw: str) -> Any:
    if yaml is not None:
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            return raw

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Only 2-space YAML indentation is supported, line {line_no}")

        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line {line_no}: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_value(value.strip())

    return root


def _set_by_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = cfg
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
        if not isinstance(cursor, dict):
            raise TypeError(f"Cannot override non-dict config node: {dotted_key}")
    cursor[keys[-1]] = value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    data = yaml.safe_load(text) if yaml is not None else _minimal_yaml_load(text)
    data = data or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = path.resolve()
    cfg = _load_yaml(path)

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = (path.parent / base_name).resolve()
        cfg = _deep_merge(load_config(base_path, []), cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_by_path(cfg, key, _parse_value(raw_value))

    return cfg


