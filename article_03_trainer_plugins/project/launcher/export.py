from __future__ import annotations

from pathlib import Path


def run_export(cfg: dict, logger) -> None:
    output_path = Path(cfg.get("export", {}).get("output_path", "exports/model.txt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("export placeholder\n", encoding="utf-8")
    logger.info("Export placeholder written to %s", output_path)
