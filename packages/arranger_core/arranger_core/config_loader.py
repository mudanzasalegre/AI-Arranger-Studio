from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    search_start = (start or Path(__file__)).resolve()
    candidates = [search_start, *search_start.parents]

    for candidate in candidates:
        if (candidate / "configs").is_dir() and (candidate / "docs").is_dir():
            return candidate

    fallback = Path.cwd().resolve()
    if (fallback / "configs").is_dir():
        return fallback

    raise FileNotFoundError("Could not locate repository root containing configs/")


class MusicConfigLoader:
    def __init__(self, config_root: str | Path | None = None) -> None:
        self.config_root = Path(config_root) if config_root else find_repo_root() / "configs"
        self.config_root = self.config_root.resolve()

    def load_yaml(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.config_root / relative_path
        return load_yaml_file(path)

    def load_yaml_files(self, pattern: str) -> dict[str, dict[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        for path in sorted(self.config_root.glob(pattern)):
            if path.is_file():
                loaded[path.stem] = load_yaml_file(path)
        return loaded


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {yaml_path}")

    return data
