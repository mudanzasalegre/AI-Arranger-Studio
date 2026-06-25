from pathlib import Path

import yaml


def test_all_yaml_configs_are_parseable():
    root = Path(__file__).resolve().parents[1]
    config_paths = sorted((root / "configs").rglob("*.yaml"))

    assert config_paths

    for config_path in config_paths:
        with config_path.open(encoding="utf-8") as file:
            assert yaml.safe_load(file) is not None, f"Empty config: {config_path}"
