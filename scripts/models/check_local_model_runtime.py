from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PyYAML. Run `python -m pip install -r requirements.txt`.") from exc

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENV_KEYS = [
    "AI_MODELS_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "AI_MODELS_CONFIG",
    "CUSTOM_MODEL_ROOT",
]
OPTIONAL_CONFIG_ENV_KEYS = [
    "LOCAL_MODEL_RUNTIME_CONFIG",
    "MODEL_REGISTRY_CONFIG",
]


def _exists(path: str | Path) -> bool:
    return Path(path).expanduser().exists()


def _writable(path: str | Path) -> bool:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    probe = p / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _spec(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_dotenv_defaults(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_model_runtime.example.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dotenv_loaded = load_dotenv_defaults(ROOT / ".env")
    config_path = (
        (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    )
    config = load_yaml(config_path)
    runtime = config.get("runtime", {})
    hf = config.get("huggingface", {})

    models_root = ROOT / runtime.get("models_root", "./models")
    artifact_root = ROOT / runtime.get("artifact_root", "./outputs/model_artifacts")
    custom_model_root = ROOT / config.get("training", {}).get(
        "custom_model_root", "./models/checkpoints/custom"
    )
    hf_home = ROOT / hf.get("hf_home", "./models/hf_cache")
    hf_hub_cache = ROOT / hf.get("hf_hub_cache", "./models/hf_cache/hub")

    env_values = {
        key: os.environ.get(key)
        for key in [*REQUIRED_ENV_KEYS, *OPTIONAL_CONFIG_ENV_KEYS, "AI_DEVICE"]
    }
    configured_paths = {
        key: str(path)
        for key in [*REQUIRED_ENV_KEYS, *OPTIONAL_CONFIG_ENV_KEYS]
        if (path := _resolve_repo_path(os.environ.get(key))) is not None
    }
    path_checks = {
        "ai_models_config": {
            "path": str(_resolve_repo_path(os.environ.get("AI_MODELS_CONFIG")) or ""),
            "exists": bool(
                (ai_models_config := _resolve_repo_path(os.environ.get("AI_MODELS_CONFIG")))
                and ai_models_config.exists()
            ),
        },
        "model_registry_config": {
            "path": str(_resolve_repo_path(os.environ.get("MODEL_REGISTRY_CONFIG")) or ""),
            "exists": bool(
                (
                    model_registry_config := _resolve_repo_path(
                        os.environ.get("MODEL_REGISTRY_CONFIG")
                    )
                )
                and model_registry_config.exists()
            ),
        },
    }

    checks = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(ROOT),
        "config_path": str(config_path),
        "dotenv_loaded": sorted(dotenv_loaded),
        "directories": {
            "models_root": {"path": str(models_root), "writable": _writable(models_root)},
            "custom_model_root": {
                "path": str(custom_model_root),
                "writable": _writable(custom_model_root),
            },
            "artifact_root": {"path": str(artifact_root), "writable": _writable(artifact_root)},
            "hf_home": {"path": str(hf_home), "writable": _writable(hf_home)},
            "hf_hub_cache": {"path": str(hf_hub_cache), "writable": _writable(hf_hub_cache)},
        },
        "env": env_values,
        "configured_paths": configured_paths,
        "config_files": path_checks,
        "optional_packages": {
            "midigpt": _spec("midigpt"),
            "torch": _spec("torch"),
            "transformers": _spec("transformers"),
            "huggingface_hub": _spec("huggingface_hub"),
            "miditok": _spec("miditok"),
            "httpx": _spec("httpx"),
        },
    }

    errors = []
    for key, item in checks["directories"].items():
        if not item["writable"]:
            errors.append(f"Directory not writable: {key} -> {item['path']}")
    for key in REQUIRED_ENV_KEYS:
        if not env_values.get(key):
            errors.append(f"Missing required environment key: {key}")
    if not path_checks["ai_models_config"]["exists"]:
        errors.append(f"AI models config not found: {path_checks['ai_models_config']['path']}")
    if (
        env_values.get("MODEL_REGISTRY_CONFIG")
        and not path_checks["model_registry_config"]["exists"]
    ):
        errors.append(
            f"Model registry config not found: {path_checks['model_registry_config']['path']}"
        )

    checks["status"] = "fail" if errors else "ok"
    checks["errors"] = errors

    out_path = ROOT / "outputs/model_smoke/local_model_runtime_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(checks, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
