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

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path)
    runtime = config.get("runtime", {})
    hf = config.get("huggingface", {})

    models_root = ROOT / runtime.get("models_root", "./models")
    artifact_root = ROOT / runtime.get("artifact_root", "./outputs/model_artifacts")
    hf_home = ROOT / hf.get("hf_home", "./models/hf_cache")
    hf_hub_cache = ROOT / hf.get("hf_hub_cache", "./models/hf_cache/hub")

    checks = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(ROOT),
        "config_path": str(config_path),
        "directories": {
            "models_root": {"path": str(models_root), "writable": _writable(models_root)},
            "artifact_root": {"path": str(artifact_root), "writable": _writable(artifact_root)},
            "hf_home": {"path": str(hf_home), "writable": _writable(hf_home)},
            "hf_hub_cache": {"path": str(hf_hub_cache), "writable": _writable(hf_hub_cache)},
        },
        "env": {
            "AI_MODELS_ROOT": os.environ.get("AI_MODELS_ROOT"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "AI_MODELS_CONFIG": os.environ.get("AI_MODELS_CONFIG"),
        },
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
