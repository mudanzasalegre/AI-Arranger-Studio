from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["midigpt", "torch", "transformers", "huggingface_hub", "miditok", "httpx", "yaml"]


def main() -> None:
    report = {
        "status": "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": {
            "AI_MODELS_ROOT": os.environ.get("AI_MODELS_ROOT"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "AI_MODELS_CONFIG": os.environ.get("AI_MODELS_CONFIG"),
        },
        "packages": {name: importlib.util.find_spec(name) is not None for name in PACKAGES},
    }
    out = ROOT / "models/manifests/install_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
