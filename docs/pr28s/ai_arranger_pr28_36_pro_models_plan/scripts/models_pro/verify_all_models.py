from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS = [
    ("runtime", [sys.executable, "scripts/models/check_local_model_runtime.py", "--config", "configs/local_model_runtime.pro.yaml"]),
    ("midigpt", [sys.executable, "scripts/models/smoke_midigpt.py"]),
    ("text2midi", [sys.executable, "scripts/models/smoke_text2midi.py"]),
    ("ollama_planner", [sys.executable, "scripts/models/smoke_ollama_planner.py"]),
    ("miditok", [sys.executable, "scripts/models/smoke_miditok.py"]),
    ("custom_role_models", [sys.executable, "scripts/models/smoke_custom_role_models.py"]),
]

def run(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
        "ok": completed.returncode == 0,
    }

def main() -> None:
    results = {name: run(cmd) for name, cmd in CHECKS}
    required = ["runtime", "midigpt", "text2midi", "ollama_planner", "miditok"]
    status = "ok" if all(results[name]["ok"] for name in required) else "fail"
    report = {
        "status": status,
        "packages": {
            "midigpt": importlib.util.find_spec("midigpt") is not None,
            "miditok": importlib.util.find_spec("miditok") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
        },
        "env": {
            "AI_MODELS_CONFIG": os.environ.get("AI_MODELS_CONFIG"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
        "results": results,
    }
    out = ROOT / "outputs/model_smoke/verify_all_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "ok":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
