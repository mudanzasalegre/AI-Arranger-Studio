from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENV_LINES = {
    "AI_MODELS_CONFIG": "./configs/ai_models.pro.yaml",
    "LOCAL_MODEL_RUNTIME_CONFIG": "./configs/local_model_runtime.pro.yaml",
    "MODEL_REGISTRY_CONFIG": "./configs/model_registry.yaml",
    "AI_MODELS_ROOT": "./models",
    "HF_HOME": "./models/hf_cache",
    "HF_HUB_CACHE": "./models/hf_cache/hub",
    "HF_ASSETS_CACHE": "./models/hf_cache/assets",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "AI_ENABLE_MOCK_SYMBOLIC": "false",
    "AI_ENABLE_MIDIGPT": "true",
    "AI_ENABLE_TEXT2MIDI": "true",
    "AI_ENABLE_LOCAL_LLM_PLANNER": "true",
    "AI_ENABLE_CUSTOM_ROLE_MODELS": "true",
    "CUSTOM_MODEL_ROOT": "./models/checkpoints/custom",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434/api",
    "OLLAMA_PLANNER_MODEL": "qwen3:8b",
}

def read_env(path: Path) -> dict[str, str]:
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        data[key.strip()] = value.strip()
    return data

def write_env(path: Path, data: dict[str, str]) -> None:
    lines = ["# Generated/updated by scripts/models_pro/activate_pro_profile.py"]
    for key in sorted(data):
        lines.append(f"{key}={data[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-env", action="store_true")
    parser.add_argument("--copy-local-configs", action="store_true")
    args = parser.parse_args()

    report = {"status": "ok", "actions": []}

    if args.copy_local_configs:
        shutil.copyfile(ROOT / "configs/ai_models.pro.yaml", ROOT / "configs/ai_models.local.yaml")
        shutil.copyfile(ROOT / "configs/local_model_runtime.pro.yaml", ROOT / "configs/local_model_runtime.yaml")
        report["actions"].append("copied_pro_configs_to_local")

    if args.write_env:
        env_path = ROOT / ".env"
        env = read_env(env_path)
        env.update(ENV_LINES)
        write_env(env_path, env)
        report["actions"].append("updated_env")

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
