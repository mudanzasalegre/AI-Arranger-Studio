from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def model_runtime_status() -> list[dict[str, object]]:
    return [
        {
            "id": "midigpt",
            "enabled": os.environ.get("AI_ENABLE_MIDIGPT", "false").lower() in {"1", "true", "yes"},
            "package_available": package_available("midigpt"),
            "model_name": os.environ.get("MIDIGPT_MODEL_NAME", "yellow"),
        },
        {
            "id": "text2midi",
            "enabled": os.environ.get("AI_ENABLE_TEXT2MIDI", "false").lower()
            in {"1", "true", "yes"},
            "repo_dir_exists": Path(
                os.environ.get("TEXT2MIDI_REPO_DIR", "models/external_repos/text2midi")
            ).exists(),
            "checkpoint_dir_exists": Path(
                os.environ.get("TEXT2MIDI_CHECKPOINT_DIR", "models/checkpoints/text2midi")
            ).exists(),
        },
        {
            "id": "local_llm_planner",
            "enabled": os.environ.get("AI_ENABLE_LOCAL_LLM_PLANNER", "false").lower()
            in {"1", "true", "yes"},
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api"),
            "model": os.environ.get("OLLAMA_PLANNER_MODEL", "qwen3:8b"),
        },
    ]
