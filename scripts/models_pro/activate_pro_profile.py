from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "models" / "manifests" / "pro_profile_activation.json"

DEFAULT_PRO_ENV = {
    "AI_MODELS_CONFIG": "./configs/ai_models.pro.yaml",
    "LOCAL_MODEL_RUNTIME_CONFIG": "./configs/local_model_runtime.pro.yaml",
    "MODEL_REGISTRY_CONFIG": "./configs/model_registry.yaml",
    "AI_MODELS_ROOT": "./models",
    "HF_HOME": "./models/hf_cache",
    "HF_HUB_CACHE": "./models/hf_cache/hub",
    "HF_ASSETS_CACHE": "./models/hf_cache/assets",
    "HF_HUB_OFFLINE": "0",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "AI_DEVICE": "auto",
    "AI_MODEL_WORKER_ENABLED": "true",
    "AI_MODEL_WORKER_URL": "http://127.0.0.1:8010",
    "AI_ENABLE_MOCK_SYMBOLIC": "false",
    "AI_ENABLE_MIDIGPT": "true",
    "AI_ENABLE_TEXT2MIDI": "true",
    "AI_ENABLE_LOCAL_LLM_PLANNER": "true",
    "AI_ENABLE_CUSTOM_ROLE_MODELS": "true",
    "MIDIGPT_MODEL_NAME": "yellow",
    "MIDIGPT_CACHE": "./models/hf_cache/hub",
    "TEXT2MIDI_REPO_DIR": "./models/external_repos/text2midi",
    "TEXT2MIDI_CHECKPOINT_DIR": "./models/checkpoints/text2midi",
    "TEXT2MIDI_MODEL_FILE": "./models/checkpoints/text2midi/pytorch_model.bin",
    "TEXT2MIDI_TOKENIZER_FILE": "./models/checkpoints/text2midi/vocab_remi.pkl",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434/api",
    "OLLAMA_PLANNER_MODEL": "qwen3:8b",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS": "120",
    "TOKENIZED_DATA_ROOT": "./data/processed/tokenized",
    "CUSTOM_MODEL_ROOT": "./models/checkpoints/custom",
}


@dataclass(frozen=True)
class ActivationOptions:
    env_file: Path
    planner_model: str
    dry_run: bool = False
    copy_local_configs: bool = False
    force: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate the local pro model profile.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--planner-model", default="qwen3:8b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy-local-configs", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --copy-local-configs to overwrite existing local config files.",
    )
    args = parser.parse_args()

    options = ActivationOptions(
        env_file=_repo_path(args.env_file),
        planner_model=args.planner_model,
        dry_run=args.dry_run,
        copy_local_configs=args.copy_local_configs,
        force=args.force,
    )
    report = activate_pro_profile(options)
    print(json.dumps(report, indent=2))


def activate_pro_profile(options: ActivationOptions) -> dict[str, Any]:
    env_updates = dict(DEFAULT_PRO_ENV)
    env_updates["OLLAMA_PLANNER_MODEL"] = options.planner_model

    before = options.env_file.read_text(encoding="utf-8") if options.env_file.exists() else ""
    after, changed, appended = update_env_content(before, env_updates)
    copied_configs = copy_local_configs(force=options.force) if options.copy_local_configs else {}

    status = "dry_run" if options.dry_run else "ok"
    report = {
        "schema_version": "0.1.0",
        "activation": "pr29_pro_profile",
        "status": status,
        "generated_at": _now(),
        "env_file": str(options.env_file),
        "planner_model": options.planner_model,
        "changed_keys": changed,
        "appended_keys": appended,
        "copy_local_configs": options.copy_local_configs,
        "copied_configs": copied_configs,
    }
    if not options.dry_run:
        options.env_file.parent.mkdir(parents=True, exist_ok=True)
        options.env_file.write_text(after, encoding="utf-8")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def update_env_content(
    content: str,
    updates: dict[str, str],
) -> tuple[str, list[str], list[str]]:
    lines = content.splitlines()
    changed: list[str] = []
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key, _value = line.split("=", maxsplit=1)
        key = key.strip()
        if key in updates:
            new_line = f"{key}={updates[key]}"
            output.append(new_line)
            seen.add(key)
            if new_line != line:
                changed.append(key)
        else:
            output.append(line)

    appended = [key for key in updates if key not in seen]
    if appended and output and output[-1].strip():
        output.append("")
    for key in appended:
        output.append(f"{key}={updates[key]}")

    return "\n".join(output).rstrip() + "\n", changed, appended


def copy_local_configs(*, force: bool) -> dict[str, Any]:
    copy_pairs = {
        "ai_models": (
            ROOT / "configs" / "ai_models.pro.yaml",
            ROOT / "configs" / "ai_models.local.yaml",
        ),
        "local_model_runtime": (
            ROOT / "configs" / "local_model_runtime.pro.yaml",
            ROOT / "configs" / "local_model_runtime.yaml",
        ),
    }
    copied: dict[str, Any] = {}
    for key, (source, target) in copy_pairs.items():
        if not source.exists():
            copied[key] = {"status": "fail", "error": f"Source not found: {source}"}
            continue
        if target.exists() and not force:
            copied[key] = {
                "status": "skipped",
                "source": str(source),
                "target": str(target),
                "reason": "target_exists",
            }
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[key] = {"status": "copied", "source": str(source), "target": str(target)}
    return copied


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
