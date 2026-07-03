from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing PyYAML. Run: python -m pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[2]

def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env or os.environ.copy(),
    )
    result = {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if check and completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result

def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def ensure_dirs(plan: dict) -> list[str]:
    created = []
    for item in plan.get("directories", []):
        path = ROOT / item
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    return created

def set_local_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AI_MODELS_ROOT", str(ROOT / "models"))
    env.setdefault("HF_HOME", str(ROOT / "models/hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(ROOT / "models/hf_cache/hub"))
    env.setdefault("HF_ASSETS_CACHE", str(ROOT / "models/hf_cache/assets"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("AI_MODELS_CONFIG", str(ROOT / "configs/ai_models.pro.yaml"))
    env.setdefault("LOCAL_MODEL_RUNTIME_CONFIG", str(ROOT / "configs/local_model_runtime.pro.yaml"))
    env.setdefault("MODEL_REGISTRY_CONFIG", str(ROOT / "configs/model_registry.yaml"))
    env.setdefault("CUSTOM_MODEL_ROOT", str(ROOT / "models/checkpoints/custom"))
    env.setdefault("AI_DEVICE", "auto")
    return env

def clone_text2midi(plan: dict, report: dict, env: dict[str, str]) -> None:
    text2 = plan["models"]["text2midi"]
    repo_dir = ROOT / text2["repo_dir"]
    if repo_dir.exists() and (repo_dir / "model/transformer_model.py").exists():
        report["steps"].append({"text2midi_repo": "already_present", "path": str(repo_dir)})
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    report["steps"].append(run(["git", "clone", text2["repo_url"], str(repo_dir)], env=env))

def install_requirements(plan: dict, report: dict, env: dict[str, str], *, skip_pip: bool) -> None:
    if skip_pip:
        report["steps"].append({"pip": "skipped"})
        return
    for group, files in plan.get("pip", {}).items():
        for file_name in files:
            path = ROOT / file_name
            if not path.exists():
                report["steps"].append({"pip": "missing", "file": file_name})
                continue
            report["steps"].append(run([sys.executable, "-m", "pip", "install", "-r", str(path)], env=env))

def download_hf_text2midi(plan: dict, report: dict, env: dict[str, str]) -> None:
    report["steps"].append(run([
        sys.executable,
        "scripts/models/download_text2midi.py",
        "--repo-id",
        plan["models"]["text2midi"]["hf_repo_id"],
        "--checkpoint-dir",
        plan["models"]["text2midi"]["repo_dir"].replace("external_repos/text2midi", "checkpoints/text2midi"),
    ], env=env))

def maybe_pull_ollama(plan: dict, report: dict) -> None:
    ollama = shutil.which("ollama")
    model = plan["models"]["ollama_planner"]["model_name"]
    if not ollama:
        report["warnings"].append("Ollama CLI not found. Install Ollama, then run: ollama pull " + model)
        return
    report["steps"].append(run(["ollama", "--version"], check=False))
    report["steps"].append(run(["ollama", "pull", model], check=False))

def smoke_all(report: dict, env: dict[str, str], *, allow_text2midi_check_only: bool) -> None:
    smoke_cmds = [
        [sys.executable, "scripts/models/smoke_midigpt.py"],
        [sys.executable, "scripts/models/smoke_miditok.py"],
        [sys.executable, "scripts/models/smoke_ollama_planner.py"],
    ]
    text2_cmd = [sys.executable, "scripts/models/smoke_text2midi.py"]
    if allow_text2midi_check_only:
        text2_cmd.append("--allow-check-only")
    smoke_cmds.append(text2_cmd)
    for cmd in smoke_cmds:
        report["steps"].append(run(cmd, check=False, env=env))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/model_install_plan.yaml")
    parser.add_argument("--profile", default="pro")
    parser.add_argument("--planner-model", default="qwen3:8b")
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-text2midi-check-only", action="store_true")
    args = parser.parse_args()

    plan_path = ROOT / args.plan
    plan = read_yaml(plan_path)
    env = set_local_env()

    report = {
        "schema_version": "0.2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "plan": str(plan_path),
        "status": "running",
        "steps": [],
        "warnings": [],
    }

    try:
        report["created_dirs"] = ensure_dirs(plan)
        install_requirements(plan, report, env, skip_pip=args.skip_pip)
        report["steps"].append(run([sys.executable, "scripts/models/download_midigpt.py"], env=env))
        clone_text2midi(plan, report, env)
        download_hf_text2midi(plan, report, env)
        maybe_pull_ollama(plan, report)
        if not args.skip_smoke:
            smoke_all(report, env, allow_text2midi_check_only=args.allow_text2midi_check_only)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)

    out = ROOT / "models/manifests/install_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
