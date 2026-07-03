from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMMANDS = {
    "ruff": [sys.executable, "-m", "ruff", "check", "apps", "packages", "scripts", "tests"],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "package_smoke": [sys.executable, "scripts/package_smoke.py"],
    "golden_generate": [sys.executable, "scripts/golden_generate.py"],
    "ai_contract_smoke": [sys.executable, "scripts/ai_contract_smoke.py"],
}

REQUIRED_FILES = [
    "configs/ai_models.local.example.yaml",
    "configs/local_model_runtime.example.yaml",
    "configs/model_registry.example.yaml",
    "requirements-ai.txt",
    "requirements-training-ai.txt",
    "scripts/models/smoke_midigpt.py",
    "scripts/models/run_text2midi_inference.py",
    "scripts/models/smoke_miditok.py",
    "packages/model_backends/model_backends/planner/ollama_planner_backend.py",
    "packages/training/training/tokenizers/miditok_real.py",
]

def run(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout_tail": completed.stdout[-2500:],
        "stderr_tail": completed.stderr[-2500:],
    }

def main() -> None:
    files = {path: (ROOT / path).exists() for path in REQUIRED_FILES}
    commands = {name: run(cmd) for name, cmd in COMMANDS.items()}
    status = "ok" if all(files.values()) and all(result["ok"] for result in commands.values()) else "fail"
    report = {"status": status, "files": files, "commands": commands}
    out_json = ROOT / "outputs/pro_audit/pr28_repo_health.json"
    out_md = ROOT / "outputs/pro_audit/pr28_repo_health.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "ok":
        raise SystemExit(1)

def _markdown(report: dict) -> str:
    lines = ["# PR-28 Repo Health", "", f"Status: **{report['status']}**", "", "## Files"]
    for path, ok in report["files"].items():
        lines.append(f"- {'✅' if ok else '❌'} `{path}`")
    lines.extend(["", "## Commands"])
    for name, result in report["commands"].items():
        lines.append(f"- {'✅' if result['ok'] else '❌'} `{name}` returncode={result['returncode']}")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    main()
