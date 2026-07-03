from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - reported by dependency checks too.
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "pro_audit"
NPM_COMMAND = "npm.cmd" if os.name == "nt" and shutil.which("npm.cmd") else "npm"
NODE_COMMAND = "node.exe" if os.name == "nt" and shutil.which("node.exe") else "node"
GIT_COMMAND = "git.exe" if os.name == "nt" and shutil.which("git.exe") else "git"
MAKE_COMMAND = "make.exe" if os.name == "nt" and shutil.which("make.exe") else "make"
OLLAMA_COMMAND = "ollama.exe" if os.name == "nt" and shutil.which("ollama.exe") else "ollama"

REQUIRED_ENV_EXAMPLE_KEYS = (
    "AI_MODELS_CONFIG",
    "LOCAL_MODEL_RUNTIME_CONFIG",
    "MODEL_REGISTRY_CONFIG",
    "AI_MODELS_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "AI_ENABLE_MIDIGPT",
    "AI_ENABLE_TEXT2MIDI",
    "AI_ENABLE_LOCAL_LLM_PLANNER",
    "AI_ENABLE_CUSTOM_ROLE_MODELS",
)

REQUIRED_GITIGNORE_PATTERNS = (
    "models/",
    "outputs/",
    "data/raw/",
    "data/private/",
    "data/processed/",
    "*.mid",
    "*.musicxml",
    "*.pt",
    "*.bin",
    "*.safetensors",
)

REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-ai.txt",
    "requirements-training-ai.txt",
    ".env.example",
    ".gitignore",
    "configs/ai_models.yaml",
    "configs/ai_models.pro.yaml",
    "configs/ai_models.local.example.yaml",
    "configs/generation_profiles.pro.yaml",
    "configs/local_model_runtime.example.yaml",
    "configs/local_model_runtime.pro.yaml",
    "configs/model_registry.example.yaml",
    "configs/professional_benchmark_gate.pro.yaml",
    "configs/quality_thresholds.pro.yaml",
    "scripts/package_smoke.py",
    "scripts/golden_generate.py",
    "scripts/ai_contract_smoke.py",
    "scripts/models/check_local_model_runtime.py",
    "scripts/models/ai_local_smoke.py",
    "scripts/models/professional_generation_benchmark.py",
    "scripts/models_pro/generate_professional_midi.py",
    "scripts/models_pro/pro_quality_gate.py",
    "scripts/models_pro/professional_benchmark_gate.py",
    "apps/api/app/main.py",
    "apps/web/package.json",
)

BASE_COMMANDS: dict[str, list[str]] = {
    "pip_install_base": [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ],
    "ruff": [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "apps",
        "packages",
        "scripts",
        "tests",
    ],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "npm_install": [NPM_COMMAND, "--prefix", "apps/web", "install"],
    "npm_lint": [NPM_COMMAND, "--prefix", "apps/web", "run", "lint"],
    "package_smoke": [sys.executable, "scripts/package_smoke.py"],
    "golden_generate": [sys.executable, "scripts/golden_generate.py"],
    "ai_contract_smoke": [sys.executable, "scripts/ai_contract_smoke.py"],
    "runtime_example_check": [
        sys.executable,
        "scripts/models/check_local_model_runtime.py",
        "--config",
        "configs/local_model_runtime.example.yaml",
    ],
    "pro_generate_professional_midi": [
        sys.executable,
        "scripts/models_pro/generate_professional_midi.py",
        "--profile",
        "pro",
        "--no-use-midigpt-infill",
        "--no-use-text2midi-sketch",
        "--run-id",
        "pr28_audit_professional_generation",
    ],
}


@dataclass(frozen=True)
class AuditOptions:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_commands: bool = True
    skip_install: bool = False
    command_timeout: int = 900
    stdout_limit: int = 4000


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    options = AuditOptions(
        output_root=_repo_path(args.output_root),
        run_commands=not args.skip_commands,
        skip_install=args.skip_install,
        command_timeout=args.timeout,
        stdout_limit=args.output_tail,
    )
    report = build_report(options)
    write_report(report, options.output_root)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


def build_report(options: AuditOptions) -> dict[str, Any]:
    env_example = parse_env_file(ROOT / ".env.example")
    gitignore_patterns = parse_gitignore(ROOT / ".gitignore")
    files = check_required_files(REQUIRED_FILES)
    env_checks = check_required_items(REQUIRED_ENV_EXAMPLE_KEYS, env_example.keys())
    gitignore_checks = check_required_items(
        REQUIRED_GITIGNORE_PATTERNS,
        normalized_gitignore_patterns(gitignore_patterns),
    )
    config_checks = check_configs()
    smoke_checks = check_smoke_scripts()
    command_results = (
        run_command_suite(options)
        if options.run_commands
        else {
            name: {
                "name": name,
                "cmd": cmd,
                "ok": True,
                "skipped": True,
                "reason": "command execution disabled",
            }
            for name, cmd in selected_commands(options).items()
        }
    )
    toolchain = collect_toolchain()

    failures = collect_failures(
        files=files,
        env_checks=env_checks,
        gitignore_checks=gitignore_checks,
        config_checks=config_checks,
        smoke_checks=smoke_checks,
        command_results=command_results,
    )
    status = "ok" if not failures else "fail"
    return {
        "schema_version": "0.1.0",
        "audit": "pr28_repo_health",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(ROOT),
        "toolchain": toolchain,
        "files": files,
        "env_example": {
            "path": str(ROOT / ".env.example"),
            "required": env_checks,
            "present_count": sum(1 for item in env_checks.values() if item),
            "missing": sorted(item for item, ok in env_checks.items() if not ok),
        },
        "gitignore": {
            "path": str(ROOT / ".gitignore"),
            "required": gitignore_checks,
            "present_count": sum(1 for item in gitignore_checks.values() if item),
            "missing": sorted(item for item, ok in gitignore_checks.items() if not ok),
        },
        "configs": config_checks,
        "smoke_scripts": smoke_checks,
        "commands": command_results,
        "failures": failures,
        "acceptance": {
            "make_lint": command_results.get("ruff", {}).get("ok", False)
            and command_results.get("npm_lint", {}).get("ok", False),
            "make_test": command_results.get("pytest", {}).get("ok", False),
            "make_package_smoke": command_results.get("package_smoke", {}).get("ok", False),
            "make_golden_baseline": command_results.get("golden_generate", {}).get("ok", False),
            "make_ai_contract_smoke": command_results.get("ai_contract_smoke", {}).get("ok", False),
        },
    }


def write_report(report: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pr28_repo_health.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "pr28_repo_health.md").write_text(
        report_markdown(report),
        encoding="utf-8",
    )


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PR-28 Repo Health",
        "",
        f"Status: **{report['status']}**",
        f"Generated: `{report['generated_at']}`",
        f"Repo: `{report['repo_root']}`",
        "",
        "## Acceptance",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for name, ok in report["acceptance"].items():
        lines.append(f"| `{name}` | {_status(ok)} |")

    lines.extend(["", "## Commands", "", "| Command | Status | Return Code |"])
    lines.append("| --- | --- | ---: |")
    for name, result in report["commands"].items():
        skipped = result.get("skipped", False)
        status = "SKIP" if skipped else _status(result.get("ok", False))
        returncode = result.get("returncode", "-")
        lines.append(f"| `{name}` | {status} | `{returncode}` |")

    lines.extend(["", "## Required Files", ""])
    for path, ok in report["files"].items():
        lines.append(f"- {_status(ok)} `{path}`")

    lines.extend(["", "## Environment Example", ""])
    for key, ok in report["env_example"]["required"].items():
        lines.append(f"- {_status(ok)} `{key}`")

    lines.extend(["", "## Gitignore", ""])
    for pattern, ok in report["gitignore"]["required"].items():
        lines.append(f"- {_status(ok)} `{pattern}`")

    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['category']}`: {failure['message']}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def run_command_suite(options: AuditOptions) -> dict[str, dict[str, Any]]:
    return {
        name: run_command(
            name,
            cmd,
            timeout=options.command_timeout,
            stdout_limit=options.stdout_limit,
        )
        for name, cmd in selected_commands(options).items()
    }


def selected_commands(options: AuditOptions) -> dict[str, list[str]]:
    if not options.skip_install:
        return dict(BASE_COMMANDS)
    return {
        name: cmd
        for name, cmd in BASE_COMMANDS.items()
        if name not in {"pip_install_base", "npm_install"}
    }


def run_command(
    name: str,
    cmd: list[str],
    *,
    timeout: int,
    stdout_limit: int,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=os.environ.copy(),
        )
        finished = datetime.now(UTC)
        return {
            "name": name,
            "cmd": cmd,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "stdout_tail": completed.stdout[-stdout_limit:],
            "stderr_tail": completed.stderr[-stdout_limit:],
        }
    except subprocess.TimeoutExpired as exc:
        finished = datetime.now(UTC)
        return {
            "name": name,
            "cmd": cmd,
            "ok": False,
            "returncode": None,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "timeout_seconds": timeout,
            "stdout_tail": _tail(exc.stdout, stdout_limit),
            "stderr_tail": _tail(exc.stderr, stdout_limit),
        }


def collect_failures(
    *,
    files: dict[str, bool],
    env_checks: dict[str, bool],
    gitignore_checks: dict[str, bool],
    config_checks: dict[str, dict[str, Any]],
    smoke_checks: dict[str, bool],
    command_results: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    failures.extend(_missing_failures("file", files))
    failures.extend(_missing_failures("env_example", env_checks))
    failures.extend(_missing_failures("gitignore", gitignore_checks))
    failures.extend(_missing_failures("smoke_script", smoke_checks))
    for path, result in config_checks.items():
        if not result.get("ok"):
            failures.append(
                {
                    "category": "config",
                    "message": f"{path}: {result.get('error', 'invalid config')}",
                }
            )
    for name, result in command_results.items():
        if not result.get("ok"):
            failures.append(
                {
                    "category": "command",
                    "message": f"{name} failed with returncode={result.get('returncode')}",
                }
            )
    return failures


def check_required_files(paths: tuple[str, ...]) -> dict[str, bool]:
    return {path: (ROOT / path).exists() for path in paths}


def check_required_items(required: tuple[str, ...], available: Any) -> dict[str, bool]:
    available_set = set(available)
    return {item: item in available_set for item in required}


def check_configs() -> dict[str, dict[str, Any]]:
    configs = (
        "configs/ai_models.yaml",
        "configs/ai_models.local.example.yaml",
        "configs/ai_models.pro.yaml",
        "configs/generation_profiles.pro.yaml",
        "configs/local_model_runtime.example.yaml",
        "configs/local_model_runtime.pro.yaml",
        "configs/model_registry.example.yaml",
        "configs/professional_benchmark_gate.pro.yaml",
        "configs/quality_thresholds.pro.yaml",
        "configs/professional_benchmarks.yaml",
    )
    return {path: load_config_status(ROOT / path) for path in configs}


def check_smoke_scripts() -> dict[str, bool]:
    scripts = (
        "scripts/package_smoke.py",
        "scripts/golden_generate.py",
        "scripts/ai_contract_smoke.py",
        "scripts/models/check_local_model_runtime.py",
        "scripts/models/smoke_midigpt.py",
        "scripts/models/smoke_text2midi.py",
        "scripts/models/smoke_ollama_planner.py",
        "scripts/models/smoke_miditok.py",
        "scripts/models/smoke_custom_role_models.py",
        "scripts/models/ai_local_smoke.py",
        "scripts/models/professional_generation_benchmark.py",
        "scripts/models_pro/generate_professional_midi.py",
        "scripts/models_pro/pro_quality_gate.py",
        "scripts/models_pro/professional_benchmark_gate.py",
    )
    return {path: (ROOT / path).exists() for path in scripts}


def load_config_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "exists": False, "error": "missing"}
    if yaml is None:
        return {"ok": False, "exists": True, "error": "PyYAML not importable"}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {
            "ok": isinstance(payload, dict),
            "exists": True,
            "keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        }
    except Exception as exc:
        return {"ok": False, "exists": True, "error": str(exc)}


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def parse_gitignore(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def normalized_gitignore_patterns(patterns: set[str]) -> set[str]:
    normalized = set(patterns)
    for pattern in patterns:
        normalized.add(pattern.lstrip("/"))
    return normalized


def collect_toolchain() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "node": command_version([NODE_COMMAND, "--version"]),
        "npm": command_version([NPM_COMMAND, "--version"]),
        "git": command_version([GIT_COMMAND, "--version"]),
        "make": command_version([MAKE_COMMAND, "--version"]),
        "ollama": command_version([OLLAMA_COMMAND, "--version"]),
        "path": os.environ.get("PATH", ""),
    }


def command_version(cmd: list[str]) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "version": output[0] if output else "",
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _missing_failures(category: str, checks: dict[str, bool]) -> list[dict[str, str]]:
    return [
        {"category": category, "message": f"missing {item}"}
        for item, ok in checks.items()
        if not ok
    ]


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _tail(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[-limit:]


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--skip-commands",
        action="store_true",
        help="Only inspect files/config/env/gitignore; do not run baseline commands.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Run baseline checks but skip pip/npm install commands.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-tail", type=int, default=4000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
