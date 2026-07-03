from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for package in ("model_backends",):
    sys.path.insert(0, str(ROOT / "packages" / package))

from model_backends import build_model_backend_registry, load_ai_models_config  # noqa: E402

MANIFEST_DIR = ROOT / "models" / "manifests"
MODEL_STATUS_PATH = MANIFEST_DIR / "model_status.json"
VERIFY_REPORT_PATH = ROOT / "outputs" / "model_smoke" / "verify_all_models.json"

LOCAL_ENV = {
    "AI_MODELS_CONFIG": "./configs/ai_models.pro.yaml",
    "LOCAL_MODEL_RUNTIME_CONFIG": "./configs/local_model_runtime.pro.yaml",
    "MODEL_REGISTRY_CONFIG": "./configs/model_registry.yaml",
    "AI_MODELS_ROOT": "./models",
    "CUSTOM_MODEL_ROOT": "./models/checkpoints/custom",
    "HF_HOME": "./models/hf_cache",
    "HF_HUB_CACHE": "./models/hf_cache/hub",
    "HF_ASSETS_CACHE": "./models/hf_cache/assets",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
REQUIRED_DIRS = (
    "models/hf_cache/hub",
    "models/hf_cache/assets",
    "models/external_repos",
    "models/checkpoints/text2midi",
    "models/checkpoints/custom/melody",
    "models/checkpoints/custom/bass",
    "models/checkpoints/custom/piano_comping",
    "models/checkpoints/custom/horns",
    "models/checkpoints/custom/drums",
    "models/manifests",
    "outputs/model_artifacts/raw",
    "outputs/model_artifacts/imported",
    "outputs/model_artifacts/rejected",
    "outputs/model_artifacts/validated",
    "outputs/model_smoke",
    "outputs/pro_benchmarks",
)
TEXT2MIDI_REQUIRED_FILES = ("pytorch_model.bin", "vocab_remi.pkl")


@dataclass
class CheckResult:
    name: str
    ok: bool
    required: bool = False
    skipped: bool = False
    category: str = "general"
    model: str | None = None
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    instruction: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, "", [])}


@dataclass(frozen=True)
class VerifyOptions:
    ai_config: Path
    runtime_config: Path
    planner_model: str
    timeout: int = 900
    skip_smoke: bool = False
    allow_text2midi_check_only: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all PR-29 local pro model wiring.")
    parser.add_argument("--ai-config", default="configs/ai_models.pro.yaml")
    parser.add_argument("--runtime-config", default="configs/local_model_runtime.pro.yaml")
    parser.add_argument("--planner-model", default="qwen3:8b")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-text2midi-check-only", action="store_true")
    args = parser.parse_args()

    options = VerifyOptions(
        ai_config=_repo_path(args.ai_config),
        runtime_config=_repo_path(args.runtime_config),
        planner_model=args.planner_model,
        timeout=args.timeout,
        skip_smoke=args.skip_smoke,
        allow_text2midi_check_only=args.allow_text2midi_check_only,
    )
    report = verify_all_models(options)
    print(json.dumps(report, indent=2))
    if report["status"] == "fail":
        raise SystemExit(1)


def verify_all_models(options: VerifyOptions) -> dict[str, Any]:
    set_local_env(options)
    checks: list[CheckResult] = [
        check_python_version(),
        check_required_files(options),
        check_required_directories(),
        run_runtime_check(options),
        inspect_registry(options),
    ]
    checks.extend(check_text2midi_files())
    checks.extend(run_smoke_checks(options))

    status, fatal_errors, nonfatal_errors = summarize_checks(checks)
    model_status = build_model_status(checks, options, status)
    report = {
        "schema_version": "0.1.0",
        "verifier": "pr29_verify_all_models",
        "status": status,
        "generated_at": _now(),
        "repo_root": str(ROOT),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "ai_config": str(options.ai_config),
        "runtime_config": str(options.runtime_config),
        "planner_model": options.planner_model,
        "fatal_errors": fatal_errors,
        "nonfatal_errors": nonfatal_errors,
        "checks": [check.to_json() for check in checks],
        "model_status": model_status,
    }
    write_json(VERIFY_REPORT_PATH, report)
    write_json(MODEL_STATUS_PATH, model_status)
    return report


def check_python_version() -> CheckResult:
    version = sys.version_info
    ok = version.major == 3 and 10 <= version.minor <= 12
    return CheckResult(
        name="python_version",
        ok=ok,
        required=True,
        details={"version": platform.python_version(), "accepted": "3.10-3.12"},
        error=None if ok else f"Python {platform.python_version()} is not in 3.10-3.12",
    )


def check_required_files(options: VerifyOptions) -> CheckResult:
    files = {
        "ai_config": options.ai_config,
        "runtime_config": options.runtime_config,
        "model_registry": ROOT / "configs" / "model_registry.yaml",
        "install_plan": ROOT / "configs" / "model_install_plan.yaml",
    }
    missing = {key: str(path) for key, path in files.items() if not path.exists()}
    return CheckResult(
        name="required_files",
        ok=not missing,
        required=True,
        details={key: str(path) for key, path in files.items()},
        error=None if not missing else f"Missing required file(s): {missing}",
    )


def check_required_directories() -> CheckResult:
    missing: list[str] = []
    unwritable: list[str] = []
    for relative_dir in REQUIRED_DIRS:
        path = ROOT / relative_dir
        if not path.exists():
            missing.append(relative_dir)
            continue
        if not _is_writable(path):
            unwritable.append(relative_dir)
    ok = not missing and not unwritable
    details = {"missing": missing, "unwritable": unwritable, "required": list(REQUIRED_DIRS)}
    return CheckResult(
        name="required_directories",
        ok=ok,
        required=True,
        details=details,
        error=None if ok else f"Directory setup incomplete: {details}",
    )


def run_runtime_check(options: VerifyOptions) -> CheckResult:
    return run_command(
        name="local_model_runtime_check",
        command=[
            sys.executable,
            "scripts/models/check_local_model_runtime.py",
            "--config",
            _relative_to_root(options.runtime_config),
        ],
        required=True,
        category="runtime",
        timeout=options.timeout,
    )


def inspect_registry(options: VerifyOptions) -> CheckResult:
    try:
        config = load_ai_models_config(options.ai_config)
        registry = build_model_backend_registry(
            config=config,
            include_disabled=True,
            include_unavailable=True,
        )
        records = registry.list()
    except Exception as exc:
        return CheckResult(
            name="model_backend_registry",
            ok=False,
            required=True,
            category="registry",
            error=str(exc),
        )

    unavailable = [
        {
            "id": item["id"],
            "type": item["backend_type"],
            "error": item.get("error"),
            "install_hint": item.get("install_hint"),
        }
        for item in records
        if item.get("enabled") and item.get("status") == "unavailable"
    ]
    return CheckResult(
        name="model_backend_registry",
        ok=True,
        required=True,
        category="registry",
        details={
            "registered": records,
            "enabled_unavailable": unavailable,
        },
    )


def check_text2midi_files() -> list[CheckResult]:
    repo_dir = ROOT / "models" / "external_repos" / "text2midi"
    checkpoint_dir = ROOT / "models" / "checkpoints" / "text2midi"
    repo_ok = (repo_dir / "model" / "transformer_model.py").exists()
    checkpoint_files = {
        filename: (checkpoint_dir / filename).exists()
        for filename in TEXT2MIDI_REQUIRED_FILES
    }
    return [
        CheckResult(
            name="text2midi_repo_files",
            ok=repo_ok,
            category="files",
            model="text2midi",
            details={"repo_dir": str(repo_dir)},
            error=None if repo_ok else f"Text2MIDI repo is incomplete: {repo_dir}",
            instruction=(
                None
                if repo_ok
                else "Run scripts/models_pro/install_all_local_models.py to clone Text2MIDI."
            ),
        ),
        CheckResult(
            name="text2midi_checkpoint_files",
            ok=all(checkpoint_files.values()),
            category="files",
            model="text2midi",
            details={"checkpoint_dir": str(checkpoint_dir), "files": checkpoint_files},
            error=None if all(checkpoint_files.values()) else "Missing Text2MIDI checkpoint files",
            instruction=(
                None
                if all(checkpoint_files.values())
                else (
                    "Run scripts/models/download_text2midi.py "
                    "--checkpoint-dir models/checkpoints/text2midi."
                )
            ),
        ),
    ]


def run_smoke_checks(options: VerifyOptions) -> list[CheckResult]:
    if options.skip_smoke:
        return [
            CheckResult(
                name="smoke_checks",
                ok=False,
                skipped=True,
                category="smoke",
                instruction="Re-run without --skip-smoke to execute model smoke tests.",
            )
        ]

    checks = [
        run_command(
            name="midigpt_smoke",
            command=[sys.executable, "scripts/models/smoke_midigpt.py"],
            category="smoke",
            model="midigpt",
            timeout=options.timeout,
        ),
        run_command(
            name="miditok_smoke",
            command=[sys.executable, "scripts/models/smoke_miditok.py"],
            category="smoke",
            model="miditok",
            timeout=options.timeout,
        ),
    ]
    text2midi_cmd = [sys.executable, "scripts/models/smoke_text2midi.py"]
    if options.allow_text2midi_check_only:
        text2midi_cmd.append("--allow-check-only")
    checks.append(
        run_command(
            name="text2midi_smoke",
            command=text2midi_cmd,
            category="smoke",
            model="text2midi",
            timeout=options.timeout,
        )
    )
    checks.append(run_ollama_smoke(options))
    return checks


def run_ollama_smoke(options: VerifyOptions) -> CheckResult:
    ollama = shutil.which("ollama")
    if not ollama:
        return CheckResult(
            name="ollama_planner_smoke",
            ok=False,
            category="smoke",
            model="ollama_planner",
            error="Ollama executable not found",
            instruction=(
                "Install Ollama from https://ollama.com/download, start it, "
                f"then run `ollama pull {options.planner_model}`."
            ),
        )
    return enrich_ollama_smoke_result(
        run_command(
            name="ollama_planner_smoke",
            command=[
                sys.executable,
                "scripts/models/smoke_ollama_planner.py",
                "--model",
                options.planner_model,
            ],
            category="smoke",
            model="ollama_planner",
            timeout=options.timeout,
        )
    )


def run_command(
    *,
    name: str,
    command: list[str],
    required: bool = False,
    category: str = "command",
    model: str | None = None,
    timeout: int = 900,
) -> CheckResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=subprocess_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return CheckResult(
            name=name,
            ok=completed.returncode == 0,
            required=required,
            category=category,
            model=model,
            command=[str(item) for item in command],
            returncode=completed.returncode,
            stdout=_tail(completed.stdout),
            stderr=_tail(completed.stderr),
            error=None if completed.returncode == 0 else f"{name} failed",
            duration_seconds=round(time.perf_counter() - start, 3),
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            name=name,
            ok=False,
            required=required,
            category=category,
            model=model,
            command=[str(item) for item in command],
            stdout=_tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr=_tail(exc.stderr if isinstance(exc.stderr, str) else ""),
            error=f"{name} timed out after {timeout} seconds",
            duration_seconds=round(time.perf_counter() - start, 3),
        )
    except OSError as exc:
        return CheckResult(
            name=name,
            ok=False,
            required=required,
            category=category,
            model=model,
            command=[str(item) for item in command],
            error=str(exc),
            duration_seconds=round(time.perf_counter() - start, 3),
        )


def summarize_checks(checks: list[CheckResult]) -> tuple[str, list[str], list[str]]:
    fatal = [_message(check) for check in checks if check.required and not check.ok]
    nonfatal = [_message(check) for check in checks if not check.required and not check.ok]
    if fatal:
        return "fail", fatal, nonfatal
    if nonfatal:
        return "partial_ok", [], nonfatal
    return "ok", [], []


def build_model_status(
    checks: list[CheckResult],
    options: VerifyOptions,
    overall_status: str,
) -> dict[str, Any]:
    by_model: dict[str, list[CheckResult]] = {}
    for check in checks:
        if check.model:
            by_model.setdefault(check.model, []).append(check)
    models = {
        "midigpt": {
            "status": _composite_status(by_model.get("midigpt", [])),
            "smoke_report": str(ROOT / "outputs/model_smoke/midigpt_smoke_summary.json"),
        },
        "text2midi": {
            "status": _composite_status(by_model.get("text2midi", [])),
            "repo_dir": str(ROOT / "models/external_repos/text2midi"),
            "checkpoint_dir": str(ROOT / "models/checkpoints/text2midi"),
            "smoke_report": str(ROOT / "outputs/model_smoke/text2midi_smoke_summary.json"),
        },
        "ollama_planner": {
            "status": _composite_status(by_model.get("ollama_planner", [])),
            "model_name": options.planner_model,
            "smoke_report": str(ROOT / "outputs/model_smoke/ollama_planner_smoke_summary.json"),
        },
        "miditok": {
            "status": _composite_status(by_model.get("miditok", [])),
            "smoke_report": str(ROOT / "outputs/model_smoke/miditok_smoke_summary.json"),
        },
    }
    registry_check = next(
        (check for check in checks if check.name == "model_backend_registry"),
        None,
    )
    if registry_check:
        custom_records = [
            item
            for item in registry_check.details.get("registered", [])
            if str(item.get("id", "")).startswith("custom_")
        ]
        models["custom_role_models"] = {
            "status": _custom_role_status(custom_records),
            "records": custom_records,
        }
    return {
        "schema_version": "0.1.0",
        "status": overall_status,
        "generated_at": _now(),
        "profile": "pro",
        "planner_model": options.planner_model,
        "models": models,
    }


def set_local_env(options: VerifyOptions) -> None:
    env = dict(LOCAL_ENV)
    env["AI_MODELS_CONFIG"] = _relative_to_root(options.ai_config)
    env["LOCAL_MODEL_RUNTIME_CONFIG"] = _relative_to_root(options.runtime_config)
    env["OLLAMA_PLANNER_MODEL"] = options.planner_model
    for key, value in env.items():
        os.environ[key] = value


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(LOCAL_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def enrich_ollama_smoke_result(result: CheckResult) -> CheckResult:
    if result.ok:
        return result
    summary_path = ROOT / "outputs" / "model_smoke" / "ollama_planner_smoke_summary.json"
    if not summary_path.exists():
        return result
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return result
    result.details["summary_path"] = str(summary_path)
    result.details["planner"] = summary.get("planner")
    result.details["fallback_used"] = summary.get("fallback_used")
    result.details["validation"] = summary.get("validation")
    result.details["attempt_errors"] = [
        attempt.get("error")
        for attempt in summary.get("attempts", [])
        if isinstance(attempt, dict) and attempt.get("error")
    ]
    availability = summary.get("availability") or {}
    validation = summary.get("validation") or {}
    if availability.get("available") and summary.get("fallback_used"):
        result.error = (
            "Ollama model responded, but the LLM planner smoke fell back after invalid "
            f"LLM JSON. Fallback validation status: {validation.get('status')}. "
            f"See {summary_path}."
        )
        result.instruction = (
            "This is nonfatal for PR-29 verification; rerun the smoke or tune the "
            "planner prompt if strict LLM-only planning is required."
        )
    return result


def _is_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _message(check: CheckResult) -> str:
    prefix = f"{check.name}: "
    if check.error:
        return prefix + check.error
    if check.instruction:
        return prefix + check.instruction
    if check.skipped:
        return prefix + "skipped"
    return prefix + "not ok"


def _composite_status(checks: list[CheckResult]) -> str:
    if not checks:
        return "not_checked"
    if any(not check.ok and not check.skipped for check in checks):
        return "fail"
    if any(check.skipped and not check.ok for check in checks):
        return "skipped"
    return "ok" if all(check.ok for check in checks) else "partial_ok"


def _custom_role_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "not_configured"
    statuses = {str(record.get("status")) for record in records}
    if statuses == {"available"}:
        return "ok"
    if "unavailable" in statuses:
        return "pending_checkpoints"
    return "partial_ok"


def _tail(text: str | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[-limit:]


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
