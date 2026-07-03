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

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PyYAML. Run: python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "models" / "manifests"
INSTALL_REPORT_PATH = MANIFEST_DIR / "install_report.json"
MODEL_STATUS_PATH = MANIFEST_DIR / "model_status.json"

LOCAL_HF_ENV = {
    "HF_HOME": "./models/hf_cache",
    "HF_HUB_CACHE": "./models/hf_cache/hub",
    "HF_ASSETS_CACHE": "./models/hf_cache/assets",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
PRO_ENV = {
    "AI_MODELS_CONFIG": "./configs/ai_models.pro.yaml",
    "LOCAL_MODEL_RUNTIME_CONFIG": "./configs/local_model_runtime.pro.yaml",
    "MODEL_REGISTRY_CONFIG": "./configs/model_registry.yaml",
    "AI_MODELS_ROOT": "./models",
    "CUSTOM_MODEL_ROOT": "./models/checkpoints/custom",
}
TEXT2MIDI_REQUIRED_FILES = ("pytorch_model.bin", "vocab_remi.pkl")


@dataclass
class StepResult:
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
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, "", [])}


@dataclass(frozen=True)
class InstallOptions:
    plan_path: Path
    profile: str
    planner_model: str
    skip_pip: bool = False
    skip_downloads: bool = False
    skip_smoke: bool = False
    allow_text2midi_check_only: bool = False
    timeout: int = 1200


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify local pro models.")
    parser.add_argument("--profile", default="pro")
    parser.add_argument("--planner-model", default="qwen3:8b")
    parser.add_argument("--plan", default="configs/model_install_plan.yaml")
    parser.add_argument("--skip-pip", action="store_true")
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-text2midi-check-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    options = InstallOptions(
        plan_path=_repo_path(args.plan),
        profile=args.profile,
        planner_model=args.planner_model,
        skip_pip=args.skip_pip,
        skip_downloads=args.skip_downloads,
        skip_smoke=args.skip_smoke,
        allow_text2midi_check_only=args.allow_text2midi_check_only,
        timeout=args.timeout,
    )
    report = install_all(options)
    print(json.dumps(report, indent=2))
    if report["status"] == "fail":
        raise SystemExit(1)


def install_all(options: InstallOptions) -> dict[str, Any]:
    set_local_model_env()
    results: list[StepResult] = []
    started_at = _now()

    plan: dict[str, Any] = {}
    try:
        plan = load_plan(options.plan_path)
        results.append(_check_plan_profile(plan, options.profile))
    except Exception as exc:
        results.append(
            StepResult(
                name="load_install_plan",
                ok=False,
                required=True,
                error=str(exc),
                details={"plan_path": str(options.plan_path)},
            )
        )

    if plan:
        results.append(check_python_version())
        results.append(ensure_directories(plan.get("directories", [])))
        results.extend(install_requirements(plan, options))
        results.extend(install_midigpt(plan, options))
        results.extend(install_text2midi(plan, options))
        results.extend(install_ollama_planner(plan, options))
        results.extend(run_miditok_smoke(plan, options))

    status, fatal_errors, nonfatal_errors = summarize_results(results)
    model_status = build_model_status(results, options)
    report = {
        "schema_version": "0.1.0",
        "installer": "pr29_automatic_local_model_installer",
        "status": status,
        "generated_at": _now(),
        "started_at": started_at,
        "repo_root": str(ROOT),
        "profile": options.profile,
        "planner_model": options.planner_model,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "environment": {
            **LOCAL_HF_ENV,
            **PRO_ENV,
        },
        "options": {
            "skip_pip": options.skip_pip,
            "skip_downloads": options.skip_downloads,
            "skip_smoke": options.skip_smoke,
            "allow_text2midi_check_only": options.allow_text2midi_check_only,
            "timeout": options.timeout,
        },
        "fatal_errors": fatal_errors,
        "nonfatal_errors": nonfatal_errors,
        "steps": [result.to_json() for result in results],
        "model_status": model_status,
    }
    write_json(INSTALL_REPORT_PATH, report)
    write_json(MODEL_STATUS_PATH, model_status)
    return report


def load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Install plan not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Install plan must be a mapping: {path}")
    return data


def _check_plan_profile(plan: dict[str, Any], expected_profile: str) -> StepResult:
    plan_profile = str(plan.get("profile") or "")
    ok = plan_profile == expected_profile
    return StepResult(
        name="check_install_plan",
        ok=ok,
        required=True,
        details={
            "plan_profile": plan_profile,
            "expected_profile": expected_profile,
            "schema_version": plan.get("schema_version"),
        },
        error=(
            None
            if ok
            else f"Install plan profile is {plan_profile!r}, expected {expected_profile!r}"
        ),
    )


def check_python_version() -> StepResult:
    version = sys.version_info
    ok = version.major == 3 and 10 <= version.minor <= 12
    return StepResult(
        name="check_python_version",
        ok=ok,
        required=True,
        details={"version": platform.python_version(), "accepted": "3.10-3.12"},
        error=None if ok else f"Python {platform.python_version()} is not in 3.10-3.12",
    )


def ensure_directories(relative_dirs: list[str]) -> StepResult:
    created_or_verified: list[str] = []
    try:
        for relative_dir in relative_dirs:
            path = _repo_path(relative_dir)
            path.mkdir(parents=True, exist_ok=True)
            created_or_verified.append(relative_dir)
        return StepResult(
            name="ensure_directories",
            ok=True,
            required=True,
            details={"created_or_verified": created_or_verified},
        )
    except OSError as exc:
        return StepResult(
            name="ensure_directories",
            ok=False,
            required=True,
            error=str(exc),
            details={"created_or_verified": created_or_verified},
        )


def install_requirements(plan: dict[str, Any], options: InstallOptions) -> list[StepResult]:
    pip_plan = plan.get("pip", {})
    if not isinstance(pip_plan, dict):
        return [
            StepResult(
                name="pip_plan",
                ok=False,
                required=True,
                error="pip section must be a mapping",
            )
        ]

    results: list[StepResult] = []
    for group in ("base", "ai", "training"):
        files = pip_plan.get(group, [])
        if not isinstance(files, list):
            results.append(
                StepResult(
                    name=f"pip_{group}",
                    ok=False,
                    required=True,
                    error=f"pip.{group} must be a list",
                )
            )
            continue
        for requirements_file in files:
            name = f"pip_install_{group}_{Path(str(requirements_file)).stem}"
            if options.skip_pip:
                results.append(
                    StepResult(
                        name=name,
                        ok=False,
                        skipped=True,
                        category="pip",
                        instruction="Re-run without --skip-pip to install requirements.",
                        details={"requirements_file": requirements_file},
                    )
                )
                continue
            path = _repo_path(str(requirements_file))
            if not path.exists():
                results.append(
                    StepResult(
                        name=name,
                        ok=False,
                        required=True,
                        category="pip",
                        error=f"Requirements file not found: {path}",
                    )
                )
                continue
            results.append(
                run_command(
                    name=name,
                    command=[sys.executable, "-m", "pip", "install", "-r", str(path)],
                    required=True,
                    category="pip",
                    timeout=options.timeout,
                )
            )
    return results


def install_midigpt(plan: dict[str, Any], options: InstallOptions) -> list[StepResult]:
    model_plan = _model_plan(plan, "midigpt")
    if not model_plan.get("enabled", False):
        return []
    model_name = str(model_plan.get("model_name") or "yellow")
    results: list[StepResult] = []
    if options.skip_downloads:
        results.append(
            StepResult(
                name="midigpt_download",
                ok=False,
                skipped=True,
                category="download",
                model="midigpt",
                instruction="Re-run without --skip-downloads to cache MIDI-GPT locally.",
            )
        )
    else:
        results.append(
            run_command(
                name="midigpt_download",
                command=[
                    sys.executable,
                    "scripts/models/download_midigpt.py",
                    "--model-name",
                    model_name,
                ],
                category="download",
                model="midigpt",
                timeout=options.timeout,
            )
        )
    if options.skip_smoke:
        results.append(
            StepResult(
                name="midigpt_smoke",
                ok=False,
                skipped=True,
                category="smoke",
                model="midigpt",
                instruction="Re-run without --skip-smoke to validate MIDI-GPT inference.",
            )
        )
    else:
        results.append(
            run_command(
                name="midigpt_smoke",
                command=[
                    sys.executable,
                    "scripts/models/smoke_midigpt.py",
                    "--model-name",
                    model_name,
                ],
                category="smoke",
                model="midigpt",
                timeout=options.timeout,
            )
        )
    return results


def install_text2midi(plan: dict[str, Any], options: InstallOptions) -> list[StepResult]:
    model_plan = _model_plan(plan, "text2midi")
    if not model_plan.get("enabled", False):
        return []
    results = [
        ensure_text2midi_repo(model_plan, options),
        install_text2midi_requirements(model_plan, options),
        download_text2midi_checkpoints(model_plan, options),
    ]
    if options.skip_smoke:
        results.append(
            StepResult(
                name="text2midi_smoke",
                ok=False,
                skipped=True,
                category="smoke",
                model="text2midi",
                instruction="Re-run without --skip-smoke to validate Text2MIDI inference.",
            )
        )
    else:
        cmd = [sys.executable, "scripts/models/smoke_text2midi.py"]
        if options.allow_text2midi_check_only:
            cmd.append("--allow-check-only")
        results.append(
            run_command(
                name="text2midi_smoke",
                command=cmd,
                category="smoke",
                model="text2midi",
                timeout=options.timeout,
            )
        )
    return results


def install_text2midi_requirements(
    model_plan: dict[str, Any],
    options: InstallOptions,
) -> StepResult:
    repo_dir = _repo_path(str(model_plan.get("repo_dir") or "models/external_repos/text2midi"))
    requirements_file = _text2midi_requirements_file(repo_dir)
    if options.skip_pip:
        return StepResult(
            name="text2midi_requirements",
            ok=False,
            skipped=True,
            category="pip",
            model="text2midi",
            instruction="Re-run without --skip-pip to install Text2MIDI requirements.",
            details={"requirements_file": str(requirements_file)},
        )
    if not requirements_file.exists():
        return StepResult(
            name="text2midi_requirements",
            ok=False,
            category="pip",
            model="text2midi",
            error=f"Text2MIDI requirements file not found: {requirements_file}",
            instruction="Clone Text2MIDI first, then re-run the installer.",
        )
    return run_command(
        name="text2midi_requirements",
        command=[sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        category="pip",
        model="text2midi",
        timeout=options.timeout,
    )


def ensure_text2midi_repo(model_plan: dict[str, Any], options: InstallOptions) -> StepResult:
    repo_dir = _repo_path(str(model_plan.get("repo_dir") or "models/external_repos/text2midi"))
    repo_url = str(model_plan.get("repo_url") or "https://github.com/AMAAI-Lab/text2midi")
    if _text2midi_repo_present(repo_dir):
        return StepResult(
            name="text2midi_repo",
            ok=True,
            category="download",
            model="text2midi",
            details={"repo_dir": str(repo_dir), "reason": "already_present"},
        )
    if options.skip_downloads:
        return StepResult(
            name="text2midi_repo",
            ok=False,
            skipped=True,
            category="download",
            model="text2midi",
            instruction="Re-run without --skip-downloads to clone Text2MIDI.",
            details={"repo_dir": str(repo_dir), "repo_url": repo_url},
        )
    if repo_dir.exists() and any(repo_dir.iterdir()):
        return StepResult(
            name="text2midi_repo",
            ok=False,
            category="download",
            model="text2midi",
            error=f"Text2MIDI directory exists but is not a usable checkout: {repo_dir}",
            instruction="Move or fix the directory, then re-run the installer.",
        )
    git = shutil.which("git")
    if not git:
        return StepResult(
            name="text2midi_repo",
            ok=False,
            category="download",
            model="text2midi",
            error="git executable not found",
            instruction=f"Install Git or clone {repo_url} into {repo_dir}.",
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        name="text2midi_repo",
        command=[git, "clone", "--depth", "1", repo_url, str(repo_dir)],
        category="download",
        model="text2midi",
        timeout=options.timeout,
    )


def download_text2midi_checkpoints(
    model_plan: dict[str, Any],
    options: InstallOptions,
) -> StepResult:
    checkpoint_dir = str(model_plan.get("checkpoint_dir") or "models/checkpoints/text2midi")
    repo_id = str(model_plan.get("hf_repo_id") or "amaai-lab/text2midi")
    flan_tokenizer = str(model_plan.get("flan_tokenizer") or "google/flan-t5-base")
    if options.skip_downloads:
        return StepResult(
            name="text2midi_checkpoints",
            ok=False,
            skipped=True,
            category="download",
            model="text2midi",
            instruction="Re-run without --skip-downloads to download Text2MIDI weights.",
            details={"checkpoint_dir": checkpoint_dir, "repo_id": repo_id},
        )
    return run_command(
        name="text2midi_checkpoints",
        command=[
            sys.executable,
            "scripts/models/download_text2midi.py",
            "--repo-id",
            repo_id,
            "--checkpoint-dir",
            checkpoint_dir,
            "--flan-tokenizer",
            flan_tokenizer,
        ],
        category="download",
        model="text2midi",
        timeout=options.timeout,
    )


def install_ollama_planner(plan: dict[str, Any], options: InstallOptions) -> list[StepResult]:
    model_plan = _model_plan(plan, "ollama_planner")
    if not model_plan.get("enabled", False):
        return []
    model_name = options.planner_model or str(model_plan.get("model_name") or "qwen3:8b")
    ollama = shutil.which("ollama")
    if not ollama:
        return [
            StepResult(
                name="ollama_available",
                ok=False,
                category="external_app",
                model="ollama_planner",
                error="Ollama executable not found",
                instruction=(
                    "Install Ollama from https://ollama.com/download, start it, "
                    f"then run `ollama pull {model_name}`."
                ),
            )
        ]

    results = [
        run_command(
            name="ollama_available",
            command=[ollama, "--version"],
            category="external_app",
            model="ollama_planner",
            timeout=120,
        )
    ]
    if options.skip_downloads:
        results.append(
            StepResult(
                name="ollama_pull",
                ok=False,
                skipped=True,
                category="external_app",
                model="ollama_planner",
                instruction=f"Re-run without --skip-downloads or run `ollama pull {model_name}`.",
            )
        )
    else:
        results.append(
            run_command(
                name="ollama_pull",
                command=[ollama, "pull", model_name],
                category="external_app",
                model="ollama_planner",
                timeout=options.timeout,
            )
        )
    if options.skip_smoke:
        results.append(
            StepResult(
                name="ollama_planner_smoke",
                ok=False,
                skipped=True,
                category="smoke",
                model="ollama_planner",
                instruction="Re-run without --skip-smoke to validate the Ollama planner.",
            )
        )
    else:
        results.append(
            enrich_ollama_smoke_result(
                run_command(
                    name="ollama_planner_smoke",
                    command=[
                        sys.executable,
                        "scripts/models/smoke_ollama_planner.py",
                        "--model",
                        model_name,
                    ],
                    category="smoke",
                    model="ollama_planner",
                    timeout=options.timeout,
                )
            )
        )
    return results


def run_miditok_smoke(plan: dict[str, Any], options: InstallOptions) -> list[StepResult]:
    model_plan = _model_plan(plan, "miditok")
    if not model_plan.get("enabled", False):
        return []
    if options.skip_smoke:
        return [
            StepResult(
                name="miditok_smoke",
                ok=False,
                skipped=True,
                category="smoke",
                model="miditok",
                instruction="Re-run without --skip-smoke to validate MidiTok tokenization.",
            )
        ]
    return [
        run_command(
            name="miditok_smoke",
            command=[sys.executable, "scripts/models/smoke_miditok.py"],
            category="smoke",
            model="miditok",
            timeout=options.timeout,
        )
    ]


def run_command(
    *,
    name: str,
    command: list[str],
    required: bool = False,
    category: str = "command",
    model: str | None = None,
    timeout: int = 1200,
) -> StepResult:
    started = _now()
    start_time = time.perf_counter()
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
        duration = round(time.perf_counter() - start_time, 3)
        return StepResult(
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
            started_at=started,
            ended_at=_now(),
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired as exc:
        return StepResult(
            name=name,
            ok=False,
            required=required,
            category=category,
            model=model,
            command=[str(item) for item in command],
            stdout=_tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr=_tail(exc.stderr if isinstance(exc.stderr, str) else ""),
            error=f"{name} timed out after {timeout} seconds",
            started_at=started,
            ended_at=_now(),
            duration_seconds=round(time.perf_counter() - start_time, 3),
        )
    except OSError as exc:
        return StepResult(
            name=name,
            ok=False,
            required=required,
            category=category,
            model=model,
            command=[str(item) for item in command],
            error=str(exc),
            started_at=started,
            ended_at=_now(),
            duration_seconds=round(time.perf_counter() - start_time, 3),
        )


def summarize_results(results: list[StepResult]) -> tuple[str, list[str], list[str]]:
    fatal = [_result_message(result) for result in results if result.required and not result.ok]
    nonfatal = [
        _result_message(result)
        for result in results
        if not result.required and (not result.ok or (result.skipped and not result.ok))
    ]
    if fatal:
        status = "fail"
    elif nonfatal:
        status = "partial_ok"
    else:
        status = "ok"
    return status, fatal, nonfatal


def build_model_status(results: list[StepResult], options: InstallOptions) -> dict[str, Any]:
    by_model: dict[str, list[StepResult]] = {}
    for result in results:
        if result.model:
            by_model.setdefault(result.model, []).append(result)

    text2midi_checkpoint_dir = ROOT / "models" / "checkpoints" / "text2midi"
    custom_root = ROOT / "models" / "checkpoints" / "custom"
    status = {
        "schema_version": "0.1.0",
        "generated_at": _now(),
        "profile": options.profile,
        "planner_model": options.planner_model,
        "models": {
            "midigpt": {
                "status": _composite_status(by_model.get("midigpt", [])),
                "download_report": str(ROOT / "models/manifests/midigpt_download_report.json"),
                "smoke_report": str(ROOT / "outputs/model_smoke/midigpt_smoke_summary.json"),
            },
            "text2midi": {
                "status": _composite_status(by_model.get("text2midi", [])),
                "repo_dir": str(ROOT / "models/external_repos/text2midi"),
                "checkpoint_dir": str(text2midi_checkpoint_dir),
                "checkpoint_files": {
                    filename: (text2midi_checkpoint_dir / filename).exists()
                    for filename in TEXT2MIDI_REQUIRED_FILES
                },
                "smoke_report": str(ROOT / "outputs/model_smoke/text2midi_smoke_summary.json"),
            },
            "ollama_planner": {
                "status": _composite_status(by_model.get("ollama_planner", [])),
                "model_name": options.planner_model,
                "smoke_report": str(
                    ROOT / "outputs/model_smoke/ollama_planner_smoke_summary.json"
                ),
            },
            "miditok": {
                "status": _composite_status(by_model.get("miditok", [])),
                "smoke_report": str(ROOT / "outputs/model_smoke/miditok_smoke_summary.json"),
            },
            "custom_role_models": {
                "status": "pending_checkpoints",
                "checkpoint_root": str(custom_root),
                "roles": {
                    "melody": (custom_root / "melody").exists(),
                    "walking_bass": (custom_root / "bass").exists(),
                    "piano_comping": (custom_root / "piano_comping").exists(),
                    "horn_responses": (custom_root / "horns").exists(),
                    "drums": (custom_root / "drums").exists(),
                },
            },
        },
        "steps": {
            result.name: {
                "ok": result.ok,
                "skipped": result.skipped,
                "returncode": result.returncode,
                "error": result.error,
                "instruction": result.instruction,
            }
            for result in results
        },
    }
    model_statuses = [
        item["status"]
        for key, item in status["models"].items()
        if key != "custom_role_models"
    ]
    status["status"] = (
        "ok"
        if model_statuses and all(item == "ok" for item in model_statuses)
        else "partial_ok"
    )
    return status


def set_local_model_env() -> None:
    for key, value in {**LOCAL_HF_ENV, **PRO_ENV}.items():
        os.environ[key] = value


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(LOCAL_HF_ENV)
    env.update(PRO_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def enrich_ollama_smoke_result(result: StepResult) -> StepResult:
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
            "This is nonfatal for PR-29 installation; rerun the smoke or tune the "
            "planner prompt if strict LLM-only planning is required."
        )
    return result


def _model_plan(plan: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = plan.get("models", {})
    if not isinstance(models, dict):
        return {}
    model_plan = models.get(model_id, {})
    return model_plan if isinstance(model_plan, dict) else {}


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _text2midi_repo_present(repo_dir: Path) -> bool:
    return (repo_dir / ".git").exists() or (repo_dir / "model" / "transformer_model.py").exists()


def _text2midi_requirements_file(repo_dir: Path) -> Path:
    if platform.system().lower() == "darwin" and (repo_dir / "requirements-mac.txt").exists():
        return repo_dir / "requirements-mac.txt"
    return repo_dir / "requirements.txt"


def _composite_status(results: list[StepResult]) -> str:
    if not results:
        return "not_checked"
    if any(not result.ok and not result.skipped for result in results):
        return "fail"
    if any(result.skipped and not result.ok for result in results):
        return "skipped"
    if all(result.ok for result in results):
        return "ok"
    return "partial_ok"


def _result_message(result: StepResult) -> str:
    prefix = f"{result.name}: "
    if result.error:
        return prefix + result.error
    if result.instruction:
        return prefix + result.instruction
    if result.skipped:
        return prefix + "skipped"
    return prefix + "not ok"


def _tail(text: str | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[-limit:]


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
