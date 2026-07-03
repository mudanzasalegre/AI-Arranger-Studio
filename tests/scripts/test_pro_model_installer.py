from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_summary_returns_partial_ok_for_optional_failures():
    module = _load_module(
        "install_all_local_models_test",
        "scripts/models_pro/install_all_local_models.py",
    )
    results = [
        module.StepResult("python_version", ok=True, required=True),
        module.StepResult(
            "ollama_available",
            ok=False,
            instruction="Install Ollama, then pull qwen3:8b.",
        ),
    ]

    status, fatal_errors, nonfatal_errors = module.summarize_results(results)

    assert status == "partial_ok"
    assert fatal_errors == []
    assert nonfatal_errors == ["ollama_available: Install Ollama, then pull qwen3:8b."]


def test_install_summary_returns_fail_for_required_failures():
    module = _load_module(
        "install_all_local_models_required_test",
        "scripts/models_pro/install_all_local_models.py",
    )
    results = [
        module.StepResult(
            "check_python_version",
            ok=False,
            required=True,
            error="Python 3.13 is not in 3.10-3.12",
        )
    ]

    status, fatal_errors, nonfatal_errors = module.summarize_results(results)

    assert status == "fail"
    assert fatal_errors == ["check_python_version: Python 3.13 is not in 3.10-3.12"]
    assert nonfatal_errors == []


def test_install_ensure_directories_uses_repo_root(tmp_path):
    module = _load_module(
        "install_all_local_models_dirs_test",
        "scripts/models_pro/install_all_local_models.py",
    )
    module.ROOT = tmp_path

    result = module.ensure_directories(["models/hf_cache/hub", "outputs/model_smoke"])

    assert result.ok is True
    assert (tmp_path / "models/hf_cache/hub").is_dir()
    assert (tmp_path / "outputs/model_smoke").is_dir()


def test_activate_profile_updates_env_without_dropping_existing_values():
    module = _load_module(
        "activate_pro_profile_test",
        "scripts/models_pro/activate_pro_profile.py",
    )
    content = (
        "# local\n"
        "APP_ENV=development\n"
        "AI_MODELS_CONFIG=./configs/ai_models.local.yaml\n"
        "AI_ENABLE_MOCK_SYMBOLIC=true\n"
    )
    updates = {
        "AI_MODELS_CONFIG": "./configs/ai_models.pro.yaml",
        "AI_ENABLE_MOCK_SYMBOLIC": "false",
        "AI_ENABLE_MIDIGPT": "true",
    }

    updated, changed, appended = module.update_env_content(content, updates)

    assert "APP_ENV=development" in updated
    assert "AI_MODELS_CONFIG=./configs/ai_models.pro.yaml" in updated
    assert "AI_ENABLE_MOCK_SYMBOLIC=false" in updated
    assert "AI_ENABLE_MIDIGPT=true" in updated
    assert changed == ["AI_MODELS_CONFIG", "AI_ENABLE_MOCK_SYMBOLIC"]
    assert appended == ["AI_ENABLE_MIDIGPT"]


def test_verify_summary_accepts_nonfatal_model_failures():
    module = _load_module(
        "verify_all_models_test",
        "scripts/models_pro/verify_all_models.py",
    )
    checks = [
        module.CheckResult("required_files", ok=True, required=True),
        module.CheckResult(
            "text2midi_smoke",
            ok=False,
            model="text2midi",
            error="text2midi_smoke failed",
        ),
    ]

    status, fatal_errors, nonfatal_errors = module.summarize_checks(checks)

    assert status == "partial_ok"
    assert fatal_errors == []
    assert nonfatal_errors == ["text2midi_smoke: text2midi_smoke failed"]


def test_verify_custom_role_status_marks_unavailable_as_pending():
    module = _load_module(
        "verify_all_models_custom_roles_test",
        "scripts/models_pro/verify_all_models.py",
    )
    records = [
        {"id": "custom_jazz_melody_v001", "status": "unavailable"},
        {"id": "custom_jazz_drums_v001", "status": "available"},
    ]

    assert module._custom_role_status(records) == "pending_checkpoints"
