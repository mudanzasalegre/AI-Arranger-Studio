from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "models_pro" / "pro_readiness_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pro_readiness_audit", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_env_file_ignores_comments_and_blank_lines(tmp_path):
    module = _load_module()
    env_path = tmp_path / ".env.example"
    env_path.write_text(
        "\n# comment\nAI_MODELS_CONFIG=./configs/ai_models.local.yaml\nHF_HOME=./models/hf_cache\n",
        encoding="utf-8",
    )

    values = module.parse_env_file(env_path)

    assert values == {
        "AI_MODELS_CONFIG": "./configs/ai_models.local.yaml",
        "HF_HOME": "./models/hf_cache",
    }


def test_gitignore_patterns_are_normalized():
    module = _load_module()

    normalized = module.normalized_gitignore_patterns({"/models/", "outputs/", "*.mid"})

    assert "models/" in normalized
    assert "outputs/" in normalized
    assert "*.mid" in normalized


def test_build_report_can_skip_commands(tmp_path):
    module = _load_module()
    options = module.AuditOptions(output_root=tmp_path, run_commands=False)

    report = module.build_report(options)

    assert report["audit"] == "pr28_repo_health"
    assert report["status"] == "ok"
    assert report["env_example"]["missing"] == []
    assert report["gitignore"]["missing"] == []
    assert all(result["skipped"] for result in report["commands"].values())
    assert report["acceptance"]["make_lint"] is True


def test_report_markdown_is_ascii_and_mentions_failures():
    module = _load_module()
    report = {
        "status": "fail",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "repo_root": str(ROOT),
        "acceptance": {"make_lint": False},
        "commands": {"ruff": {"ok": False, "returncode": 1}},
        "files": {"README.md": True},
        "env_example": {"required": {"AI_MODELS_CONFIG": True}},
        "gitignore": {"required": {"models/": True}},
        "failures": [{"category": "command", "message": "ruff failed"}],
    }

    markdown = module.report_markdown(report)

    assert "PR-28 Repo Health" in markdown
    assert "ruff failed" in markdown
    markdown.encode("ascii")
