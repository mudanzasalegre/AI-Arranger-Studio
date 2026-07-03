from __future__ import annotations

import json

from arranger_core.planning.provider_factory import build_planner_provider_from_registry
from model_backends.planner.ollama_planner_backend import OllamaPlannerBackend


def test_provider_factory_returns_none_when_local_planner_disabled(tmp_path, monkeypatch):
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  local_llm_planner:
    enabled: false
    type: planner
    adapter: model_backends.planner.ollama_planner_backend.OllamaPlannerBackend
    tasks:
      - plan_song
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))

    assert build_planner_provider_from_registry() is None


def test_provider_factory_returns_available_local_llm_planner(tmp_path, monkeypatch):
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  local_llm_planner:
    enabled: true
    type: planner
    adapter: model_backends.planner.ollama_planner_backend.OllamaPlannerBackend
    model_name: qwen3:8b
    base_url: http://127.0.0.1:11434/api
    commercial_use: review_required
    tasks:
      - plan_song
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(OllamaPlannerBackend, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaPlannerBackend,
        "generate_plan_json",
        lambda self, **kwargs: json.dumps({"style": "hard_bop"}),
    )

    provider = build_planner_provider_from_registry()

    assert isinstance(provider, OllamaPlannerBackend)
    assert provider.model_name == "qwen3:8b"


def test_provider_factory_returns_none_when_local_planner_unavailable(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  local_llm_planner:
    enabled: true
    type: planner
    adapter: model_backends.planner.ollama_planner_backend.OllamaPlannerBackend
    model_name: qwen3:8b
    base_url: http://127.0.0.1:11434/api
    tasks:
      - plan_song
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(OllamaPlannerBackend, "is_available", lambda self: False)

    assert build_planner_provider_from_registry() is None
