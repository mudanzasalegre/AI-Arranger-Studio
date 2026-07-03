from __future__ import annotations

from pathlib import Path

import pytest
from model_backends import (
    MockSymbolicBackend,
    ModelBackendRegistry,
    ModelBackendUnavailableError,
    ModelCapabilities,
    build_model_backend_registry,
    load_ai_models_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_registry_registers_and_lists_mock_backend():
    registry = ModelBackendRegistry()
    backend = MockSymbolicBackend(backend_id="mock_test")

    registry.register(
        backend,
        tasks=["infill_bars", "generate_full_sketch"],
        commercial_use="allowed",
    )

    assert registry.get("mock_test") is backend
    listed = registry.list()
    assert listed == [
        {
            "id": "mock_test",
            "backend_type": "symbolic",
            "enabled": True,
            "status": "available",
            "adapter": None,
            "capabilities": backend.capabilities.model_dump(mode="json"),
            "tasks": ["infill_bars", "generate_full_sketch"],
            "commercial_use": "allowed",
            "install_hint": None,
            "error": None,
            "metadata": {},
        }
    ]


def test_registry_rejects_disabled_backend_get():
    registry = ModelBackendRegistry()
    registry.register_configured(
        backend_id="disabled_backend",
        status="disabled",
        backend_type="symbolic",
        enabled=False,
        adapter="example.Disabled",
        capabilities=ModelCapabilities(symbolic_midi=True),
        tasks=["generate_track"],
        commercial_use="unknown",
    )

    with pytest.raises(ModelBackendUnavailableError):
        registry.get("disabled_backend")


def test_default_ai_models_config_builds_registry_with_disabled_optionals():
    config = load_ai_models_config(ROOT / "configs" / "ai_models.yaml")
    registry = build_model_backend_registry(config=config, include_disabled=True)
    models = {model["id"]: model for model in registry.list()}

    assert models["mock_symbolic"]["status"] == "available"
    assert models["mock_symbolic"]["capabilities"]["symbolic_midi"] is True
    assert models["mock_symbolic"]["capabilities"]["json_planning"] is True
    assert models["midigpt"]["status"] == "disabled"
    assert models["text2midi"]["status"] == "disabled"
    assert models["custom_jazz_melody_v001"]["status"] in {"available", "unavailable"}
    assert models["custom_jazz_melody_v001"]["backend_type"] == "custom_role"
    assert models["custom_jazz_melody_v001"]["metadata"]["role"] == "melody"
    if models["custom_jazz_melody_v001"]["status"] == "unavailable":
        assert "training_manifest.yaml" in models["custom_jazz_melody_v001"]["error"]
    assert registry.get("mock_symbolic").backend_id == "mock_symbolic"


def test_default_ai_models_config_can_hide_disabled_backends():
    config = load_ai_models_config(ROOT / "configs" / "ai_models.yaml")
    registry = build_model_backend_registry(config=config, include_disabled=False)

    assert registry.ids() == [
        "custom_jazz_drums_v001",
        "custom_jazz_horn_responses_v001",
        "custom_jazz_melody_v001",
        "custom_jazz_piano_comping_v001",
        "custom_jazz_walking_bass_v001",
        "mock_symbolic",
    ]
