from __future__ import annotations

from app.main import app
from fastapi.testclient import TestClient


def test_ai_models_endpoint_lists_mock_and_disabled_optional_backends():
    client = TestClient(app)

    response = client.get("/v1/ai/models")

    assert response.status_code == 200
    payload = response.json()
    models = {model["id"]: model for model in payload["models"]}
    assert payload["status"] == "ok"
    assert models["mock_symbolic"]["status"] == "available"
    assert models["mock_symbolic"]["capabilities"]["symbolic_midi"] is True
    assert models["mock_symbolic"]["capabilities"]["bar_infill"] is True
    assert models["mock_symbolic"]["commercial_use"] == "allowed"
    assert models["midigpt"]["status"] == "disabled"
    assert models["text2midi"]["status"] == "disabled"
    assert models["custom_jazz_melody_v001"]["status"] == "unavailable"
    assert models["custom_jazz_melody_v001"]["backend_type"] == "custom_role"
    assert models["custom_jazz_melody_v001"]["metadata"]["role"] == "melody"
    assert payload["settings"]["forbid_audio_models"] is True


def test_ai_models_endpoint_can_hide_disabled_backends():
    client = TestClient(app)

    response = client.get("/v1/ai/models", params={"include_disabled": False})

    assert response.status_code == 200
    payload = response.json()
    assert [model["id"] for model in payload["models"]] == [
        "custom_jazz_drums_v001",
        "custom_jazz_horn_responses_v001",
        "custom_jazz_melody_v001",
        "custom_jazz_piano_comping_v001",
        "custom_jazz_walking_bass_v001",
        "mock_symbolic",
    ]
