from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.main import app
from arranger_core import ArrangementProject
from fastapi.testclient import TestClient


def test_text_to_midi_sketch_mock_imports_experimental_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "mock_symbolic",
            "prompt": "Hard bop minor blues sketch in C minor, alto sax lead",
            "seed": 601,
            "metadata": {"mock_artifact": "unlabeled_midi"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "sketch_uncertain"
    assert payload["backend"] == "mock_symbolic"
    assert payload["artifact"]["status"] == "validated"
    assert payload["validation"]["status"] in {"pass", "pass_with_warnings"}
    assert "no_roles_detected" in payload["sketch"]["uncertainty_reasons"]

    sketch_id = payload["sketch_id"]
    project_path = tmp_path / "api-storage" / "sketches" / sketch_id / "arrangement_project.json"
    assert project_path.exists()
    assert not (tmp_path / "api-storage" / "projects" / sketch_id).exists()

    project = ArrangementProject.load_json(project_path)
    assert project.metadata["project_type"] == "text2midi_sketch"
    assert project.metadata["professional_project"] is False
    assert project.metadata["auto_merge_allowed"] is False
    assert project.metadata["sketch_status"] == "sketch_uncertain"
    assert project.tracks
    assert all(track.metadata["source"] == "text2midi_sketch" for track in project.tracks)

    records = _artifact_records(tmp_path)
    assert records[0]["project_id"] == sketch_id
    assert records[0]["task"] == "generate_full_sketch"
    assert records[0]["status"] == "validated"


def test_text_to_midi_sketch_invalid_midi_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "mock_symbolic",
            "prompt": "Broken sketch fixture",
            "metadata": {"mock_artifact": "invalid_midi"},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "sketch_rejected"
    assert "MIDI sketch is not parseable" in detail["reason"]
    records = _artifact_records(tmp_path)
    assert records[0]["status"] == "rejected"


def test_text_to_midi_sketch_text2midi_missing_dependency_is_controlled(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  text2midi:
    enabled: true
    type: symbolic
    adapter: model_backends.symbolic.text2midi_backend.Text2MidiBackend
    commercial_use: review_required
    dependency_mode: optional
    install_hint: install Text2MIDI in the optional model worker profile
    tasks:
      - generate_full_sketch
    capabilities:
      symbolic_midi: true
      text_prompt: true
      commercial_use: review_required
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "text2midi",
            "prompt": "Hard bop minor blues sketch",
        },
    )

    assert response.status_code == 409
    assert "Text2MIDI is not installed" in response.json()["detail"]


def _artifact_records(tmp_path: Path) -> list[dict]:
    manifest_path = tmp_path / "api-storage" / "model_artifacts" / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload["artifacts"]
