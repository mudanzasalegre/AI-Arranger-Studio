from __future__ import annotations

import json

from app.main import app
from arranger_core import ArrangementProject
from fastapi.testclient import TestClient
from model_backends.planner.ollama_planner_backend import OllamaPlannerBackend


def test_ai_plan_endpoint_versions_song_plan_without_touching_tracks(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-ai-plan"
    generate_response = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
            "seed": 404,
            "project_id": project_id,
            "options": {"validate": True},
        },
    )
    assert generate_response.status_code == 200

    project_dir = tmp_path / "api-storage" / "projects" / project_id
    before = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    before_tracks = [track.model_dump(mode="json") for track in before.tracks]

    plan_response = client.post(
        f"/v1/projects/{project_id}/ai/plan",
        json={
            "prompt": (
                "hard bop nocturno en Do menor, 132 bpm, blues menor, "
                "mantener forma y planificar respuestas de metales"
            ),
            "mode": "create_or_patch_plan",
            "locked_tracks": [],
            "locked_sections": [],
        },
    )

    assert plan_response.status_code == 200
    payload = plan_response.json()
    assert payload["status"] == "ok"
    assert payload["planner"] == "fallback_rule_based"
    assert payload["plan_version"].startswith("plan_")
    assert payload["validation"]["status"] == "pass"
    assert {file["kind"] for file in payload["files"]} == {
        "plan_version_json",
        "song_plan_json",
    }

    after = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    assert [track.model_dump(mode="json") for track in after.tracks] == before_tracks
    assert after.metadata["active_plan_version"] == payload["plan_version"]
    assert after.metadata["song_plan"]["song_id"] == project_id
    assert after.metadata["plan_versions"][-1]["plan_version"] == payload["plan_version"]
    assert (project_dir / "plan_versions" / f"{payload['plan_version']}.json").exists()
    assert (project_dir / "song_plan.json").exists()


def test_ai_plan_endpoint_returns_404_for_missing_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    response = client.post(
        "/v1/projects/missing/ai/plan",
        json={"prompt": "hard bop plan"},
    )

    assert response.status_code == 404


def test_ai_plan_endpoint_uses_enabled_local_llm_planner(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    monkeypatch.setenv("AI_MODELS_CONFIG", str(_local_planner_config(tmp_path)))

    monkeypatch.setattr(OllamaPlannerBackend, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaPlannerBackend,
        "generate_plan_json",
        lambda self, **kwargs: json.dumps(_valid_plan_patch()),
    )

    client = TestClient(app)
    project_id, project_dir, before_tracks = _generate_project(client, tmp_path, "api-ai-plan-llm")

    plan_response = client.post(
        f"/v1/projects/{project_id}/ai/plan",
        json={"prompt": "planificar respuesta de metales con energia media", "seed": 406},
    )

    assert plan_response.status_code == 200
    payload = plan_response.json()
    assert payload["planner"] == "llm"
    assert payload["fallback_used"] is False
    assert [attempt["status"] for attempt in payload["attempts"]] == ["pass"]

    after = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    assert [track.model_dump(mode="json") for track in after.tracks] == before_tracks


def test_ai_plan_endpoint_falls_back_when_enabled_local_planner_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    monkeypatch.setenv("AI_MODELS_CONFIG", str(_local_planner_config(tmp_path)))

    def unavailable(self):
        self.unavailable_reason = "Ollama offline"
        return False

    monkeypatch.setattr(OllamaPlannerBackend, "is_available", unavailable)

    client = TestClient(app)
    project_id, _project_dir, _before_tracks = _generate_project(
        client,
        tmp_path,
        "api-ai-plan-llm-fallback",
    )

    plan_response = client.post(
        f"/v1/projects/{project_id}/ai/plan",
        json={"prompt": "hard bop fallback plan", "seed": 407},
    )

    assert plan_response.status_code == 200
    payload = plan_response.json()
    assert payload["planner"] == "fallback_rule_based"
    assert payload["fallback_used"] is True
    assert [attempt["status"] for attempt in payload["attempts"]] == ["pass"]
    assert [attempt["source"] for attempt in payload["attempts"]] == ["fallback_rule_based"]


def _generate_project(client: TestClient, tmp_path, project_id: str):
    generate_response = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
            "seed": 404,
            "project_id": project_id,
            "options": {"validate": True},
        },
    )
    assert generate_response.status_code == 200
    project_dir = tmp_path / "api-storage" / "projects" / project_id
    before = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    return project_id, project_dir, [track.model_dump(mode="json") for track in before.tracks]


def _local_planner_config(tmp_path):
    path = tmp_path / "ai_models.yaml"
    path.write_text(
        """
backends:
  local_llm_planner:
    enabled: true
    type: planner
    adapter: model_backends.planner.ollama_planner_backend.OllamaPlannerBackend
    provider: ollama
    base_url: http://127.0.0.1:11434/api
    model_name: qwen3:8b
    commercial_use: review_required
    dependency_mode: optional
    install_hint: install Ollama
    tasks:
      - plan_song
    capabilities:
      text_prompt: true
      json_planning: true
      commercial_use: review_required
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _valid_plan_patch() -> dict:
    return {
        "style": "hard_bop",
        "substyle": "minor_blues",
        "tempo": 132,
        "meter": "4/4",
        "key": "C minor",
        "form": "minor_blues_12",
        "ensemble": "jazz_sextet",
        "instruments": [
            "drum_kit",
            "double_bass",
            "piano",
            "alto_sax",
            "trumpet_bflat",
            "trombone",
        ],
        "sections": [
            {
                "name": "Head",
                "start_bar": 1,
                "end_bar": 4,
                "energy": 0.55,
                "density_by_role": {"melody": 0.68, "walking_bass": 0.75},
                "groove_feel": "swing",
                "role_focus": ["melody"],
            },
            {
                "name": "Response",
                "start_bar": 5,
                "end_bar": 8,
                "energy": 0.72,
                "density_by_role": {"horn_response": 0.78, "drums": 0.7},
                "groove_feel": "swing",
                "role_focus": ["horn_response"],
            },
            {
                "name": "Turnaround",
                "start_bar": 9,
                "end_bar": 12,
                "energy": 0.82,
                "density_by_role": {"melody": 0.7, "walking_bass": 0.82},
                "groove_feel": "swing",
                "role_focus": ["melody", "horn_response"],
            },
        ],
        "generation_strategy": {
            "mode": "llm_plan",
            "priority_roles": ["melody", "horn_response", "walking_bass"],
            "forbid_audio_models": True,
            "allow_note_generation": False,
            "allow_midi_export": False,
        },
    }
