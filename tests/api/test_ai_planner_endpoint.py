from __future__ import annotations

from app.main import app
from arranger_core import ArrangementProject
from fastapi.testclient import TestClient


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
