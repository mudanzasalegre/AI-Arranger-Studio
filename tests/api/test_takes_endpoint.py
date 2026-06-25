from __future__ import annotations

from datetime import UTC, datetime

from app.main import app
from arranger_core import ArrangementProject, TakeManager
from arranger_core.takes.models import ModelArtifactRecord
from fastapi.testclient import TestClient


def test_takes_endpoints_list_accept_reject_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-takes"
    generate_response = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, trio",
            "seed": 308,
            "project_id": project_id,
            "options": {"validate": True},
        },
    )
    assert generate_response.status_code == 200

    list_response = client.get(f"/v1/projects/{project_id}/takes")
    assert list_response.status_code == 200
    assert list_response.json()["active_take_id"] == "take_base"

    project_dir = tmp_path / "api-storage" / "projects" / project_id
    project = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    manager = TakeManager(project_dir)

    rejected_candidate = project.model_copy(deep=True)
    rejected_candidate.metadata["candidate_marker"] = "reject-api"
    rejected_take = manager.create_pending_take(
        base_project=project,
        candidate_project=rejected_candidate,
        artifact_records=[_record("artifact_api_reject", project_id)],
        validation_report=_pass_report(project_id),
        track_id="piano",
        bars=[1],
    )
    reject_response = client.post(f"/v1/projects/{project_id}/takes/{rejected_take.take_id}/reject")
    assert reject_response.status_code == 200
    assert reject_response.json()["take"]["status"] == "rejected"
    assert ArrangementProject.load_json(project_dir / "arrangement_project.json").metadata.get(
        "candidate_marker"
    ) is None

    accepted_candidate = project.model_copy(deep=True)
    accepted_candidate.metadata["candidate_marker"] = "accept-api"
    accepted_take = manager.create_pending_take(
        base_project=project,
        candidate_project=accepted_candidate,
        artifact_records=[_record("artifact_api_accept", project_id)],
        validation_report=_pass_report(project_id),
        track_id="piano",
        bars=[2],
    )
    accept_response = client.post(f"/v1/projects/{project_id}/takes/{accepted_take.take_id}/accept")
    assert accept_response.status_code == 200
    assert accept_response.json()["status"] == "accepted"
    assert ArrangementProject.load_json(project_dir / "arrangement_project.json").metadata[
        "candidate_marker"
    ] == "accept-api"

    restore_response = client.post(f"/v1/projects/{project_id}/takes/take_base/accept")
    assert restore_response.status_code == 200
    restored = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    assert restored.metadata.get("candidate_marker") is None
    assert restored.metadata["active_take_id"] == "take_base"


def _record(artifact_id: str, project_id: str) -> ModelArtifactRecord:
    return ModelArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        backend_id="mock_symbolic",
        task="infill_bars",
        artifact_type="midi",
        raw_path=f"outputs/model_artifacts/raw/{artifact_id}.mid",
        status="validated",
        created_at=datetime.now(UTC).isoformat(),
    )


def _pass_report(project_id: str) -> dict:
    return {
        "status": "pass",
        "project_id": project_id,
        "errors": [],
        "warnings": [],
        "metrics": {"errors": 0, "warnings": 0},
    }
