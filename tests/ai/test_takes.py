from __future__ import annotations

from datetime import UTC, datetime

import pytest
from arranger_core import ArrangementProject, GenerationSpec, TakeManager, generate_arrangement
from arranger_core.takes.models import ModelArtifactRecord


def test_take_manager_accept_reject_and_restore_previous_take(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=306),
        project_id="takes-manager",
    )
    project_dir = tmp_path / "project"
    project.save_json(project_dir / "arrangement_project.json")
    manager = TakeManager(project_dir)

    rejected_candidate = project.model_copy(deep=True)
    rejected_candidate.metadata["candidate_marker"] = "reject-me"
    rejected_take = manager.create_pending_take(
        base_project=project,
        candidate_project=rejected_candidate,
        artifact_records=[_record("artifact_reject", project.project_id)],
        validation_report=_pass_report(project.project_id),
        track_id="piano",
        bars=[1],
    )
    rejected = manager.reject_take(rejected_take.take_id, reason="not useful")
    assert rejected.status == "rejected"
    assert ArrangementProject.load_json(project_dir / "arrangement_project.json").metadata.get(
        "candidate_marker"
    ) is None

    accepted_candidate = project.model_copy(deep=True)
    accepted_candidate.metadata["candidate_marker"] = "accept-me"
    accepted_take = manager.create_pending_take(
        base_project=project,
        candidate_project=accepted_candidate,
        artifact_records=[_record("artifact_accept", project.project_id)],
        validation_report=_pass_report(project.project_id),
        track_id="piano",
        bars=[2],
    )
    accepted, active_project = manager.accept_take(accepted_take.take_id)

    assert accepted.status == "accepted"
    assert active_project.metadata["candidate_marker"] == "accept-me"
    assert ArrangementProject.load_json(project_dir / "arrangement_project.json").metadata[
        "active_take_id"
    ] == accepted_take.take_id

    restored, restored_project = manager.accept_take("take_base")
    assert restored.status == "accepted"
    assert restored_project.metadata.get("candidate_marker") is None
    assert ArrangementProject.load_json(project_dir / "arrangement_project.json").metadata[
        "active_take_id"
    ] == "take_base"


def test_take_manager_rejects_active_take(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=307),
        project_id="takes-active",
    )
    manager = TakeManager(tmp_path / "project")
    manager.ensure_base_take(project)

    with pytest.raises(ValueError):
        manager.reject_take("take_base")


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
