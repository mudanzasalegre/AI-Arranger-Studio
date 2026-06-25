from __future__ import annotations

import pytest
from arranger_core import (
    ArtifactImporter,
    ArtifactStore,
    GenerationSpec,
    ProjectMerger,
    TakeManager,
    ValidationGate,
    generate_arrangement,
)
from arranger_core.ai.artifact_importer import ArtifactImportError
from model_backends import MockSymbolicBackend, ModelGenerationRequest


def test_valid_mock_artifact_creates_pending_take_without_changing_active_project(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=301),
        project_id="artifact-valid",
    )
    project_dir = tmp_path / "project"
    project.save_json(project_dir / "arrangement_project.json")
    active_before = (project_dir / "arrangement_project.json").read_text(encoding="utf-8")

    backend = MockSymbolicBackend(output_dir=tmp_path / "backend_raw")
    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="valid-artifact",
            track_id="alto_sax",
            bars=[1],
            seed=301,
            song_plan=project.metadata["song_plan"],
            groove_map=project.metadata["song_plan"]["groove_map"],
        )
    )

    artifact_store = ArtifactStore(tmp_path / "model_artifacts")
    records = artifact_store.store_generation_result(result, project_id=project.project_id)
    importer = ArtifactImporter(
        artifact_store=artifact_store,
        imported_root=tmp_path / "model_artifacts" / "imported",
    )
    imported = importer.import_record(
        records[0],
        project=project,
        target_track_id="alto_sax",
        target_bars=[1],
    )
    candidate = ProjectMerger().merge(
        project,
        imported,
        target_track_id="alto_sax",
        target_bars=[1],
    )
    report = ValidationGate().validate_candidate(
        base_project=project,
        candidate_project=candidate,
        target_track_id="alto_sax",
        target_bars=[1],
    )
    assert report["status"] in {"pass", "pass_with_warnings"}

    artifact_store.mark_validated(records[0], metadata={"validation": report})
    take = TakeManager(project_dir).create_pending_take(
        base_project=project,
        candidate_project=candidate,
        artifact_records=[artifact_store.get(records[0].artifact_id)],
        validation_report=report,
        track_id="alto_sax",
        bars=[1],
        instruction="mock infill",
        seed=301,
    )

    assert take.status == "pending"
    assert take.artifact_ids == [records[0].artifact_id]
    assert project.to_json() == project.model_copy().to_json()
    assert (project_dir / "arrangement_project.json").read_text(encoding="utf-8") == active_before
    assert artifact_store.get(records[0].artifact_id).status == "validated"

    takes = TakeManager(project_dir).list_takes(project=project)
    assert takes["active_take_id"] == "take_base"
    assert {item["status"] for item in takes["takes"]} == {"accepted", "pending"}


def test_invalid_mock_artifact_is_marked_rejected(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=302),
        project_id="artifact-invalid",
    )
    backend = MockSymbolicBackend(output_dir=tmp_path / "backend_raw")
    result = backend.generate(
        ModelGenerationRequest(
            task="generate_track",
            request_id="invalid-artifact",
            track_id="alto_sax",
            bars=[1],
            seed=302,
            metadata={"mock_artifact": "invalid_midi"},
        )
    )

    artifact_store = ArtifactStore(tmp_path / "model_artifacts")
    record = artifact_store.store_generation_result(result, project_id=project.project_id)[0]
    importer = ArtifactImporter(artifact_store=artifact_store)

    with pytest.raises(ArtifactImportError):
        importer.import_record(
            record,
            project=project,
            target_track_id="alto_sax",
            target_bars=[1],
        )

    rejected = artifact_store.get(record.artifact_id)
    assert rejected.status == "rejected"
    assert rejected.rejected_path is not None
    assert "MIDI artifact is not parseable" in rejected.metadata["rejection_reason"]
