from __future__ import annotations

from arranger_core import (
    ArtifactImporter,
    ArtifactStore,
    GenerationSpec,
    NoteEvent,
    ProjectMerger,
    ValidationGate,
    generate_arrangement,
)
from model_backends import MockSymbolicBackend, ModelGenerationRequest


def test_validation_gate_rejects_changes_outside_target_bars(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=304),
        project_id="gate-outside-target",
    )
    candidate = project.model_copy(deep=True)
    alto = next(track for track in candidate.tracks if track.id == "alto_sax")
    alto.bars[1].events.append(NoteEvent(pitch="C4", start=0.0, duration=0.5))

    report = ValidationGate().validate_candidate(
        base_project=project,
        candidate_project=candidate,
        target_track_id="alto_sax",
        target_bars=[1],
    )

    assert report["status"] == "fail"
    assert _has_gate_error(report, "outside_target_bars_modified")


def test_validation_gate_rejects_non_target_track_removal(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=309),
        project_id="gate-non-target-removed",
    )
    candidate = project.model_copy(deep=True)
    candidate.tracks = [track for track in candidate.tracks if track.id != "piano"]

    report = ValidationGate().validate_candidate(
        base_project=project,
        candidate_project=candidate,
        target_track_id="alto_sax",
        target_bars=[1],
    )

    assert report["status"] == "fail"
    assert _has_gate_error(report, "non_target_track_modified")


def test_validation_gate_accepts_project_merger_candidate(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=305),
        project_id="gate-valid",
    )
    backend = MockSymbolicBackend(output_dir=tmp_path / "backend_raw")
    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="gate-valid",
            track_id="alto_sax",
            bars=[1],
            seed=305,
        )
    )
    store = ArtifactStore(tmp_path / "model_artifacts")
    record = store.store_generation_result(result, project_id=project.project_id)[0]
    imported = ArtifactImporter(artifact_store=store).import_record(
        record,
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
        locked_tracks=["double_bass", "piano"],
    )

    assert report["status"] in {"pass", "pass_with_warnings"}
    assert not report["errors"]


def _has_gate_error(report, code):
    return any(issue["code"] == code for issue in report["errors"])
