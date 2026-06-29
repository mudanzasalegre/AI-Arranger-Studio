
import json
from datetime import UTC, datetime

import mido
import pytest
from arranger_core import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    MeterMark,
    MusicValidationError,
    NoteEvent,
    RestEvent,
    Section,
    TakeManager,
    TempoMark,
    Track,
    export_project,
)
from arranger_core.takes.models import ModelArtifactRecord
from music21 import converter


def test_export_project_writes_manifest_midi_and_musicxml(tmp_path):
    project = _exportable_project()

    manifest = export_project(project, tmp_path, include_pdf=False)

    assert manifest["status"] == "exported"
    assert (tmp_path / "export_manifest.json").exists()
    assert (tmp_path / "arrangement_project.json").exists()
    assert (tmp_path / "generation_spec.json").exists()
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "takes_manifest.json").exists()
    assert (tmp_path / "model_trace.json").exists()
    assert (tmp_path / "session_readme.md").exists()
    assert (tmp_path / "full_arrangement.mid").stat().st_size > 0
    assert (tmp_path / "full_score.musicxml").stat().st_size > 0
    assert {file["kind"] for file in manifest["files"]} >= {
        "midi_full",
        "midi_track",
        "musicxml_full",
        "model_trace_json",
        "session_readme",
        "takes_manifest_json",
    }
    trace = json.loads((tmp_path / "model_trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "no_model_artifacts"
    assert trace["model_artifacts"] == []


def test_daw_export_traces_only_active_accepted_model_take(tmp_path):
    project = _exportable_project()
    manager = TakeManager(tmp_path)
    manager.ensure_base_take(project)
    candidate = project.model_copy(deep=True)
    accepted_take = manager.create_pending_take(
        base_project=project,
        candidate_project=candidate,
        artifact_records=[_artifact_record("artifact_active", project.project_id)],
        validation_report=_pass_report(project.project_id),
        track_id="piano",
        bars=[1],
        instruction="syncopated comping answer",
        seed=4201,
        metadata={
            "model_trace": {
                "backend": "mock_symbolic",
                "task": "infill_bars",
                "track_id": "piano",
                "bars": [1],
                "instruction": "syncopated comping answer",
                "seed": 4201,
                "validation_status": "pass",
                "commercial_use": "allowed",
                "context_midi_path": "internal/context.mid",
            }
        },
    )
    _take, accepted_project = manager.accept_take(accepted_take.take_id)
    pending_take = manager.create_pending_take(
        base_project=accepted_project,
        candidate_project=accepted_project.model_copy(deep=True),
        artifact_records=[_artifact_record("artifact_pending", project.project_id)],
        validation_report=_pass_report(project.project_id),
        track_id="piano",
        bars=[2],
        instruction="pending comping answer",
        seed=4202,
    )

    with pytest.raises(MusicValidationError) as exc_info:
        export_project(accepted_project, tmp_path, include_pdf=False)
    assert _has_error(exc_info.value.report, "pending_take_present")

    manager.reject_take(pending_take.take_id, reason="superseded before final export")
    export_project(accepted_project, tmp_path, include_pdf=False)

    takes_manifest = json.loads((tmp_path / "takes_manifest.json").read_text(encoding="utf-8"))
    take_ids = {take["take_id"] for take in takes_manifest["takes"]}
    assert accepted_take.take_id in take_ids
    assert pending_take.take_id not in take_ids
    assert {take["status"] for take in takes_manifest["takes"]} == {"accepted"}
    assert all("project_snapshot_path" not in take for take in takes_manifest["takes"])

    trace = json.loads((tmp_path / "model_trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "traced"
    assert trace["active_take_id"] == accepted_take.take_id
    assert trace["model_artifacts"] == [
        {
            "take_id": accepted_take.take_id,
            "status": "accepted",
            "backend_id": "mock_symbolic",
            "task": "infill_bars",
            "prompt": "export test",
            "instruction": "syncopated comping answer",
            "track_id": "piano",
            "bars": [1],
            "seed": 4201,
            "validation_result": "pass",
            "commercial_use": "allowed",
        }
    ]


def test_commercial_release_blocks_review_required_model_trace(tmp_path):
    project = _exportable_project()
    manager = TakeManager(tmp_path)
    manager.ensure_base_take(project)
    candidate = project.model_copy(deep=True)
    take = manager.create_pending_take(
        base_project=project,
        candidate_project=candidate,
        artifact_records=[_artifact_record("artifact_review", project.project_id)],
        validation_report=_pass_report(project.project_id),
        track_id="piano",
        bars=[1],
        instruction="review required model",
        seed=4210,
        metadata={
            "model_trace": {
                "backend": "midigpt",
                "task": "infill_bars",
                "track_id": "piano",
                "bars": [1],
                "instruction": "review required model",
                "seed": 4210,
                "validation_status": "pass",
                "commercial_use": "review_required",
            }
        },
    )
    _take, accepted_project = manager.accept_take(take.take_id)

    with pytest.raises(MusicValidationError) as exc_info:
        export_project(
            accepted_project,
            tmp_path,
            include_pdf=False,
            export_mode="commercial",
        )

    assert _has_error(exc_info.value.report, "model_license_incompatible")


def test_release_gate_validates_declared_dataset_manifest(tmp_path):
    manifest_path = tmp_path / "unsafe_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "entries": [
                    {
                        "path": "unsafe.mid",
                        "source": "unknown",
                        "license": "unknown",
                        "license_confidence": "low",
                        "commercial_training": "review_required",
                        "local_learning_only": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    project = _exportable_project()
    project.metadata["dataset_manifest_paths"] = [str(manifest_path)]

    with pytest.raises(MusicValidationError) as exc_info:
        export_project(project, tmp_path, include_pdf=False)

    assert _has_error(exc_info.value.report, "dataset_license_blocked")


def test_full_midi_contains_separate_named_tracks_and_drum_channel(tmp_path):
    export_project(_exportable_project(), tmp_path, include_pdf=False)

    midi = mido.MidiFile(tmp_path / "full_arrangement.mid")
    names = [
        message.name
        for track in midi.tracks
        for message in track
        if message.type == "track_name"
    ]
    assert names == ["Conductor", "Drum Kit", "Double Bass", "Piano"]

    drum_track = midi.tracks[1]
    drum_note_channels = {
        message.channel
        for message in drum_track
        if message.type == "note_on" and message.velocity > 0
    }
    assert drum_note_channels == {9}


def test_midi_track_exports_are_individual_files(tmp_path):
    export_project(_exportable_project(), tmp_path, include_pdf=False)

    expected_files = [
        tmp_path / "midi_tracks/drums.mid",
        tmp_path / "midi_tracks/double_bass.mid",
        tmp_path / "midi_tracks/piano.mid",
    ]
    for path in expected_files:
        assert path.exists()
        midi = mido.MidiFile(path)
        musical_tracks = [
            track
            for track in midi.tracks
            if any(message.type == "note_on" and message.velocity > 0 for message in track)
        ]
        assert len(musical_tracks) == 1


def test_musicxml_full_score_is_parseable_and_contains_harmony(tmp_path):
    export_project(_exportable_project(), tmp_path, include_pdf=False)

    musicxml_path = tmp_path / "full_score.musicxml"
    parsed = converter.parse(musicxml_path)
    xml_text = musicxml_path.read_text(encoding="utf-8")

    assert len(parsed.parts) == 3
    assert "<harmony" in xml_text
    assert "Head In" in xml_text


def _exportable_project() -> ArrangementProject:
    return ArrangementProject(
        project_id="export-test",
        metadata={"title": "Export Test"},
        generation_spec=GenerationSpec(
            prompt="export test",
            style="hard_bop",
            key="C minor",
            meter="4/4",
            tempo=132,
            form="minor_blues_12",
            ensemble="jazz_trio",
            duration_bars=2,
            instruments=["drum_kit", "double_bass", "piano"],
            seed=7,
        ),
        tempo_map=[TempoMark(bar=1, bpm=132)],
        meter_map=[MeterMark(bar=1, meter="4/4")],
        form=[Section(name="Head In", start_bar=1, end_bar=2)],
        chord_grid=[
            ChordSymbol(symbol="Cm7", bar=1, beat=1),
            ChordSymbol(symbol="G7alt", bar=2, beat=1),
        ],
        tracks=[
            Track(
                id="drum_kit",
                instrument="drum_kit",
                role="drums",
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(pitch="C2", start=0, duration=1),
                            NoteEvent(pitch="D2", start=1, duration=1),
                            NoteEvent(pitch="C2", start=2, duration=1),
                            NoteEvent(pitch="D2", start=3, duration=1),
                        ],
                    ),
                    Bar(number=2, events=[RestEvent(start=0, duration=4)]),
                ],
            ),
            Track(
                id="double_bass",
                instrument="double_bass",
                role="walking_bass",
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(pitch="C2", start=0, duration=1),
                            NoteEvent(pitch="Eb2", start=1, duration=1),
                            NoteEvent(pitch="G2", start=2, duration=1),
                            NoteEvent(pitch="Bb2", start=3, duration=1),
                        ],
                    ),
                    Bar(
                        number=2,
                        events=[
                            NoteEvent(pitch="G2", start=0, duration=1),
                            NoteEvent(pitch="B2", start=1, duration=1),
                            NoteEvent(pitch="D3", start=2, duration=1),
                            NoteEvent(pitch="F3", start=3, duration=1),
                        ],
                    ),
                ],
            ),
            Track(
                id="piano",
                instrument="piano",
                role="comping",
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(pitch="Eb4", start=0, duration=4),
                            NoteEvent(pitch="Bb4", start=0, duration=4),
                        ],
                    ),
                    Bar(number=2, events=[RestEvent(start=0, duration=4)]),
                ],
            ),
        ],
    )


def _artifact_record(artifact_id: str, project_id: str) -> ModelArtifactRecord:
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


def _has_error(report: dict, code: str) -> bool:
    return any(issue.get("code") == code for issue in report.get("errors", []))
