from pathlib import Path

import pytest
from arranger_core import (
    ArrangementProject,
    Bar,
    BarDurationValidationError,
    ChordSymbol,
    GenerationSpec,
    KeyMark,
    MeterMark,
    NoteEvent,
    RestEvent,
    Section,
    TempoMark,
    Track,
    load_project_json,
    meter_to_quarter_beats,
)
from pydantic import ValidationError


def test_create_empty_arrangement_project():
    project = ArrangementProject(project_id="empty-project")

    assert project.schema_version == "0.1.0"
    assert project.project_id == "empty-project"
    assert project.tracks == []
    assert project.bar_count == 0
    assert project.validate_bar_durations() == []
    project.assert_bar_durations_valid()


def test_create_four_bar_two_track_project_and_validate_durations():
    project = _four_bar_project()

    assert project.bar_count == 4
    assert [track.id for track in project.tracks] == ["double_bass", "piano"]
    assert project.validate_bar_durations() == []
    project.assert_bar_durations_valid()


def test_project_json_round_trip_without_critical_loss(tmp_path):
    project = _four_bar_project()

    loaded_from_text = ArrangementProject.from_json(project.to_json())
    assert loaded_from_text.model_dump(mode="json") == project.model_dump(mode="json")

    path = project.save_json(tmp_path / "arrangement_project.json")
    loaded_from_disk = load_project_json(path)
    assert loaded_from_disk.model_dump(mode="json") == project.model_dump(mode="json")


def test_existing_example_project_json_loads():
    root = Path(__file__).resolve().parents[1]
    project = ArrangementProject.load_json(
        root / "examples/projects/arrangement_project.example.json"
    )

    assert project.project_id == "demo-hard-bop-001"
    assert project.form[0].name == "Head In"
    assert project.chord_grid[0].symbol == "Cm7"
    assert [track.id for track in project.tracks] == ["double_bass", "piano"]


def test_bar_duration_validation_reports_gaps_and_overflow():
    project = ArrangementProject(
        project_id="broken-bars",
        tracks=[
            Track(
                id="bass",
                instrument="double_bass",
                role="walking_bass",
                bars=[
                    Bar(number=1, events=[NoteEvent(pitch="C2", start=0, duration=2)]),
                    Bar(number=2, events=[RestEvent(start=0, duration=5)]),
                ],
            )
        ],
    )

    issues = project.validate_bar_durations()

    assert len(issues) == 2
    assert "gap" in issues[0].message
    assert "exceeds" in issues[1].message
    with pytest.raises(BarDurationValidationError) as exc_info:
        project.assert_bar_durations_valid()
    assert len(exc_info.value.issues) == 2


def test_meter_to_quarter_beats_supports_common_meters():
    assert meter_to_quarter_beats("4/4") == 4
    assert meter_to_quarter_beats("3/4") == 3
    assert meter_to_quarter_beats("6/8") == 3


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(ValidationError):
        ArrangementProject(project_id="future", schema_version="99.0.0")


def _four_bar_project() -> ArrangementProject:
    return ArrangementProject(
        project_id="obj1-four-bar-project",
        metadata={"title": "Objective 1 Round Trip"},
        generation_spec=GenerationSpec(
            prompt="hard bop minor blues quartet",
            style="hard_bop",
            key="C minor",
            meter="4/4",
            tempo=132,
            form="minor_blues_12",
            ensemble="jazz_quartet",
            duration_bars=4,
            instruments=["double_bass", "piano"],
            seed=42,
        ),
        tempo_map=[TempoMark(bar=1, bpm=132)],
        key_map=[KeyMark(bar=1, key="C minor")],
        meter_map=[MeterMark(bar=1, meter="4/4")],
        form=[Section(name="Head In", start_bar=1, end_bar=4)],
        chord_grid=[
            ChordSymbol(symbol="Cm7", bar=1, beat=1),
            ChordSymbol(symbol="Fm7", bar=2, beat=1),
            ChordSymbol(symbol="Cm7", bar=3, beat=1),
            ChordSymbol(symbol="G7alt", bar=4, beat=1),
        ],
        tracks=[
            Track(
                id="double_bass",
                instrument="double_bass",
                role="walking_bass",
                bars=[
                    Bar(
                        number=bar_number,
                        events=[
                            NoteEvent(pitch="C2", start=0, duration=1, velocity=76),
                            NoteEvent(pitch="G2", start=1, duration=1, velocity=72),
                            NoteEvent(pitch="Bb2", start=2, duration=1, velocity=74),
                            NoteEvent(pitch="B2", start=3, duration=1, velocity=70),
                        ],
                    )
                    for bar_number in range(1, 5)
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
                            NoteEvent(pitch="Eb4", start=0, duration=4, velocity=64),
                            NoteEvent(pitch="Bb4", start=0, duration=4, velocity=62),
                        ],
                    ),
                    Bar(number=2, events=[RestEvent(start=0, duration=4)]),
                    Bar(number=3, events=[RestEvent(start=0, duration=4)]),
                    Bar(number=4, events=[RestEvent(start=0, duration=4)]),
                ],
            ),
        ],
    )
