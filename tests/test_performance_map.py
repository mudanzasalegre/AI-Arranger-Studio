from __future__ import annotations

import mido
from arranger_core import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    ImportedModelArtifact,
    MeterMark,
    NoteEvent,
    PerformanceMapper,
    ProjectMerger,
    RestEvent,
    Section,
    TempoMark,
    Track,
    generate_arrangement,
    write_full_midi,
)


def test_rule_based_arrangement_gets_performance_map_and_sources():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=701),
        project_id="performance-rule-based",
    )

    assert project.metadata["performance_applied"] is True
    assert project.metadata["performance_map"]["swing_ratio"] == project.metadata[
        "song_plan"
    ]["groove_map"]["swing_ratio"]
    assert project.metadata["performance_map"]["performance_sources"] == ["rule_based"]
    assert project.validate_bar_durations() == []
    assert {
        event.annotations["performance_source"]
        for track in project.tracks
        for event in _notes(track)
    } == {"rule_based"}

    applied_again = PerformanceMapper().apply(project, seed=701)
    assert applied_again.model_dump(mode="json") == project.model_dump(mode="json")


def test_imported_model_material_is_normalized_without_double_humanizing():
    base = generate_arrangement(
        GenerationSpec(ensemble="jazz_quartet_alto", form="minor_blues_12", seed=702),
        project_id="performance-imported",
    )
    imported_track = Track(
        id="alto_sax",
        instrument="alto_sax",
        role="melody",
        bars=[
            Bar(
                number=1,
                events=[
                    RestEvent(start=0.0, duration=0.5),
                    NoteEvent(
                        pitch="C4",
                        start=0.5,
                        duration=1.0,
                        velocity=127,
                        annotations={
                            "source": "model_artifact",
                            "humanized_timing_ms": 22,
                        },
                    ),
                    RestEvent(start=1.5, duration=0.5),
                    NoteEvent(
                        pitch="Eb4",
                        start=2.0,
                        duration=1.0,
                        velocity=5,
                        annotations={"source": "model_artifact"},
                    ),
                    RestEvent(start=3.0, duration=1.0),
                ],
            )
        ],
    )
    imported = ImportedModelArtifact(
        artifact_id="artifact_perf",
        project_id=base.project_id,
        backend_id="mock_symbolic",
        task="infill_bars",
        artifact_type="midi",
        track_id="alto_sax",
        bars=[1],
        track=imported_track,
    )

    candidate = ProjectMerger().merge(
        base,
        imported,
        target_track_id="alto_sax",
        target_bars=[1],
    )
    alto = next(track for track in candidate.tracks if track.id == "alto_sax")
    imported_notes = [event for event in _notes(alto) if event.annotations.get("source")]

    assert imported_notes[0].annotations["performance_source"] == "normalized_model"
    assert imported_notes[0].annotations["performance_timing_status"] == "preserved_imported"
    assert imported_notes[0].annotations["performance_microtiming_ms"] == 22
    assert imported_notes[0].velocity <= 98
    assert imported_notes[1].annotations["performance_source"] == "normalized_model"
    assert imported_notes[1].velocity >= 62
    assert candidate.validate_bar_durations() == []


def test_midi_export_applies_performance_microtiming_without_changing_notation(tmp_path):
    project = _one_note_project()
    write_full_midi(project, tmp_path / "timed.mid")

    midi = mido.MidiFile(tmp_path / "timed.mid")
    absolute_note_on_ticks: list[int] = []
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                absolute_note_on_ticks.append(absolute)

    assert project.tracks[0].bars[0].events[0].start == 1.0
    assert absolute_note_on_ticks == [528]


def _one_note_project() -> ArrangementProject:
    return ArrangementProject(
        project_id="performance-midi",
        generation_spec=GenerationSpec(
            prompt="performance midi",
            tempo=120,
            meter="4/4",
            instruments=["piano"],
            seed=703,
        ),
        tempo_map=[TempoMark(bar=1, bpm=120)],
        meter_map=[MeterMark(bar=1, meter="4/4")],
        form=[Section(name="A", start_bar=1, end_bar=1)],
        chord_grid=[ChordSymbol(symbol="Cmaj7", bar=1, beat=1)],
        tracks=[
            Track(
                id="piano",
                instrument="piano",
                role="comping",
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(
                                pitch="C4",
                                start=1.0,
                                duration=1.0,
                                velocity=70,
                                annotations={"performance_microtiming_ms": 50},
                            )
                        ],
                    )
                ],
            )
        ],
    )


def _notes(track: Track) -> list[NoteEvent]:
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
