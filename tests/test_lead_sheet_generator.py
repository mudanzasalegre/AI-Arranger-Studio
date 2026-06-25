from arranger_core import (
    GenerationSpec,
    NoteEvent,
    RestEvent,
    export_project,
    generate_lead_sheet_project,
    note_to_midi,
)
from music21 import converter


def test_minor_blues_lead_sheet_has_motif_phrases_breaths_and_range():
    spec = GenerationSpec(
        key="C minor",
        form="minor_blues_12",
        complexity=0.4,
        seed=17,
        constraints={
            "lead_instrument": "alto_sax",
            "melody_range": {"low": "C4", "high": "Bb5"},
        },
    )

    project = generate_lead_sheet_project(spec, project_id="minor-blues-lead")
    track = project.tracks[0]
    notes = _notes(track)
    rests = _rests(track)

    assert project.bar_count == 12
    assert track.role == "melody"
    assert track.name == "Lead Sheet"
    assert len(track.bars) == 12
    assert project.metadata["phrase_length_bars"] == 4
    assert project.metadata["initial_motif"]["roles"] == [
        "root",
        "third",
        "approach_upper",
        "seventh",
    ]
    assert project.validate_bar_durations() == []
    assert notes
    assert rests
    assert any(rest.annotations.get("breath") for rest in rests)
    assert any(note.articulations for note in notes)
    assert all(
        note_to_midi("C4") <= note_to_midi(note.pitch) <= note_to_midi("Bb5")
        for note in notes
    )


def test_aaba_lead_sheet_generates_32_bars_and_respects_configured_range():
    spec = GenerationSpec(
        key="F major",
        form="aaba_32",
        complexity=0.45,
        seed=42,
        constraints={"melody_range": "D4-A4"},
    )

    first = generate_lead_sheet_project(spec, project_id="aaba-lead")
    second = generate_lead_sheet_project(spec, project_id="aaba-lead")
    track = first.tracks[0]
    notes = _notes(track)

    assert first.bar_count == 32
    assert len(track.bars) == 32
    assert first.validate_bar_durations() == []
    assert [note.model_dump(mode="json") for note in notes] == [
        note.model_dump(mode="json") for note in _notes(second.tracks[0])
    ]
    assert all(
        note_to_midi("D4") <= note_to_midi(note.pitch) <= note_to_midi("A4")
        for note in notes
    )


def test_lead_sheet_chords_articulations_and_dynamics_export_to_musicxml(tmp_path):
    spec = GenerationSpec(
        key="C minor",
        form="minor_blues_12",
        complexity=0.55,
        seed=29,
        constraints={"melody_range": ["C4", "C5"]},
    )
    project = generate_lead_sheet_project(spec, project_id="lead-sheet-export")

    export_project(project, tmp_path, include_pdf=False)

    musicxml_path = tmp_path / "full_score.musicxml"
    converter.parse(musicxml_path)
    xml_text = musicxml_path.read_text(encoding="utf-8")

    assert "<harmony" in xml_text
    assert "<part-name>Lead Sheet</part-name>" in xml_text
    assert "minor-seventh" in xml_text
    assert "<articulations>" in xml_text
    assert "<dynamics" in xml_text


def _notes(track):
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]


def _rests(track):
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, RestEvent)
    ]
