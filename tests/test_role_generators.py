import mido
from arranger_core import (
    ChordParser,
    GenerationSpec,
    NoteEvent,
    export_project,
    generate_arrangement,
    note_to_midi,
)
from music21 import converter


def test_jazz_trio_arrangement_has_rhythm_section_melody_and_valid_bars():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=71),
        project_id="trio-arrangement",
    )

    assert [track.id for track in project.tracks] == ["drum_kit", "double_bass", "piano"]
    assert project.validate_bar_durations() == []
    assert project.metadata["arranger"] == "rule_based_v0"
    assert project.tracks[2].metadata["contains_melody"] is True
    assert any(
        event.annotations.get("melodic_role")
        for event in _notes(project.tracks[2])
    )


def test_walking_bass_hits_roots_and_piano_uses_rootless_voicings():
    parser = ChordParser.load_default()
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=72),
        project_id="bass-piano-check",
    )
    bass = _track(project, "double_bass")
    piano = _track(project, "piano")

    root_hits = 0
    for bar in bass.bars:
        first_note = next(event for event in bar.events if isinstance(event, NoteEvent))
        chord = parser.parse(first_note.annotations["source_chord"])
        if note_to_midi(first_note.pitch) % 12 == chord.root_pc:
            root_hits += 1
    assert root_hits == project.bar_count

    rootless_notes = [
        event
        for event in _notes(piano)
        if event.annotations.get("voicing") == "rootless"
    ]
    assert rootless_notes
    assert all(
        note_to_midi(event.pitch) % 12 != event.annotations["root_pc"]
        for event in rootless_notes
    )


def test_drums_generate_swing_groove_fills_and_channel_10(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=73),
        project_id="drum-check",
    )
    drums = _track(project, "drum_kit")

    assert any(bar.metadata["fill"] for bar in drums.bars)
    assert any(
        event.annotations.get("drum") == "ride"
        for event in _notes(drums)
    )

    export_project(project, tmp_path, include_pdf=False)
    midi = mido.MidiFile(tmp_path / "full_arrangement.mid")
    drum_channels = {
        message.channel
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0 and message.note in {36, 38, 51}
    }
    assert 9 in drum_channels
    assert (tmp_path / "midi_tracks/drums.mid").exists()


def test_quartet_exports_melody_horn_and_midi_tracks(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_quartet_alto", form="aaba_32", seed=74),
        project_id="quartet-arrangement",
    )

    assert [track.id for track in project.tracks] == [
        "drum_kit",
        "double_bass",
        "piano",
        "alto_sax",
    ]
    assert _track(project, "alto_sax").role == "melody"
    assert project.validate_bar_durations() == []

    export_project(project, tmp_path, include_pdf=False)
    converter.parse(tmp_path / "full_score.musicxml")
    assert (tmp_path / "midi_tracks/alto_sax.mid").exists()
    assert (tmp_path / "midi_tracks/piano.mid").exists()


def test_sextet_generates_horn_responses_and_shout_chorus():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=75),
        project_id="sextet-arrangement",
    )

    trumpet = _track(project, "trumpet_bflat")
    trombone = _track(project, "trombone")
    alto = _track(project, "alto_sax")

    assert trumpet.role == "horn_response"
    assert trombone.role == "horn_response"
    assert any(
        event.annotations.get("horn_role") == "response"
        for event in _notes(trumpet)
    )
    assert any(
        event.annotations.get("horn_role") == "shout_chorus"
        for event in _notes(trombone)
    )
    assert any(bar.metadata.get("shout_chorus") for bar in alto.bars)
    assert project.validate_bar_durations() == []


def test_humanizer_is_seed_reproducible():
    spec = GenerationSpec(ensemble="jazz_quintet", form="minor_blues_12", seed=76)

    first = generate_arrangement(spec, project_id="humanized")
    second = generate_arrangement(spec, project_id="humanized")
    different_seed = generate_arrangement(
        spec.model_copy(update={"seed": 77}),
        project_id="humanized",
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.model_dump(mode="json") != different_seed.model_dump(mode="json")
    assert any(
        "humanized_timing_ms" in event.annotations
        for track in first.tracks
        for event in _notes(track)
    )


def _track(project, track_id):
    return next(track for track in project.tracks if track.id == track_id)


def _notes(track):
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
