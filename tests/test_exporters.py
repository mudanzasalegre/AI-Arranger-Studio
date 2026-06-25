
import mido
from arranger_core import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    MeterMark,
    NoteEvent,
    RestEvent,
    Section,
    TempoMark,
    Track,
    export_project,
)
from music21 import converter


def test_export_project_writes_manifest_midi_and_musicxml(tmp_path):
    project = _exportable_project()

    manifest = export_project(project, tmp_path, include_pdf=False)

    assert manifest["status"] == "exported"
    assert (tmp_path / "export_manifest.json").exists()
    assert (tmp_path / "arrangement_project.json").exists()
    assert (tmp_path / "generation_spec.json").exists()
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "full_arrangement.mid").stat().st_size > 0
    assert (tmp_path / "full_score.musicxml").stat().st_size > 0
    assert {file["kind"] for file in manifest["files"]} >= {
        "midi_full",
        "midi_track",
        "musicxml_full",
    }


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
