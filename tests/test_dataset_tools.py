from __future__ import annotations

import json
import shutil
from pathlib import Path

import mido
from arranger_core import GenerationSpec, NoteEvent, export_project, generate_arrangement
from dataset_tools import PatternIndex, create_manifest, import_dataset
from music21 import harmony, meter, note, stream

TICKS_PER_BEAT = 480
HALF_BEATS = tuple(index * 0.5 for index in range(8))


def test_dataset_import_extracts_patterns_and_respects_usage_flags(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_test_midi_set(source_dir)
    shutil.copy2(source_dir / "learn_09.mid", source_dir / "learn_09_copy.mid")

    manifest_path = tmp_path / "dataset_manifest.json"
    create_manifest(
        source_dir,
        manifest_path,
        default_metadata=_default_metadata(),
        metadata_by_name={
            "learn_00.mid": {"usable_for_pattern_extraction": False},
            "learn_01.mid": {"quality": 2},
            "learn_08.mid": {"usable_for_training": False},
        },
    )

    summary = import_dataset(source_dir, manifest_path, tmp_path / "imported")
    index = PatternIndex.load_json(summary.pattern_index_path)
    normalized_files = json.loads(Path(summary.normalized_files_path).read_text())

    assert summary.imported_files == 10
    assert summary.duplicate_files == 1
    assert summary.skipped_for_license == 1
    assert summary.skipped_for_quality == 1
    assert summary.extracted_patterns > 0
    assert any(item["duplicate_of"] for item in normalized_files)
    assert all(Path(item["normalized_path"]).exists() for item in normalized_files)

    categories = {pattern.category for pattern in index.patterns}
    assert {
        "drum_grooves",
        "walking_bass_cells",
        "piano_voicings",
        "melodic_motifs",
        "horn_responses",
    } <= categories
    assert index.search(category="walking_bass_cells", role="walking_bass", min_quality=3)
    assert index.search(category="drum_grooves", usable_for_pattern_extraction=True)
    assert not index.search(category="drum_grooves", usable_for_pattern_extraction=False)

    skipped_sources = ("learn_00.mid", "learn_01.mid")
    assert not any(pattern.source_path.endswith(skipped_sources) for pattern in index.patterns)

    training_disabled = [
        pattern for pattern in index.patterns if pattern.source_path.endswith("learn_08.mid")
    ]
    assert training_disabled
    assert all(not pattern.usable_for_training for pattern in training_disabled)
    assert all(pattern.usable_for_pattern_extraction for pattern in index.patterns)


def test_musicxml_progressions_are_indexed(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_progression_musicxml(source_dir / "progression.musicxml")

    manifest_path = tmp_path / "dataset_manifest.json"
    create_manifest(source_dir, manifest_path, default_metadata=_default_metadata())

    summary = import_dataset(source_dir, manifest_path, tmp_path / "imported")
    index = PatternIndex.load_json(summary.pattern_index_path)

    progressions = index.search(category="progressions", role="harmony", min_quality=3)
    assert summary.pattern_counts["progressions"] >= 3
    assert progressions
    assert all(pattern.payload["length"] in {2, 4, 8} for pattern in progressions)


def test_rule_based_generation_uses_dataset_pattern_index_and_exports(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_test_midi_set(source_dir)

    manifest_path = tmp_path / "dataset_manifest.json"
    create_manifest(source_dir, manifest_path, default_metadata=_default_metadata())
    summary = import_dataset(source_dir, manifest_path, tmp_path / "imported")

    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            seed=82,
            constraints={"pattern_index_path": summary.pattern_index_path},
        ),
        project_id="learned-pattern-generation",
    )

    assert project.metadata["pattern_index_used"] is True
    assert project.validate_bar_durations() == []
    assert _track(project, "drum_kit").metadata["learned_pattern_id"]
    assert _track(project, "double_bass").metadata["learned_pattern_id"]
    assert _track(project, "piano").metadata["learned_pattern_id"]
    assert any(event.annotations.get("drum") == "learned" for event in _notes(project))
    assert any(event.annotations.get("learned_pattern_id") for event in _notes(project))

    manifest = export_project(project, tmp_path / "export", include_pdf=False)
    assert manifest["status"] == "exported"
    assert (tmp_path / "export/full_arrangement.mid").exists()
    assert (tmp_path / "export/full_score.musicxml").exists()


def _default_metadata() -> dict[str, object]:
    return {
        "source": "synthetic_obj9_test",
        "license": "CC0-1.0",
        "copyright_notes": "Generated test fixture",
        "usable_for_training": True,
        "usable_for_pattern_extraction": True,
        "style": "hard_bop",
        "quality": 4,
        "tags": ["drums", "walking_bass", "piano", "melody", "horn_response"],
    }


def _write_test_midi_set(source_dir: Path) -> None:
    for variant in range(10):
        _write_dataset_midi(source_dir / f"learn_{variant:02d}.mid", variant=variant)


def _write_dataset_midi(path: Path, *, variant: int) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    midi_file.tracks.append(_midi_track("Drum Kit", _drum_events(variant)))
    midi_file.tracks.append(_midi_track("Double Bass", _bass_events(variant)))
    midi_file.tracks.append(_midi_track("Piano", _piano_events(variant)))
    midi_file.tracks.append(_midi_track("Alto Sax Lead", _melody_events(variant)))
    midi_file.tracks.append(_midi_track("Trumpet in Bb", _horn_events(variant)))
    midi_file.save(path)


def _drum_events(variant: int) -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    ride_pitch = 51 if variant % 2 == 0 else 42
    for bar in range(4):
        bar_start = bar * 4
        for beat in HALF_BEATS:
            _add_note(events, channel=9, note_number=ride_pitch, start=bar_start + beat)
            if beat in {0.0, 2.0}:
                _add_note(events, channel=9, note_number=36, start=bar_start + beat)
            if beat in {1.5, 3.5}:
                _add_note(events, channel=9, note_number=38, start=bar_start + beat)
            if beat in {1.0, 3.0}:
                _add_note(events, channel=9, note_number=44, start=bar_start + beat)
    return events


def _bass_events(variant: int) -> list[tuple[int, int, mido.Message]]:
    cells = ([0, 3, 7, 10], [0, 4, 7, 9], [0, 2, 5, 11], [0, 5, 7, 10])
    root = 36 + variant
    events: list[tuple[int, int, mido.Message]] = []
    for bar in range(4):
        for beat, interval in enumerate(cells[(variant + bar) % len(cells)]):
            _add_note(
                events,
                channel=0,
                note_number=root + interval + (bar % 2) * 2,
                start=bar * 4 + beat,
                duration=0.9,
                velocity=76,
            )
    return events


def _piano_events(variant: int) -> list[tuple[int, int, mido.Message]]:
    shapes = ([0, 3, 6, 10], [0, 4, 7, 11], [0, 5, 9, 14], [0, 3, 7, 12])
    base = 52 + variant % 5
    events: list[tuple[int, int, mido.Message]] = []
    for bar in range(4):
        for start in (0.5, 1.75, 3.0):
            for interval in shapes[(variant + bar) % len(shapes)]:
                _add_note(
                    events,
                    channel=1,
                    note_number=base + interval,
                    start=bar * 4 + start,
                    duration=0.7,
                    velocity=68,
                )
    return events


def _melody_events(variant: int) -> list[tuple[int, int, mido.Message]]:
    motifs = ([0, 2, 3, 7], [0, -2, 1, 5], [0, 3, 5, 8], [0, 1, -1, 4])
    base = 62 + variant % 6
    events: list[tuple[int, int, mido.Message]] = []
    for bar in range(4):
        for start, interval in zip(
            (0.0, 0.75, 1.5, 2.5),
            motifs[variant % len(motifs)],
            strict=False,
        ):
            _add_note(
                events,
                channel=2,
                note_number=base + interval + bar % 2,
                start=bar * 4 + start,
                duration=0.5,
                velocity=82,
            )
    return events


def _horn_events(variant: int) -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    base = 67 + variant % 4
    for bar in range(4):
        for start, interval in ((2.0, 0), (2.5, 4 + variant % 3)):
            _add_note(
                events,
                channel=3,
                note_number=base + interval + bar % 2,
                start=bar * 4 + start,
                duration=0.5,
                velocity=84,
            )
    return events


def _add_note(
    events: list[tuple[int, int, mido.Message]],
    *,
    channel: int,
    note_number: int,
    start: float,
    duration: float = 0.5,
    velocity: int = 80,
) -> None:
    start_tick = round(start * TICKS_PER_BEAT)
    end_tick = start_tick + round(duration * TICKS_PER_BEAT)
    events.append(
        (
            start_tick,
            1,
            mido.Message("note_on", channel=channel, note=note_number, velocity=velocity),
        )
    )
    events.append(
        (
            end_tick,
            0,
            mido.Message("note_off", channel=channel, note=note_number, velocity=0),
        )
    )


def _midi_track(
    name: str,
    events: list[tuple[int, int, mido.Message]],
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    return track


def _write_progression_musicxml(path: Path) -> None:
    chords = ["Cm7", "F7", "G7", "C7", "Am7", "D7", "Em7", "A7"]
    score = stream.Score(id="obj9-progression-fixture")
    part = stream.Part(id="harmony")
    for index, symbol in enumerate(chords, start=1):
        measure = stream.Measure(number=index)
        if index == 1:
            measure.insert(0, meter.TimeSignature("4/4"))
        measure.insert(0, harmony.ChordSymbol(symbol))
        measure.append(note.Rest(quarterLength=4.0))
        part.append(measure)
    score.append(part)
    score.write("musicxml", fp=str(path))


def _track(project, track_id):
    return next(track for track in project.tracks if track.id == track_id)


def _notes(project):
    return [
        event
        for track in project.tracks
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
