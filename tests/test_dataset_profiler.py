from __future__ import annotations

import json
from pathlib import Path

import mido
from dataset_tools import (
    DatasetProfileReport,
    PatternIndex,
    create_manifest,
    import_dataset,
    profile_dataset_file,
)
from music21 import harmony, meter, note, stream

TICKS_PER_BEAT = 480


def test_profile_dataset_file_classifies_midi_tracks_and_features(tmp_path):
    midi_path = tmp_path / "combo.mid"
    _write_profile_midi(midi_path)

    profile = profile_dataset_file(
        midi_path,
        file_id="combo",
        metadata={
            "source": "profile_test",
            "license": "CC0-1.0",
            "usable_for_training": True,
            "usable_for_pattern_extraction": True,
            "style": "hard_bop",
            "quality": 4,
        },
    )
    roles = {track.classification.role for track in profile.track_profiles}

    assert {"drums", "bass", "comping", "melody", "horns", "pad"} <= roles
    assert profile.contains_melody is True
    assert profile.contains_chords is True
    assert profile.contains_arrangement is True
    assert profile.commercial_training == "allowed"
    assert profile.pattern_sensitivity["level"] == "low"
    assert profile.file_features["note_count"] > 0
    assert all(track.no_memorization_fingerprint for track in profile.track_profiles)
    assert min(track.classification.confidence for track in profile.track_profiles) >= 0.72


def test_musicxml_profile_detects_chords_and_harmony_role(tmp_path):
    musicxml_path = tmp_path / "progression.musicxml"
    _write_progression_musicxml(musicxml_path)

    profile = profile_dataset_file(
        musicxml_path,
        file_id="lead-sheet",
        metadata={
            "source": "profile_test",
            "license": "CC0-1.0",
            "usable_for_training": True,
            "usable_for_pattern_extraction": True,
        },
    )

    assert profile.format == "musicxml"
    assert profile.contains_chords is True
    assert "harmony" in profile.role_coverage
    assert profile.file_features["chord_symbols"] == 4


def test_import_dataset_writes_profile_manifest_and_enriches_patterns(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_profile_midi(source_dir / "combo.mid")

    manifest_path = tmp_path / "dataset_manifest.json"
    create_manifest(
        source_dir,
        manifest_path,
        default_metadata={
            "source": "profile_import_test",
            "license": "CC0-1.0",
            "usable_for_training": True,
            "usable_for_pattern_extraction": True,
            "style": "hard_bop",
            "quality": 4,
        },
    )
    summary = import_dataset(source_dir, manifest_path, tmp_path / "imported")
    profile_report = DatasetProfileReport.load_json(summary.profile_report_path)
    role_manifest = json.loads(Path(summary.role_manifest_path).read_text(encoding="utf-8"))
    normalized_files = json.loads(Path(summary.normalized_files_path).read_text(encoding="utf-8"))
    pattern_index = PatternIndex.load_json(summary.pattern_index_path)

    assert summary.profiled_files == 1
    assert {"drums", "bass", "comping", "melody", "horns", "pad"} <= set(
        summary.role_counts
    )
    assert profile_report.file_count == 1
    assert role_manifest["files"][0]["roles"]
    assert {"bass", "comping", "drums"} <= set(normalized_files[0]["roles"])
    assert normalized_files[0]["stats"]["contains_arrangement"] is True
    assert pattern_index.patterns
    assert all(pattern.context.get("classified_role") for pattern in pattern_index.patterns)
    assert all(pattern.context.get("role_confidence", 0) > 0 for pattern in pattern_index.patterns)
    assert all(
        pattern.context.get("no_memorization_fingerprint")
        for pattern in pattern_index.patterns
    )
    assert all(
        pattern.context["pattern_sensitivity"]["level"] == "low"
        for pattern in pattern_index.patterns
    )


def _write_profile_midi(path: Path) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    midi_file.tracks.append(_midi_track("Drum Kit", _drum_events()))
    midi_file.tracks.append(_midi_track("Upright Bass", _bass_events()))
    midi_file.tracks.append(_midi_track("Piano Comping", _piano_events()))
    midi_file.tracks.append(_midi_track("Alto Sax Lead", _melody_events()))
    midi_file.tracks.append(_midi_track("Trumpet Section", _horn_events()))
    midi_file.tracks.append(_midi_track("Warm Synth Pad", _pad_events()))
    midi_file.save(path)


def _drum_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for beat_index in range(8):
        start = beat_index * 0.5
        _add_note(events, channel=9, note_number=51, start=start)
        if start in {0.0, 2.0}:
            _add_note(events, channel=9, note_number=36, start=start)
        if start in {1.5, 3.5}:
            _add_note(events, channel=9, note_number=38, start=start)
    return events


def _bass_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for beat, note_number in enumerate((36, 39, 43, 46)):
        _add_note(events, channel=0, note_number=note_number, start=float(beat), duration=0.9)
    return events


def _piano_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for start in (0.5, 1.75, 3.0):
        for note_number in (52, 55, 58, 62):
            _add_note(events, channel=1, note_number=note_number, start=start, duration=0.7)
    return events


def _melody_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for start, note_number in zip((0.0, 0.75, 1.5, 2.5), (62, 65, 67, 70), strict=False):
        _add_note(events, channel=2, note_number=note_number, start=start, duration=0.5)
    return events


def _horn_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for start, note_number in ((2.0, 67), (2.5, 71)):
        _add_note(events, channel=3, note_number=note_number, start=start, duration=0.5)
    return events


def _pad_events() -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for note_number in (60, 67):
        _add_note(events, channel=4, note_number=note_number, start=0.0, duration=4.0)
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
    score = stream.Score(id="profile-progression")
    part = stream.Part(id="harmony")
    for index, symbol in enumerate(("Cm7", "F7", "G7", "C7"), start=1):
        measure = stream.Measure(number=index)
        if index == 1:
            measure.insert(0, meter.TimeSignature("4/4"))
        measure.insert(0, harmony.ChordSymbol(symbol))
        measure.append(note.Rest(quarterLength=4.0))
        part.append(measure)
    score.append(part)
    score.write("musicxml", fp=str(path))
