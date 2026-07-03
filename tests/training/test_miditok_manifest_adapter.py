from __future__ import annotations

from pathlib import Path

import mido
from dataset_tools import DatasetManifest, DatasetManifestEntry
from training import miditok_sources_from_dataset_manifest

TICKS_PER_BEAT = 480


def test_miditok_sources_from_dataset_manifest_use_profiled_track_roles(tmp_path):
    midi_path = tmp_path / "combo.mid"
    _write_manifest_combo_midi(midi_path)
    manifest = DatasetManifest(
        entries=[
            DatasetManifestEntry(
                path="combo.mid",
                source="synthetic_manifest",
                license="CC0-1.0",
                commercial_training="allowed",
                usable_for_training=True,
                usable_for_pattern_extraction=True,
                style="hard_bop",
                quality=4,
                tags=["manifest", "smoke"],
            )
        ]
    )

    sources = miditok_sources_from_dataset_manifest(manifest, source_root=tmp_path)

    assert len(sources) == 1
    source = sources[0]
    assert source.source_dataset == "synthetic_manifest"
    assert source.training_allowed is True
    assert source.commercial_training == "allowed"
    assert source.metadata["quality"] == 4
    assert source.track_roles["Alto Sax Melody"] == "melody"
    assert source.track_roles["Upright Bass Walking"] == "walking_bass"
    assert source.track_roles["Piano Comping"] == "piano_comping"
    assert source.track_roles["Drum Kit"] == "drums"


def _write_manifest_combo_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(132), time=0))
    midi.tracks.append(meta)
    midi.tracks.append(_track("Alto Sax Melody", 0, [(0, 480, 60), (480, 480, 63)]))
    midi.tracks.append(_track("Upright Bass Walking", 1, [(0, 480, 36), (480, 480, 43)]))
    midi.tracks.append(_track("Piano Comping", 2, [(0, 720, 60), (0, 720, 63)]))
    midi.tracks.append(_track("Drum Kit", 9, [(0, 120, 36), (480, 120, 42)]))
    midi.save(path)


def _track(name: str, channel: int, notes: list[tuple[int, int, int]]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if channel != 9:
        track.append(mido.Message("program_change", channel=channel, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for start, duration, note in notes:
        events.append(
            (start, 0, mido.Message("note_on", channel=channel, note=note, velocity=80))
        )
        events.append(
            (
                start + duration,
                1,
                mido.Message("note_off", channel=channel, note=note, velocity=0),
            )
        )
    current = 0
    for absolute_time, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=absolute_time - current))
        current = absolute_time
    return track
