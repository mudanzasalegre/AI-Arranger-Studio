from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest
from training import (
    MIDITOK_TRAINING_ROLES,
    MidiTokRealTokenizer,
    MidiTokSource,
    MidiTokUnavailableError,
    export_miditok_role_dataset,
    load_miditok_segments,
)

TICKS_PER_BEAT = 480


def test_miditok_real_export_writes_role_dataset_and_blocks_license(tmp_path):
    pytest.importorskip("miditok")
    midi_path = tmp_path / "combo.mid"
    _write_combo_midi(midi_path)

    summary = export_miditok_role_dataset(
        [
            MidiTokSource(
                path=str(midi_path),
                source_file_id="allowed_combo",
                style="hard_bop",
                license="CC0-1.0",
                chord_context=["Cm7", "F7"],
                source_dataset="synthetic_test",
                track_roles=_track_roles(),
                training_allowed=True,
                commercial_training="allowed",
            ),
            MidiTokSource(
                path=str(midi_path),
                source_file_id="blocked_combo",
                style="hard_bop",
                license="research_only",
                chord_context=["Cm7", "F7"],
                source_dataset="synthetic_test",
                track_roles=_track_roles(),
                training_allowed=True,
                commercial_training="research_only",
            ),
        ],
        tmp_path / "tokenized",
    )

    segments = load_miditok_segments(summary.tokenized_segments_path)
    metadata = _read_jsonl(summary.metadata_path)
    license_report = json.loads(Path(summary.license_report_path).read_text(encoding="utf-8"))

    assert Path(summary.summary_path).exists()
    assert Path(summary.tokenizer_path).exists()
    assert Path(summary.tokenizer_config_path).exists()
    assert Path(summary.quality_report_path).exists()
    assert summary.total_segments == len(MIDITOK_TRAINING_ROLES)
    assert summary.rejected_sources == 1
    assert summary.rejected_segments == 0
    assert summary.acceptable_information_loss is True
    assert {segment.role for segment in segments} == set(MIDITOK_TRAINING_ROLES)
    assert {segment.source_file_id for segment in segments} == {"allowed_combo"}
    assert sum(summary.split_counts[split] for split in ("train", "val", "test")) == len(
        MIDITOK_TRAINING_ROLES
    )
    assert all(segment.token_count > 0 for segment in segments)
    assert all(Path(segment.reconstructed_midi_path or "").exists() for segment in segments)
    assert all(
        {"role", "style", "chord_context", "source_file_id", "license"} <= set(item)
        for item in metadata
    )
    assert license_report["rejected_sources"][0]["source_file_id"] == "blocked_combo"
    for role in MIDITOK_TRAINING_ROLES:
        for split in ("train", "val", "test"):
            assert (tmp_path / "tokenized" / role / f"{split}.jsonl").exists()
        assert (tmp_path / "tokenized" / role / "metadata.jsonl").exists()


def test_miditok_real_converts_musicxml_to_midi(tmp_path):
    pytest.importorskip("miditok")
    stream = pytest.importorskip("music21.stream")
    note = pytest.importorskip("music21.note")
    meter = pytest.importorskip("music21.meter")
    tempo = pytest.importorskip("music21.tempo")

    score = stream.Score()
    part = stream.Part(id="melody")
    part.append(meter.TimeSignature("4/4"))
    part.append(tempo.MetronomeMark(number=132))
    part.append(note.Note("C4", quarterLength=1.0))
    part.append(note.Note("Eb4", quarterLength=1.0))
    score.append(part)
    musicxml_path = tmp_path / "melody.musicxml"
    score.write("musicxml", fp=str(musicxml_path))

    summary = export_miditok_role_dataset(
        [
            MidiTokSource(
                path=str(musicxml_path),
                source_file_id="musicxml_allowed",
                style="hard_bop",
                license="CC0-1.0",
                chord_context=["Cm7"],
                track_roles={"*": "melody"},
            )
        ],
        tmp_path / "musicxml_tokenized",
        roles=("melody",),
    )
    segments = load_miditok_segments(summary.tokenized_segments_path)

    assert summary.total_segments == 1
    assert sum(summary.split_counts[split] for split in ("train", "val", "test")) == 1
    assert segments[0].role == "melody"
    assert segments[0].source_file_id == "musicxml_allowed"
    assert Path(segments[0].midi_path).exists()


def test_miditok_real_missing_dependency_is_controlled(monkeypatch):
    from training.tokenizers import miditok_real

    def fake_import_module(name: str):
        if name == "miditok":
            raise ImportError("missing miditok")
        return __import__(name)

    monkeypatch.setattr(miditok_real.importlib, "import_module", fake_import_module)

    with pytest.raises(MidiTokUnavailableError):
        _ = MidiTokRealTokenizer().tokenizer


def test_miditok_tokenizer_can_be_built_from_yaml_config(tmp_path):
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text(
        """
tokenizer:
  family: REMI
  use_programs: true
  use_tempos: true
  use_time_signatures: true
  beat_res:
    0_4: 8
    4_12: 4
""".lstrip(),
        encoding="utf-8",
    )

    tokenizer = MidiTokRealTokenizer.from_config(config_path)

    assert tokenizer.tokenizer_family == "REMI"
    assert tokenizer.beat_res == {(0, 4): 8, (4, 12): 4}
    assert tokenizer.use_programs is True


def _write_combo_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(132), time=0))
    midi.tracks.append(meta)
    midi.tracks.append(_track("Alto Sax Melody", 0, [(0, 480, 60), (480, 480, 63)]))
    midi.tracks.append(_track("Upright Bass Walking", 1, [(0, 480, 36), (480, 480, 43)]))
    midi.tracks.append(_track("Piano Comping", 2, [(0, 720, 60), (0, 720, 63)]))
    midi.tracks.append(_track("Trumpet Horn Responses", 3, [(480, 240, 67), (720, 240, 70)]))
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


def _track_roles() -> dict[str, str]:
    return {
        "Alto Sax Melody": "melody",
        "Upright Bass Walking": "walking_bass",
        "Piano Comping": "piano_comping",
        "Trumpet Horn Responses": "horn_responses",
        "Drum Kit": "drums",
    }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
