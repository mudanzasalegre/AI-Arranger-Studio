from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import mido

ROOT = Path(__file__).resolve().parents[2]
for package in ("dataset_tools", "training"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from training import (  # noqa: E402
    MIDITOK_TRAINING_ROLES,
    MidiTokSource,
    export_miditok_role_dataset,
    load_miditok_segments,
)

TICKS_PER_BEAT = 480


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the real MidiTok role tokenizer.")
    parser.add_argument("--output-dir", default="data/processed/tokenized")
    parser.add_argument("--fixture-dir", default="outputs/model_smoke/miditok_fixture")
    parser.add_argument("--max-loss", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    fixture_dir = (ROOT / args.fixture_dir).resolve()
    _clean_generated_path(output_dir)
    _clean_generated_path(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    source_midi = fixture_dir / "licensed_combo.mid"
    blocked_midi = fixture_dir / "blocked_combo.mid"
    _write_role_fixture(source_midi)
    shutil.copyfile(source_midi, blocked_midi)

    summary = export_miditok_role_dataset(
        [
            MidiTokSource(
                path=str(source_midi),
                source_file_id="synthetic_pr24_allowed",
                style="hard_bop",
                license="CC0-1.0",
                chord_context=["Cm7", "F7", "Bbmaj7", "Ebmaj7"],
                source_dataset="synthetic_pr24",
                track_roles=_track_roles(),
                training_allowed=True,
                commercial_training="allowed",
                tags=["pr24", "smoke", "licensed"],
            ),
            MidiTokSource(
                path=str(blocked_midi),
                source_file_id="synthetic_pr24_blocked",
                style="hard_bop",
                license="research_only",
                chord_context=["Cm7", "F7", "Bbmaj7", "Ebmaj7"],
                source_dataset="synthetic_pr24",
                track_roles=_track_roles(),
                training_allowed=True,
                commercial_training="research_only",
                tags=["pr24", "smoke", "blocked"],
            ),
        ],
        output_dir,
        max_acceptable_loss_ratio=args.max_loss,
    )
    segments = load_miditok_segments(summary.tokenized_segments_path)
    metadata = _read_jsonl(summary.metadata_path)
    license_report = json.loads(Path(summary.license_report_path).read_text(encoding="utf-8"))

    if {segment.role for segment in segments} != set(MIDITOK_TRAINING_ROLES):
        raise RuntimeError("MidiTok smoke did not produce one segment per training role")
    if any(segment.source_file_id == "synthetic_pr24_blocked" for segment in segments):
        raise RuntimeError("Blocked-license source was tokenized")
    if not license_report["rejected_sources"]:
        raise RuntimeError("MidiTok license report did not capture the blocked source")
    if not summary.acceptable_information_loss:
        raise RuntimeError(
            f"MidiTok reconstruction loss too high: {summary.max_information_loss_ratio}"
        )
    if not all(Path(segment.reconstructed_midi_path or "").exists() for segment in segments):
        raise RuntimeError("MidiTok smoke did not reconstruct every tokenized segment")
    if not _metadata_fields_present(metadata):
        raise RuntimeError("MidiTok metadata is missing a required PR-24 field")
    for role in MIDITOK_TRAINING_ROLES:
        for split in ("train", "val", "test"):
            if not (output_dir / role / f"{split}.jsonl").exists():
                raise RuntimeError(f"Missing stable split output for {role}/{split}")

    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "fixture_dir": str(fixture_dir),
        "roles": list(MIDITOK_TRAINING_ROLES),
        "total_segments": summary.total_segments,
        "train_segments": summary.train_segments,
        "val_segments": summary.val_segments,
        "test_segments": summary.test_segments,
        "rejected_sources": summary.rejected_sources,
        "role_counts": summary.role_counts,
        "split_counts": summary.split_counts,
        "average_information_loss_ratio": summary.average_information_loss_ratio,
        "max_information_loss_ratio": summary.max_information_loss_ratio,
        "acceptable_information_loss": summary.acceptable_information_loss,
        "tokenization_summary_path": summary.summary_path,
        "license_report_path": summary.license_report_path,
        "quality_report_path": summary.quality_report_path,
        "tokenizer_path": summary.tokenizer_path,
    }
    report_path = ROOT / "outputs/model_smoke/miditok_smoke_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_role_fixture(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("key_signature", key="Cm", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(132), time=0))
    midi.tracks.append(meta)
    midi.tracks.append(_midi_track("Alto Sax Melody", 65, 0, _melody_events()))
    midi.tracks.append(_midi_track("Upright Bass Walking", 32, 1, _bass_events()))
    midi.tracks.append(_midi_track("Piano Comping", 0, 2, _piano_events()))
    midi.tracks.append(_midi_track("Trumpet Horn Responses", 56, 3, _horn_events()))
    midi.tracks.append(_midi_track("Drum Kit", 0, 9, _drum_events()))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def _midi_track(
    name: str,
    program: int,
    channel: int,
    events: list[tuple[int, int, mido.Message]],
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if channel != 9:
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
    current = 0
    for absolute_time, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=absolute_time - current))
        current = absolute_time
    return track


def _melody_events() -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(0, 480, 60), (480, 480, 63), (960, 480, 67), (1440, 480, 70)],
        channel=0,
        velocity=88,
    )


def _bass_events() -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(0, 480, 36), (480, 480, 43), (960, 480, 46), (1440, 480, 48)],
        channel=1,
        velocity=82,
    )


def _piano_events() -> list[tuple[int, int, mido.Message]]:
    notes = [
        (0, 720, 51),
        (0, 720, 58),
        (0, 720, 63),
        (960, 720, 53),
        (960, 720, 60),
        (960, 720, 65),
    ]
    return _note_events(notes, channel=2, velocity=72)


def _horn_events() -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(480, 240, 67), (720, 240, 70), (1440, 240, 75), (1680, 240, 77)],
        channel=3,
        velocity=84,
    )


def _drum_events() -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [
            (0, 120, 36),
            (480, 120, 42),
            (960, 120, 38),
            (1440, 120, 46),
        ],
        channel=9,
        velocity=90,
    )


def _note_events(
    notes: list[tuple[int, int, int]],
    *,
    channel: int,
    velocity: int,
) -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for start, duration, note in notes:
        events.append(
            (
                start,
                0,
                mido.Message("note_on", channel=channel, note=note, velocity=velocity),
            )
        )
        events.append(
            (
                start + duration,
                1,
                mido.Message("note_off", channel=channel, note=note, velocity=0),
            )
        )
    return events


def _track_roles() -> dict[str, str]:
    return {
        "Alto Sax Melody": "melody",
        "Upright Bass Walking": "walking_bass",
        "Piano Comping": "piano_comping",
        "Trumpet Horn Responses": "horn_responses",
        "Drum Kit": "drums",
    }


def _metadata_fields_present(metadata: list[dict[str, Any]]) -> bool:
    required = {"role", "style", "chord_context", "source_file_id", "license"}
    return bool(metadata) and all(required <= set(item) for item in metadata)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clean_generated_path(path: Path) -> None:
    allowed_roots = [
        (ROOT / "data" / "processed").resolve(),
        (ROOT / "outputs").resolve(),
    ]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise RuntimeError(f"Refusing to clean path outside generated roots: {path}")
    if path.exists():
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
