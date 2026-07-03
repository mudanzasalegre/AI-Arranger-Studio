from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[2]
for package in ("dataset_tools", "training"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from dataset_tools import create_manifest, sha256_file  # noqa: E402
from training import (  # noqa: E402
    MIDITOK_TRAINING_ROLES,
    MidiTokRealTokenizer,
    export_miditok_role_dataset,
    load_miditok_segments,
    miditok_sources_from_dataset_manifest,
)
from training.tokenizers.miditok_real import _split_for_source_file_id  # noqa: E402

TICKS_PER_BEAT = 480


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a MidiTok role dataset from a DatasetManifest."
    )
    parser.add_argument("--output-dir", default="data/processed/tokenized")
    parser.add_argument("--fixture-dir", default="outputs/pro_benchmarks/miditok_manifest_fixture")
    parser.add_argument("--max-loss", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    fixture_dir = (ROOT / args.fixture_dir).resolve()
    _clean_generated_path(output_dir)
    _clean_generated_path(fixture_dir)
    source_dir = fixture_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    allowed_count = _write_allowed_sources_covering_all_splits(source_dir)
    blocked_path = source_dir / "zz_blocked_non_commercial.mid"
    _write_role_fixture(blocked_path, transpose=11)

    manifest_path = fixture_dir / "dataset_manifest.json"
    manifest = create_manifest(
        source_dir,
        manifest_path,
        default_metadata={
            "source": "synthetic_pr33_manifest",
            "license": "CC0-1.0",
            "commercial_training": "allowed",
            "usable_for_training": True,
            "usable_for_pattern_extraction": True,
            "style": "hard_bop",
            "quality": 4,
            "tags": ["pr33", "manifest", "miditok"],
        },
        metadata_by_name={
            blocked_path.name: {
                "source": "synthetic_pr33_manifest",
                "license": "CC-BY-NC",
                "commercial_training": "forbidden",
                "usable_for_training": True,
                "usable_for_pattern_extraction": True,
                "style": "hard_bop",
                "quality": 4,
                "tags": ["pr33", "manifest", "blocked"],
            }
        },
    )

    tokenizer_config = fixture_dir / "miditok_tokenizer.yaml"
    tokenizer_config.write_text(
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
    sources = miditok_sources_from_dataset_manifest(
        manifest,
        source_root=source_dir,
        min_role_confidence=0.5,
    )
    summary = export_miditok_role_dataset(
        sources,
        output_dir,
        tokenizer=MidiTokRealTokenizer.from_config(tokenizer_config),
        max_acceptable_loss_ratio=args.max_loss,
        min_quality=3,
        min_notes_per_source=2,
        min_duration_beats=1.0,
        supported_time_signatures=("4/4",),
    )
    segments = load_miditok_segments(summary.tokenized_segments_path)
    license_report = json.loads(Path(summary.license_report_path).read_text(encoding="utf-8"))
    quality_report = json.loads(Path(summary.quality_report_path).read_text(encoding="utf-8"))

    expected_segments = allowed_count * len(MIDITOK_TRAINING_ROLES)
    split_counts = {split: summary.split_counts.get(split, 0) for split in ("train", "val", "test")}
    if summary.total_segments != expected_segments:
        raise RuntimeError(
            f"Expected {expected_segments} tokenized segments, got {summary.total_segments}"
        )
    if not all(count > 0 for count in split_counts.values()):
        raise RuntimeError(f"Manifest smoke did not cover all splits: {split_counts}")
    if summary.rejected_sources != 1:
        raise RuntimeError(
            f"Expected one rejected non-commercial source, got {summary.rejected_sources}"
        )
    if any(segment.license.lower() == "cc-by-nc" for segment in segments):
        raise RuntimeError("Non-commercial source was tokenized")
    if not summary.acceptable_information_loss:
        raise RuntimeError(
            "MidiTok information loss exceeded threshold: "
            f"{summary.max_information_loss_ratio}"
        )
    if not license_report["rejected_sources"]:
        raise RuntimeError("License report did not include rejected source")
    if not quality_report["information_loss"]["acceptable"]:
        raise RuntimeError("Quality report marked information loss as unacceptable")

    report = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "allowed_sources": allowed_count,
        "rejected_sources": summary.rejected_sources,
        "total_segments": summary.total_segments,
        "role_counts": summary.role_counts,
        "split_counts": split_counts,
        "max_information_loss_ratio": summary.max_information_loss_ratio,
        "tokenization_summary_path": summary.summary_path,
        "license_report_path": summary.license_report_path,
        "quality_report_path": summary.quality_report_path,
    }
    report_path = ROOT / "outputs/pro_benchmarks/miditok_dataset_from_manifest_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_allowed_sources_covering_all_splits(source_dir: Path) -> int:
    for variant in range(1, 41):
        path = source_dir / f"allowed_{variant:02d}.mid"
        _write_role_fixture(path, transpose=variant % 12)
        splits = set()
        allowed_paths = sorted(source_dir.glob("allowed_*.mid"))
        for index, midi_path in enumerate(allowed_paths, start=1):
            source_id = f"file_{index:04d}_{sha256_file(midi_path)[:12]}"
            splits.add(_split_for_source_file_id(source_id))
        if {"train", "val", "test"} <= splits:
            return len(allowed_paths)
    raise RuntimeError("Could not synthesize manifest sources covering train/val/test splits")


def _write_role_fixture(path: Path, *, transpose: int) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("key_signature", key="Cm", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(132), time=0))
    midi.tracks.append(meta)
    midi.tracks.append(_midi_track("Alto Sax Melody", 65, 0, _melody_events(transpose)))
    midi.tracks.append(_midi_track("Upright Bass Walking", 32, 1, _bass_events(transpose)))
    midi.tracks.append(_midi_track("Piano Comping", 0, 2, _piano_events(transpose)))
    midi.tracks.append(_midi_track("Trumpet Horn Responses", 56, 3, _horn_events(transpose)))
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


def _melody_events(transpose: int) -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(0, 480, 60 + transpose), (480, 480, 63 + transpose), (960, 480, 67 + transpose)],
        channel=0,
        velocity=88,
    )


def _bass_events(transpose: int) -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(0, 480, 36 + transpose), (480, 480, 43 + transpose), (960, 480, 46 + transpose)],
        channel=1,
        velocity=82,
    )


def _piano_events(transpose: int) -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [
            (0, 720, 51 + transpose),
            (0, 720, 58 + transpose),
            (0, 720, 63 + transpose),
            (960, 720, 53 + transpose),
            (960, 720, 60 + transpose),
            (960, 720, 65 + transpose),
        ],
        channel=2,
        velocity=72,
    )


def _horn_events(transpose: int) -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(480, 240, 67 + transpose), (720, 240, 70 + transpose), (1440, 240, 75 + transpose)],
        channel=3,
        velocity=84,
    )


def _drum_events() -> list[tuple[int, int, mido.Message]]:
    return _note_events(
        [(0, 120, 36), (480, 120, 42), (960, 120, 38)],
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
