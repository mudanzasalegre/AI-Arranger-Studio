from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import mido
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/api"))
sys.path.insert(0, str(ROOT / "packages/arranger_core"))
sys.path.insert(0, str(ROOT / "packages/dataset_tools"))
sys.path.insert(0, str(ROOT / "packages/midi_models"))
sys.path.insert(0, str(ROOT / "packages/model_backends"))

from app.main import app  # noqa: E402

TICKS_PER_BEAT = 480


def main() -> None:
    demo_root = (ROOT / "outputs/obj10_api_demo").resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if outputs_root not in demo_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {demo_root}")
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True)

    os.environ["AI_ARRANGER_API_STORAGE"] = str(demo_root / "api_storage")
    client = TestClient(app)

    openapi = _ok(client.get("/openapi.json")).json()
    generated = _ok(
        client.post(
            "/v1/projects/generate",
            json={
                "project_id": "obj10-api-smoke",
                "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, trio",
                "seed": 1010,
                "options": {"export": True, "validate": True, "include_pdf": False},
            },
        )
    ).json()
    project = _ok(client.get("/v1/projects/obj10-api-smoke")).json()
    validation = _ok(client.get("/v1/projects/obj10-api-smoke/validation")).json()
    regenerated = _ok(
        client.post(
            "/v1/projects/obj10-api-smoke/regenerate",
            json={
                "target": {"track": "piano", "bars": [9, 10, 11, 12]},
                "instruction": "menos movimiento",
                "seed": 1011,
                "options": {"validate": True},
            },
        )
    ).json()

    source_dir = demo_root / "dataset_source"
    source_dir.mkdir()
    _write_dataset_midi(source_dir / "api_pattern_source.mid")
    dataset = _ok(
        client.post(
            "/v1/datasets/import",
            json={
                "dataset_id": "obj10-dataset",
                "source_dir": str(source_dir),
                "default_metadata": {
                    "source": "api_smoke",
                    "license": "CC0-1.0",
                    "copyright_notes": "Generated fixture",
                    "usable_for_training": True,
                    "usable_for_pattern_extraction": True,
                    "style": "hard_bop",
                    "quality": 4,
                    "tags": ["drums", "walking_bass"],
                },
            },
        )
    ).json()
    datasets = _ok(client.get("/v1/datasets")).json()
    patterns = _ok(
        client.get(
            "/v1/patterns/search",
            params={
                "dataset_id": "obj10-dataset",
                "category": "walking_bass_cells",
                "role": "walking_bass",
                "style": "hard_bop",
            },
        )
    ).json()

    summary = {
        "openapi_paths": len(openapi["paths"]),
        "project_id": generated["project_id"],
        "generated_status": generated["status"],
        "project_bar_count": project["project"]["bar_count"],
        "validation_status": validation["status"],
        "regenerated_status": regenerated["status"],
        "exported_files": len(generated["files"]),
        "dataset_status": dataset["status"],
        "datasets_count": datasets["count"],
        "patterns_found": patterns["total"],
    }
    (demo_root / "api_smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _ok(response):
    if response.status_code >= 400:
        raise RuntimeError(f"API request failed {response.status_code}: {response.text}")
    return response


def _write_dataset_midi(path: Path) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    midi_file.tracks.append(_midi_track("Drum Kit", _drum_events()))
    midi_file.tracks.append(_midi_track("Double Bass", _bass_events()))
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
        _add_note(
            events,
            channel=0,
            note_number=note_number,
            start=float(beat),
            duration=0.9,
            velocity=76,
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


if __name__ == "__main__":
    main()
