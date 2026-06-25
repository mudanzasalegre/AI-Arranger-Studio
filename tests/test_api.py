from __future__ import annotations

from pathlib import Path

import mido
from app.main import app
from fastapi.testclient import TestClient

TICKS_PER_BEAT = 480


def test_health_endpoint_reports_ok_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ai-arranger-api",
        "status": "ok",
    }


def test_compile_prompt_endpoint_returns_generation_spec():
    client = TestClient(app)

    response = client.post(
        "/v1/prompts/compile",
        json={
            "prompt": (
                "hard bop nocturno en Do menor, 132 bpm, blues menor, "
                "sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria"
            ),
            "seed": 1234,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["style"] == "hard_bop"
    assert payload["key"] == "C minor"
    assert payload["tempo"] == 132
    assert payload["form"] == "minor_blues_12"
    assert payload["ensemble"] == "jazz_sextet"
    assert payload["seed"] == 1234


def test_openapi_contains_objective_10_paths():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/v1/prompts/compile" in paths
    assert "/v1/ai/models" in paths
    assert "/v1/projects/generate" in paths
    assert "/v1/projects/{project_id}" in paths
    assert "/v1/projects/{project_id}/export" in paths
    assert "/v1/projects/{project_id}/takes" in paths
    assert "/v1/projects/{project_id}/takes/{take_id}/accept" in paths
    assert "/v1/projects/{project_id}/takes/{take_id}/reject" in paths
    assert "/v1/projects/{project_id}/ai/plan" in paths
    assert "/v1/projects/{project_id}/ai/infill" in paths
    assert "/v1/ai/text-to-midi-sketch" in paths
    assert "/v1/projects/{project_id}/validation" in paths
    assert "/v1/projects/{project_id}/regenerate" in paths
    assert "/v1/datasets/import" in paths
    assert "/v1/datasets" in paths
    assert "/v1/patterns/search" in paths


def test_project_generation_export_validation_and_regeneration_endpoints(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    generate_response = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, trio",
            "seed": 1201,
            "project_id": "api-lifecycle",
            "options": {"export": True, "validate": True, "include_pdf": False},
        },
    )

    assert generate_response.status_code == 200
    generated = generate_response.json()
    assert generated["project_id"] == "api-lifecycle"
    assert generated["status"] == "generated"
    assert generated["project"]["bar_count"] == 12
    assert generated["validation"]["status"] in {"pass", "pass_with_warnings"}
    assert {file["kind"] for file in generated["files"]} >= {
        "midi_full",
        "musicxml_full",
        "midi_track",
    }

    project_response = client.get("/v1/projects/api-lifecycle")
    assert project_response.status_code == 200
    project_payload = project_response.json()
    assert project_payload["project"]["project_id"] == "api-lifecycle"
    assert project_payload["generation_spec"]["seed"] == 1201
    assert project_payload["export_manifest"]["status"] == "exported"

    validation_response = client.get("/v1/projects/api-lifecycle/validation")
    assert validation_response.status_code == 200
    assert validation_response.json()["status"] in {"pass", "pass_with_warnings"}

    export_response = client.post(
        "/v1/projects/api-lifecycle/export",
        json={"include_pdf": False},
    )
    assert export_response.status_code == 200
    assert export_response.json()["status"] == "exported"

    musicxml_response = client.get("/v1/projects/api-lifecycle/file?kind=musicxml")
    assert musicxml_response.status_code == 200
    assert b"<score-partwise" in musicxml_response.content

    midi_response = client.get("/v1/projects/api-lifecycle/file?kind=midi")
    assert midi_response.status_code == 200
    assert midi_response.content[:4] == b"MThd"

    zip_response = client.get("/v1/projects/api-lifecycle/zip")
    assert zip_response.status_code == 200
    assert zip_response.content[:2] == b"PK"

    regenerate_response = client.post(
        "/v1/projects/api-lifecycle/regenerate",
        json={
            "target": {"track": "piano", "bars": [9, 10, 11, 12]},
            "instruction": "menos movimiento",
            "seed": 1301,
            "options": {"validate": True},
        },
    )
    assert regenerate_response.status_code == 200
    regenerated = regenerate_response.json()
    assert regenerated["status"] == "regenerated"
    assert regenerated["validation"]["status"] in {"pass", "pass_with_warnings"}

    project_after_regenerate = client.get("/v1/projects/api-lifecycle").json()
    assert project_after_regenerate["generation_spec"]["seed"] == 1301
    assert (
        project_after_regenerate["project"]["metadata"]["regenerate_instruction"]
        == "menos movimiento"
    )


def test_dataset_import_list_and_pattern_search_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    source_dir = tmp_path / "dataset-source"
    source_dir.mkdir()
    _write_dataset_midi(source_dir / "pattern_source.mid")
    client = TestClient(app)

    import_response = client.post(
        "/v1/datasets/import",
        json={
            "dataset_id": "fixture-dataset",
            "source_dir": str(source_dir),
            "default_metadata": {
                "source": "api_test",
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

    assert import_response.status_code == 200
    imported = import_response.json()
    assert imported["dataset_id"] == "fixture-dataset"
    assert imported["status"] == "imported"
    assert imported["summary"]["imported_files"] == 1
    assert imported["summary"]["extracted_patterns"] > 0

    list_response = client.get("/v1/datasets")
    assert list_response.status_code == 200
    datasets = list_response.json()
    assert datasets["count"] == 1
    assert datasets["datasets"][0]["dataset_id"] == "fixture-dataset"

    search_response = client.get(
        "/v1/patterns/search",
        params={
            "dataset_id": "fixture-dataset",
            "category": "walking_bass_cells",
            "role": "walking_bass",
            "style": "hard_bop",
            "usable_for_pattern_extraction": True,
        },
    )
    assert search_response.status_code == 200
    search = search_response.json()
    assert search["total"] > 0
    assert all(pattern["role"] == "walking_bass" for pattern in search["patterns"])
    assert all(pattern["usable_for_pattern_extraction"] for pattern in search["patterns"])


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
