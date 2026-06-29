from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

from app.main import app
from arranger_core import ArrangementProject
from fastapi.testclient import TestClient


def test_ai_infill_mock_creates_pending_take_without_changing_active_project(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-ai-infill"
    _generate_project(client, project_id)
    project_dir = tmp_path / "api-storage" / "projects" / project_id
    before = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    before_tracks = [track.model_dump(mode="json") for track in before.tracks]

    response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "mock_symbolic",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "bebop phrase, medium density",
            "density": "medium",
            "temperature": 0.85,
            "seed": 501,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending_take"
    assert payload["backend"] == "mock_symbolic"
    assert payload["take"]["status"] == "pending"
    assert payload["take"]["track_id"] == "alto_sax"
    assert payload["take"]["bars"] == [1]
    assert payload["artifact"]["status"] == "validated"
    assert payload["validation"]["status"] in {"pass", "pass_with_warnings"}

    active_after = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    assert [track.model_dump(mode="json") for track in active_after.tracks] == before_tracks

    candidate = ArrangementProject.load_json(payload["take"]["project_snapshot_path"])
    _assert_only_target_bar_changed(before, candidate, track_id="alto_sax", bars={1})
    assert candidate.metadata["take_status"] == "pending"
    assert payload["take"]["metadata"]["model_trace"]["backend"] == "mock_symbolic"

    takes = client.get(f"/v1/projects/{project_id}/takes").json()
    assert takes["active_take_id"] == "take_base"
    assert {take["status"] for take in takes["takes"]} == {"accepted", "pending"}


def test_ai_infill_invalid_midi_artifact_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-ai-infill-invalid"
    _generate_project(client, project_id, seed=502)
    project_dir = tmp_path / "api-storage" / "projects" / project_id
    before = ArrangementProject.load_json(project_dir / "arrangement_project.json")

    response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "mock_symbolic",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "broken fixture",
            "metadata": {"mock_artifact": "invalid_midi"},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["status"] == "rejected"
    active_after = ArrangementProject.load_json(project_dir / "arrangement_project.json")
    assert active_after.model_dump(mode="json") == before.model_dump(mode="json")
    rejected = _artifact_records(tmp_path)[0]
    assert rejected["status"] == "rejected"
    assert "MIDI artifact is not parseable" in rejected["metadata"]["rejection_reason"]


def test_ai_infill_empty_candidate_fails_validation_and_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-ai-infill-empty"
    _generate_project(client, project_id, seed=503)

    response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "mock_symbolic",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "empty fixture",
            "metadata": {"mock_artifact": "empty_midi"},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "rejected"
    assert detail["validation"]["status"] == "fail"
    assert {issue["code"] for issue in detail["validation"]["errors"]} == {
        "empty_target_material"
    }
    rejected = _artifact_records(tmp_path)[0]
    assert rejected["status"] == "rejected"
    assert rejected["metadata"]["rejection_reason"] == "validation_failed"


def test_daw_ready_zip_contains_trace_and_excludes_pending_takes(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-daw-export"
    _generate_project(client, project_id, seed=505)

    accepted_response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "mock_symbolic",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "bebop phrase, medium density",
            "density": "medium",
            "temperature": 0.85,
            "seed": 5051,
        },
    )
    assert accepted_response.status_code == 200
    accepted_take_id = accepted_response.json()["take"]["take_id"]

    accept_response = client.post(f"/v1/projects/{project_id}/takes/{accepted_take_id}/accept")
    assert accept_response.status_code == 200
    export_response = client.post(
        f"/v1/projects/{project_id}/export",
        json={"include_pdf": False},
    )
    assert export_response.status_code == 200

    zip_response = client.get(f"/v1/projects/{project_id}/zip")
    assert zip_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        names = set(archive.namelist())
        trace = json.loads(archive.read("model_trace.json").decode("utf-8"))
        takes_manifest = json.loads(archive.read("takes_manifest.json").decode("utf-8"))

    assert {
        "arrangement_project.json",
        "export_manifest.json",
        "full_arrangement.mid",
        "full_score.musicxml",
        "model_trace.json",
        "session_readme.md",
        "takes_manifest.json",
        "validation_report.html",
    } <= names
    assert not any(name.startswith(("takes/", "model_contexts/")) for name in names)
    assert trace["model_artifacts"][0]["take_id"] == accepted_take_id
    assert trace["model_artifacts"][0]["backend_id"] == "mock_symbolic"
    assert trace["model_artifacts"][0]["track_id"] == "alto_sax"
    assert trace["model_artifacts"][0]["bars"] == [1]
    assert trace["model_artifacts"][0]["validation_result"] in {"pass", "pass_with_warnings"}
    exported_take_ids = {take["take_id"] for take in takes_manifest["takes"]}
    assert accepted_take_id in exported_take_ids
    assert {take["status"] for take in takes_manifest["takes"]} == {"accepted"}


def test_pending_take_blocks_final_export(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)
    project_id = "api-daw-pending-block"
    _generate_project(client, project_id, seed=506)

    pending_response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "mock_symbolic",
            "track_id": "piano",
            "bars": [2],
            "instruction": "short comping answer",
            "density": "medium",
            "temperature": 0.85,
            "seed": 5061,
        },
    )
    assert pending_response.status_code == 200

    export_response = client.post(
        f"/v1/projects/{project_id}/export",
        json={"include_pdf": False},
    )

    assert export_response.status_code == 422
    errors = export_response.json()["detail"]["errors"]
    assert any(issue["code"] == "pending_take_present" for issue in errors)


def test_ai_infill_midigpt_missing_dependency_is_controlled(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  midigpt:
    enabled: true
    type: symbolic
    adapter: model_backends.symbolic.midigpt_backend.MidiGptBackend
    commercial_use: review_required
    dependency_mode: optional
    install_hint: pip install "midigpt[inference]"
    tasks:
      - infill_bars
    capabilities:
      symbolic_midi: true
      multitrack: true
      bar_infill: true
      track_generation: true
      commercial_use: review_required
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    client = TestClient(app)
    project_id = "api-ai-infill-midigpt-missing"
    _generate_project(client, project_id, seed=504)

    response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "midigpt",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "bebop phrase",
        },
    )

    assert response.status_code == 409
    assert "MIDI-GPT is not installed" in response.json()["detail"]


def _generate_project(client: TestClient, project_id: str, *, seed: int = 500) -> None:
    response = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
            "seed": seed,
            "project_id": project_id,
            "options": {"validate": True},
        },
    )
    assert response.status_code == 200


def _artifact_records(tmp_path: Path) -> list[dict]:
    manifest_path = tmp_path / "api-storage" / "model_artifacts" / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload["artifacts"]


def _assert_only_target_bar_changed(
    before: ArrangementProject,
    after: ArrangementProject,
    *,
    track_id: str,
    bars: set[int],
) -> None:
    before_tracks = {track.id: track.model_dump(mode="json") for track in before.tracks}
    after_tracks = {track.id: track.model_dump(mode="json") for track in after.tracks}
    assert set(after_tracks) == set(before_tracks)
    for current_track_id, before_track in before_tracks.items():
        after_track = after_tracks[current_track_id]
        if current_track_id != track_id:
            assert after_track == before_track
            continue
        before_by_bar = {bar["number"]: bar for bar in before_track["bars"]}
        after_by_bar = {bar["number"]: bar for bar in after_track["bars"]}
        assert set(after_by_bar) == set(before_by_bar)
        for bar_number, before_bar in before_by_bar.items():
            if bar_number in bars:
                assert after_by_bar[bar_number] != before_bar
            else:
                assert after_by_bar[bar_number] == before_bar
