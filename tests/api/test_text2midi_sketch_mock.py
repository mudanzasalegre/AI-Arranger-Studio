from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import mido
from app.main import app
from arranger_core import ArrangementProject
from fastapi.testclient import TestClient


def test_text_to_midi_sketch_mock_imports_experimental_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "mock_symbolic",
            "prompt": "Hard bop minor blues sketch in C minor, alto sax lead",
            "seed": 601,
            "metadata": {"mock_artifact": "unlabeled_midi"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "sketch_uncertain"
    assert payload["backend"] == "mock_symbolic"
    assert payload["artifact"]["status"] == "validated"
    assert payload["validation"]["status"] in {"pass", "pass_with_warnings"}
    assert "no_roles_detected" in payload["sketch"]["uncertainty_reasons"]

    sketch_id = payload["sketch_id"]
    project_path = tmp_path / "api-storage" / "sketches" / sketch_id / "arrangement_project.json"
    assert project_path.exists()
    assert not (tmp_path / "api-storage" / "projects" / sketch_id).exists()

    project = ArrangementProject.load_json(project_path)
    assert project.metadata["project_type"] == "text2midi_sketch"
    assert project.metadata["professional_project"] is False
    assert project.metadata["auto_merge_allowed"] is False
    assert project.metadata["sketch_status"] == "sketch_uncertain"
    assert project.tracks
    assert all(track.metadata["source"] == "text2midi_sketch" for track in project.tracks)

    records = _artifact_records(tmp_path)
    assert records[0]["project_id"] == sketch_id
    assert records[0]["task"] == "generate_full_sketch"
    assert records[0]["status"] == "validated"


def test_text_to_midi_sketch_invalid_midi_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "mock_symbolic",
            "prompt": "Broken sketch fixture",
            "metadata": {"mock_artifact": "invalid_midi"},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "sketch_rejected"
    assert "MIDI sketch is not parseable" in detail["reason"]
    records = _artifact_records(tmp_path)
    assert records[0]["status"] == "rejected"


def test_text_to_midi_sketch_text2midi_missing_dependency_is_controlled(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        """
backends:
  text2midi:
    enabled: true
    type: symbolic
    adapter: model_backends.symbolic.text2midi_backend.Text2MidiBackend
    commercial_use: review_required
    dependency_mode: optional
    install_hint: install Text2MIDI in the optional model worker profile
    tasks:
      - generate_full_sketch
    capabilities:
      symbolic_midi: true
      text_prompt: true
      commercial_use: review_required
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "text2midi",
            "prompt": "Hard bop minor blues sketch",
        },
    )

    assert response.status_code == 409
    assert "Text2MIDI is not installed" in response.json()["detail"]


def test_text_to_midi_sketch_text2midi_subprocess_imports_sketch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(tmp_path / "api-storage"))
    fake_repo = tmp_path / "external" / "text2midi"
    (fake_repo / "model").mkdir(parents=True)
    (fake_repo / "model" / "transformer_model.py").write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints" / "text2midi"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "pytorch_model.bin").write_bytes(b"fake")
    (checkpoint_dir / "vocab_remi.pkl").write_bytes(b"fake")
    wrapper_path = tmp_path / "run_text2midi_inference.py"
    wrapper_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "ai_models.yaml"
    config_path.write_text(
        f"""
backends:
  text2midi:
    enabled: true
    type: symbolic
    adapter: model_backends.symbolic.text2midi_backend.Text2MidiBackend
    model_name: text2midi
    commercial_use: review_required
    dependency_mode: optional
    execution_mode: subprocess_or_worker
    install_hint: test fake Text2MIDI
    repo_dir: "{_yaml_path(fake_repo)}"
    checkpoint_dir: "{_yaml_path(checkpoint_dir)}"
    model_file: pytorch_model.bin
    tokenizer_file: vocab_remi.pkl
    wrapper_path: "{_yaml_path(wrapper_path)}"
    output_dir: "{_yaml_path(tmp_path / "raw")}"
    max_len: 64
    tasks:
      - generate_full_sketch
    capabilities:
      symbolic_midi: true
      text_prompt: true
      commercial_use: review_required
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_MODELS_CONFIG", str(config_path))
    required = {
        "torch",
        "transformers",
        "sentencepiece",
        "einops",
        "jsonlines",
        "accelerate",
        "st_moe_pytorch",
    }
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: object() if name in required else None,
    )

    def fake_run(cmd, *, cwd, text, capture_output, check, **kwargs):
        output_path = Path(cmd[cmd.index("--output") + 1])
        _write_valid_text2midi_sketch(output_path)
        if "--summary" in cmd:
            summary_path = Path(cmd[cmd.index("--summary") + 1])
            summary_path.write_text(
                json.dumps({"status": "ok", "output": str(output_path)}) + "\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = TestClient(app)

    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "text2midi",
            "prompt": "Hard bop minor blues in C minor with alto sax, piano, bass and drums",
            "seed": 2201,
        },
    )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["backend"] == "text2midi"
    assert payload["status"] in {"sketch_ready", "sketch_uncertain"}
    assert payload["artifact"]["status"] == "validated"
    sketch_id = payload["sketch_id"]
    assert not (tmp_path / "api-storage" / "projects" / sketch_id).exists()
    project_path = tmp_path / "api-storage" / "sketches" / sketch_id / "arrangement_project.json"
    project = ArrangementProject.load_json(project_path)
    assert project.metadata["auto_merge_allowed"] is False
    assert project.metadata["professional_project"] is False
    assert project.metadata["source_backend"] == "text2midi"
    assert {track.role for track in project.tracks} >= {
        "melody",
        "comping",
        "walking_bass",
        "drums",
    }
    drum_track = next(track for track in project.tracks if track.role == "drums")
    assert any(
        getattr(event, "annotations", {}).get("normalized_drum_pitch")
        for bar in drum_track.bars
        for event in bar.events
    )


def _artifact_records(tmp_path: Path) -> list[dict]:
    manifest_path = tmp_path / "api-storage" / "model_artifacts" / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload["artifacts"]


def _yaml_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_valid_text2midi_sketch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
    _append_note_track(midi_file, "Alto Sax Lead", channel=0, program=65, notes=[64, 67, 69, 72])
    _append_note_track(midi_file, "Piano Comp", channel=1, program=0, notes=[60, 64, 67, 64])
    _append_note_track(midi_file, "Walking Bass", channel=2, program=32, notes=[36, 38, 40, 43])
    _append_note_track(midi_file, "Drums", channel=9, program=None, notes=[36, 38, 46, 85])
    midi_file.save(path)


def _append_note_track(
    midi_file: mido.MidiFile,
    name: str,
    *,
    channel: int,
    program: int | None,
    notes: list[int],
) -> None:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=82, channel=channel, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=480))
    midi_file.tracks.append(track)
