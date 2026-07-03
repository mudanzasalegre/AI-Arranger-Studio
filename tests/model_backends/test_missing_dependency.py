from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import mido
import pytest
from model_backends import (
    ModelBackendUnavailableError,
    ModelGenerationRequest,
    build_model_backend_registry,
    load_ai_models_config,
)


def test_enabled_midigpt_missing_dependency_is_unavailable_without_import(
    tmp_path,
    monkeypatch,
):
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
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"midigpt", "torch", "transformers"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    was_midigpt_loaded = "midigpt" in sys.modules
    config = load_ai_models_config(config_path)
    registry = build_model_backend_registry(config=config, include_unavailable=True)

    models = registry.list()
    assert models[0]["id"] == "midigpt"
    assert models[0]["status"] == "unavailable"
    assert "MIDI-GPT is not installed" in models[0]["error"]
    if not was_midigpt_loaded:
        assert "midigpt" not in sys.modules
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules

    with pytest.raises(ModelBackendUnavailableError):
        registry.get("midigpt")


def test_midigpt_backend_generate_reports_controlled_missing_dependency(monkeypatch):
    from model_backends.symbolic.midigpt_backend import MidiGptBackend

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    backend = MidiGptBackend()

    with pytest.raises(ModelBackendUnavailableError):
        backend.generate(ModelGenerationRequest(task="infill_bars"))


def test_midigpt_backend_uses_real_session_api(tmp_path, monkeypatch):
    from model_backends.symbolic.midigpt_backend import MidiGptBackend

    calls = {}
    context_path = tmp_path / "context.mid"
    _write_context_midi(context_path, ["Conductor", "alto_sax"])

    class FakeTrack:
        def __init__(self, name):
            self.name = name

    class FakeScore:
        def __init__(self, tracks):
            self.tracks = tracks

        @classmethod
        def from_midi(cls, path):
            calls["context_path"] = path
            return cls([FakeTrack("alto_sax")])

    class FakeInferenceConfig:
        def __init__(self, **kwargs):
            calls["inference_config"] = kwargs

    class FakeTrackPrompt:
        def __init__(self, *, id, bars, ignore=False):
            calls["track_prompt"] = {"id": id, "bars": bars, "ignore": ignore}

    class FakeGenerationRequest:
        def __init__(self, *, tracks, config):
            calls["generation_request"] = {"tracks": tracks, "config": config}

    class FakeResult:
        def to_midi(self, path):
            calls["to_midi_path"] = path
            midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=64, velocity=80, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, time=480))
            midi_file.tracks.append(track)
            midi_file.save(path)

    class FakeSession:
        def run(self):
            calls["run"] = True
            return FakeResult()

    class FakeInferenceEngine:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "yellow"
            calls["model_name"] = model_name
            return cls()

        def session(self, score, request):
            calls["session"] = {"score": score, "request": request}
            return FakeSession()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: object())
    midigpt_module = types.ModuleType("midigpt")
    midigpt_module.Score = FakeScore
    inference_module = types.ModuleType("midigpt.inference")
    inference_module.InferenceEngine = FakeInferenceEngine
    inference_module.GenerationRequest = FakeGenerationRequest
    inference_module.InferenceConfig = FakeInferenceConfig
    inference_module.TrackPrompt = FakeTrackPrompt
    monkeypatch.setitem(sys.modules, "midigpt", midigpt_module)
    monkeypatch.setitem(sys.modules, "midigpt.inference", inference_module)

    backend = MidiGptBackend(output_dir=tmp_path)
    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="fake-engine",
            track_id="alto_sax",
            bars=[9],
            instruction="bebop phrase",
            seed=77,
            metadata={"context_midi_path": str(context_path)},
        )
    )

    artifact = result.artifacts[0]
    assert artifact.artifact_type == "midi"
    assert artifact.metadata["model_name"] == "yellow"
    assert artifact.metadata["track_id"] == "alto_sax"
    assert calls["context_path"] == str(context_path)
    assert calls["track_prompt"] == {"id": 0, "bars": [8], "ignore": False}
    assert calls["inference_config"] == {
        "temperature": 0.8,
        "seed": 77,
        "top_p": 0.95,
        "model_dim": 9,
        "mask_mode": "attention",
        "polyphony_hard_limit": 4,
    }
    assert calls["run"] is True
    assert calls["to_midi_path"] == str(tmp_path / "midigpt_infill_bars_fake-engine.full.mid")
    assert (tmp_path / "midigpt_infill_bars_fake-engine.mid").read_bytes()[:4] == b"MThd"


def test_enabled_text2midi_missing_dependency_is_unavailable_without_import(
    tmp_path,
    monkeypatch,
):
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
settings:
  artifact_raw_dir: outputs/model_artifacts/raw
""".lstrip(),
        encoding="utf-8",
    )

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"text2midi", "text_to_midi", "torch", "transformers"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    was_text2midi_loaded = "text2midi" in sys.modules
    config = load_ai_models_config(config_path)
    registry = build_model_backend_registry(config=config, include_unavailable=True)

    models = registry.list()
    assert models[0]["id"] == "text2midi"
    assert models[0]["status"] == "unavailable"
    assert "Text2MIDI is not installed" in models[0]["error"]
    if not was_text2midi_loaded:
        assert "text2midi" not in sys.modules
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules

    with pytest.raises(ModelBackendUnavailableError):
        registry.get("text2midi")


def test_text2midi_backend_generate_reports_controlled_missing_dependency(monkeypatch):
    from model_backends.symbolic.text2midi_backend import Text2MidiBackend

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    backend = Text2MidiBackend()

    with pytest.raises(ModelBackendUnavailableError):
        backend.generate(ModelGenerationRequest(task="generate_full_sketch"))


def test_text2midi_backend_runs_subprocess_wrapper(tmp_path, monkeypatch):
    from model_backends.symbolic.text2midi_backend import Text2MidiBackend

    fake_repo = tmp_path / "text2midi"
    (fake_repo / "model").mkdir(parents=True)
    (fake_repo / "model" / "transformer_model.py").write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "pytorch_model.bin").write_bytes(b"fake")
    (checkpoint_dir / "vocab_remi.pkl").write_bytes(b"fake")
    wrapper_path = tmp_path / "run_text2midi_inference.py"
    wrapper_path.write_text("", encoding="utf-8")

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
    calls = {}

    def fake_run(cmd, *, cwd, text, capture_output, check):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["text"] = text
        calls["capture_output"] = capture_output
        calls["check"] = check
        output_path = Path(cmd[cmd.index("--output") + 1])
        _write_context_midi(output_path, ["Alto Sax Lead"])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = Text2MidiBackend(
        output_dir=tmp_path,
        repo_dir=fake_repo,
        checkpoint_dir=checkpoint_dir,
        wrapper_path=wrapper_path,
    )
    result = backend.generate(
        ModelGenerationRequest(
            task="generate_full_sketch",
            request_id="fake-text2midi",
            prompt="hard bop sketch",
            seed=88,
        )
    )

    artifact = result.artifacts[0]
    assert artifact.artifact_type == "midi"
    assert artifact.metadata["model_name"] == "text2midi"
    assert artifact.metadata["sketch_only"] is True
    assert artifact.metadata["execution_mode"] == "subprocess"
    assert calls["cmd"][0] == sys.executable
    assert calls["cmd"][calls["cmd"].index("--prompt") + 1] == "hard bop sketch"
    assert calls["cmd"][calls["cmd"].index("--seed") + 1] == "88"
    artifact_path = tmp_path / "text2midi_generate_full_sketch_fake-text2midi.mid"
    assert artifact_path.read_bytes()[:4] == b"MThd"


def _write_context_midi(path, track_names: list[str]) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
    for name in track_names:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if name != "Conductor":
            track.append(mido.Message("note_on", note=64, velocity=80, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, time=480))
        midi_file.tracks.append(track)
    midi_file.save(path)
