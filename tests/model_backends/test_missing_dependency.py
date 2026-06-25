from __future__ import annotations

import importlib.util
import sys
import types

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


def test_midigpt_backend_materializes_fake_engine_output(tmp_path, monkeypatch):
    from model_backends.symbolic.midigpt_backend import MidiGptBackend

    class FakeInferenceEngine:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "yellow"
            return cls()

        def generate_infill(self, **kwargs):
            assert kwargs["target_track_id"] == "alto_sax"
            midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=64, velocity=80, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, time=480))
            midi_file.tracks.append(track)
            return midi_file

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: object())
    midigpt_module = types.ModuleType("midigpt")
    inference_module = types.ModuleType("midigpt.inference")
    engine_module = types.ModuleType("midigpt.inference.engine")
    engine_module.InferenceEngine = FakeInferenceEngine
    monkeypatch.setitem(sys.modules, "midigpt", midigpt_module)
    monkeypatch.setitem(sys.modules, "midigpt.inference", inference_module)
    monkeypatch.setitem(sys.modules, "midigpt.inference.engine", engine_module)

    backend = MidiGptBackend(output_dir=tmp_path)
    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="fake-engine",
            track_id="alto_sax",
            bars=[1],
            instruction="bebop phrase",
            seed=77,
            metadata={"context_midi_path": "context.mid"},
        )
    )

    artifact = result.artifacts[0]
    assert artifact.artifact_type == "midi"
    assert artifact.metadata["model_name"] == "yellow"
    assert artifact.metadata["track_id"] == "alto_sax"
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


def test_text2midi_backend_materializes_fake_engine_output(tmp_path, monkeypatch):
    from model_backends.symbolic.text2midi_backend import Text2MidiBackend

    class FakeText2MidiPipeline:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "text2midi"
            return cls()

        def generate_full_sketch(self, **kwargs):
            assert kwargs["prompt"] == "hard bop sketch"
            midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name="Alto Sax Lead", time=0))
            track.append(mido.Message("program_change", program=65, channel=0, time=0))
            track.append(mido.Message("note_on", note=64, velocity=80, channel=0, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))
            midi_file.tracks.append(track)
            return midi_file

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: object() if name == "text2midi" else None,
    )
    text2midi_module = types.ModuleType("text2midi")
    text2midi_module.Text2MidiPipeline = FakeText2MidiPipeline
    monkeypatch.setitem(sys.modules, "text2midi", text2midi_module)

    backend = Text2MidiBackend(output_dir=tmp_path)
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
    artifact_path = tmp_path / "text2midi_generate_full_sketch_fake-text2midi.mid"
    assert artifact_path.read_bytes()[:4] == b"MThd"
