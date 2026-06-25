from __future__ import annotations

import importlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import mido

from model_backends.artifact import artifact_path, artifact_record
from model_backends.base import ModelCapabilities, ModelGenerationRequest, ModelGenerationResult
from model_backends.errors import (
    ModelBackendUnavailableError,
    ModelGenerationError,
    UnsupportedModelTaskError,
)


class Text2MidiBackend:
    backend_id = "text2midi"
    backend_version = "0.1.0"
    unavailable_reason = "Text2MIDI is not installed"
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=False,
        bar_infill=False,
        track_generation=False,
        text_prompt=True,
        json_planning=False,
        token_output=False,
        supports_training=True,
        commercial_use="review_required",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        model_name: str = "text2midi",
        output_dir: str | Path = "outputs/model_artifacts/raw",
        install_hint: str | None = None,
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.install_hint = install_hint or "Install Text2MIDI in the model worker profile"
        self._engine: Any | None = None

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("text2midi") is not None
            or importlib.util.find_spec("text_to_midi") is not None
        )

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task != "generate_full_sketch":
            raise UnsupportedModelTaskError("Text2MIDI only supports generate_full_sketch")
        engine = self._load_engine()
        output_path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        generated = self._run_engine(engine, request=request, output_path=output_path)
        self._materialize_output(generated, output_path)
        if not output_path.exists():
            raise ModelGenerationError("Text2MIDI did not produce an output MIDI sketch")
        return ModelGenerationResult(
            backend_id=self.backend_id,
            task=request.task,
            artifacts=[
                artifact_record(
                    "midi",
                    output_path,
                    metadata={
                        "backend_id": self.backend_id,
                        "backend_version": self.backend_version,
                        "model_name": self.model_name,
                        "prompt": request.prompt or request.instruction or "",
                        "seed": request.seed,
                        "sketch_only": True,
                    },
                )
            ],
            confidence=None,
            raw_metadata={
                "backend_version": self.backend_version,
                "model_name": self.model_name,
                "reproducibility": "best_effort",
                "sketch_only": True,
            },
        )

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if not self.is_available():
            raise ModelBackendUnavailableError(
                f"Text2MIDI is not installed. Install hint: {self.install_hint}"
            )

        last_error: Exception | None = None
        for module_name in ("text2midi.inference", "text2midi", "text_to_midi"):
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                last_error = exc
                continue
            engine = self._engine_from_module(module)
            if engine is not None:
                self._engine = engine
                return engine

        raise ModelBackendUnavailableError(
            f"Text2MIDI inference API is unavailable. Install hint: {self.install_hint}"
        ) from last_error

    def _engine_from_module(self, module: Any) -> Any | None:
        for class_name in (
            "Text2MidiPipeline",
            "Text2MIDIPipeline",
            "TextToMidiPipeline",
            "Text2Midi",
            "Text2MIDI",
            "TextToMidi",
        ):
            engine_class = getattr(module, class_name, None)
            if engine_class is None:
                continue
            try:
                from_pretrained = getattr(engine_class, "from_pretrained", None)
                if callable(from_pretrained):
                    return from_pretrained(self.model_name)
                return engine_class(self.model_name)
            except TypeError:
                try:
                    return engine_class()
                except Exception as exc:
                    raise ModelGenerationError(
                        f"Could not initialize Text2MIDI engine {class_name}"
                    ) from exc
            except Exception as exc:
                raise ModelGenerationError(
                    f"Could not load Text2MIDI model {self.model_name!r}"
                ) from exc

        if any(callable(getattr(module, name, None)) for name in _GENERATION_METHODS):
            return module
        return None

    def _run_engine(
        self,
        engine: Any,
        *,
        request: ModelGenerationRequest,
        output_path: Path,
    ) -> Any:
        payload = {
            "prompt": request.prompt or request.instruction or "",
            "seed": request.seed,
            "temperature": request.temperature,
            "style": request.style,
            "output_path": str(output_path),
            "task": request.task,
        }
        for method_name in _GENERATION_METHODS:
            method = getattr(engine, method_name, None)
            if not callable(method):
                continue
            try:
                return method(**payload)
            except TypeError:
                try:
                    return method(
                        payload["prompt"],
                        output_path=str(output_path),
                        seed=request.seed,
                    )
                except TypeError:
                    try:
                        return method(payload["prompt"])
                    except TypeError:
                        try:
                            return method(payload)
                        except TypeError:
                            continue
            except Exception as exc:
                raise ModelGenerationError(f"Text2MIDI generation failed: {exc}") from exc
        raise ModelGenerationError("Text2MIDI engine exposes no supported generation method")

    def _materialize_output(self, generated: Any, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if generated is None:
            return
        if isinstance(generated, mido.MidiFile):
            generated.save(output_path)
            return
        if isinstance(generated, bytes):
            output_path.write_bytes(generated)
            return
        if isinstance(generated, str | Path):
            generated_path = Path(generated)
            if generated_path.exists():
                if generated_path.resolve() != output_path.resolve():
                    shutil.copy2(generated_path, output_path)
                return
        if isinstance(generated, dict):
            for key in ("midi_path", "path", "output_path"):
                value = generated.get(key)
                if value:
                    self._materialize_output(value, output_path)
                    return
            midi_bytes = generated.get("midi_bytes")
            if isinstance(midi_bytes, bytes):
                output_path.write_bytes(midi_bytes)
                return
        raise ModelGenerationError("Text2MIDI returned an unsupported artifact payload")


_GENERATION_METHODS = (
    "generate_full_sketch",
    "text_to_midi",
    "generate_midi",
    "generate",
)
