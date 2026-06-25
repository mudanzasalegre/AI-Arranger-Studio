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


class MidiGptBackend:
    backend_id = "midigpt"
    backend_version = "0.1.0"
    unavailable_reason = "MIDI-GPT is not installed"
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=True,
        bar_infill=True,
        track_generation=True,
        text_prompt=False,
        json_planning=False,
        token_output=False,
        supports_training=True,
        commercial_use="review_required",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        model_name: str = "yellow",
        output_dir: str | Path = "outputs/model_artifacts/raw",
        install_hint: str | None = None,
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.install_hint = install_hint or 'pip install "midigpt[inference]"'
        self._engine: Any | None = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("midigpt") is not None

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task not in {
            "infill_bars",
            "generate_track",
            "continue_section",
            "generate_variation",
        }:
            raise UnsupportedModelTaskError(f"MIDI-GPT does not support {request.task}")
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
            raise ModelGenerationError("MIDI-GPT did not produce an output MIDI artifact")
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
                        "track_id": request.track_id,
                        "bars": request.bars or [],
                        "instruction": request.instruction,
                        "density": request.density,
                        "temperature": request.temperature,
                        "seed": request.seed,
                        "context_midi_path": request.metadata.get("context_midi_path"),
                    },
                )
            ],
            confidence=None,
            raw_metadata={
                "backend_version": self.backend_version,
                "model_name": self.model_name,
                "reproducibility": "best_effort",
            },
        )

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if not self.is_available():
            raise ModelBackendUnavailableError(
                f"MIDI-GPT is not installed. Install hint: {self.install_hint}"
            )
        try:
            module = importlib.import_module("midigpt.inference.engine")
            engine_class = module.InferenceEngine
        except (ImportError, AttributeError) as exc:
            raise ModelBackendUnavailableError(
                f"MIDI-GPT inference engine is unavailable. Install hint: {self.install_hint}"
            ) from exc
        try:
            self._engine = engine_class.from_pretrained(self.model_name)
        except Exception as exc:
            raise ModelGenerationError(
                f"Could not load MIDI-GPT model {self.model_name!r}"
            ) from exc
        return self._engine

    def _run_engine(
        self,
        engine: Any,
        *,
        request: ModelGenerationRequest,
        output_path: Path,
    ) -> Any:
        payload = {
            "task": request.task,
            "context_midi_path": request.metadata.get("context_midi_path"),
            "target_track_id": request.track_id,
            "track_id": request.track_id,
            "bars": request.bars or [],
            "instruction": request.instruction or "",
            "density": request.density,
            "temperature": request.temperature,
            "seed": request.seed,
            "output_path": str(output_path),
            "song_plan": request.song_plan,
            "groove_map": request.groove_map,
            "role_intent": request.role_intent,
        }
        method_names = {
            "infill_bars": ("generate_infill", "infill_bars", "generate"),
            "generate_track": ("generate_track", "generate"),
            "continue_section": ("continue_section", "generate"),
            "generate_variation": ("generate_variation", "generate_infill", "generate"),
        }[request.task]
        for method_name in method_names:
            method = getattr(engine, method_name, None)
            if not callable(method):
                continue
            try:
                return method(**payload)
            except TypeError:
                try:
                    return method(payload)
                except TypeError:
                    continue
            except Exception as exc:
                raise ModelGenerationError(f"MIDI-GPT generation failed: {exc}") from exc
        raise ModelGenerationError("MIDI-GPT engine exposes no supported generation method")

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
        raise ModelGenerationError("MIDI-GPT returned an unsupported artifact payload")
