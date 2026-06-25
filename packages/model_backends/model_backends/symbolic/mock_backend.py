from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mido

from model_backends.artifact import artifact_path, artifact_record, write_json_artifact
from model_backends.base import (
    ModelCapabilities,
    ModelGenerationRequest,
    ModelGenerationResult,
    ModelTask,
)
from model_backends.errors import ModelGenerationError, UnsupportedModelTaskError

MOCK_BACKEND_VERSION = "0.1.0"
MIDI_TASKS: set[ModelTask] = {
    "generate_full_sketch",
    "generate_track",
    "infill_bars",
    "continue_section",
    "generate_variation",
}


class MockSymbolicBackend:
    backend_id = "mock_symbolic"
    backend_version = MOCK_BACKEND_VERSION
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=True,
        bar_infill=True,
        track_generation=True,
        text_prompt=True,
        json_planning=True,
        token_output=True,
        supports_training=False,
        commercial_use="allowed",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        output_dir: str | Path = "outputs/model_artifacts/raw",
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.output_dir = Path(output_dir)

    def is_available(self) -> bool:
        return True

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        mode = str(
            request.metadata.get("mock_artifact")
            or request.metadata.get("mock_mode")
            or ""
        )
        if request.metadata.get("simulate_error") or mode == "error":
            raise ModelGenerationError("MockSymbolicBackend simulated generation error")

        if request.task == "plan_song":
            artifact = self._write_json_plan(request)
        elif request.task in MIDI_TASKS:
            if mode == "invalid_midi":
                artifact = self._write_invalid_midi(request)
            elif mode == "empty_midi":
                artifact = self._write_empty_midi(request)
            elif mode == "unlabeled_midi":
                artifact = self._write_valid_midi(request, labeled=False)
            else:
                artifact = self._write_valid_midi(request)
        else:
            raise UnsupportedModelTaskError(f"MockSymbolicBackend does not support {request.task}")

        return ModelGenerationResult(
            backend_id=self.backend_id,
            task=request.task,
            artifacts=[artifact],
            confidence=0.1 if mode in {"invalid_midi", "empty_midi"} else 0.95,
            warnings=[f"mock_{mode}"] if mode in {"invalid_midi", "empty_midi"} else [],
            raw_metadata={
                "backend_version": self.backend_version,
                "mock_mode": mode or "default",
                "reproducibility": "deterministic",
            },
        )

    def _write_json_plan(self, request: ModelGenerationRequest):
        path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".json",
            request_id=request.request_id,
        )
        payload = {
            "schema_version": "0.1.0",
            "style": request.style or "hard_bop",
            "song_plan": request.song_plan or {},
            "section_plan": request.section_plan or {},
            "phrase_plan": request.phrase_plan or {},
            "groove_map": request.groove_map or {},
            "role_intent": request.role_intent or {},
            "generation_strategy": {
                "base": "mock",
                "forbid_audio_models": True,
            },
        }
        return write_json_artifact(
            path,
            payload,
            metadata={
                "backend_id": self.backend_id,
                "task": request.task,
                "valid": True,
            },
        )

    def _write_valid_midi(self, request: ModelGenerationRequest, *, labeled: bool = True):
        path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        track_name = f"{self.backend_id}:{request.task}" if labeled else "Model Output"
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        if labeled:
            track.append(mido.Message("program_change", program=65, channel=0, time=0))
        track.append(mido.Message("note_on", note=60, velocity=72, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
        track.append(mido.Message("note_on", note=63, velocity=70, channel=0, time=0))
        track.append(mido.Message("note_off", note=63, velocity=0, channel=0, time=480))
        midi_file.tracks.append(track)
        path.parent.mkdir(parents=True, exist_ok=True)
        midi_file.save(path)
        return artifact_record(
            "midi",
            path,
            metadata={
                "backend_id": self.backend_id,
                "task": request.task,
                "valid": True,
                "track_id": request.track_id,
                "bars": request.bars or [],
            },
        )

    def _write_invalid_midi(self, request: ModelGenerationRequest):
        path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a valid midi file")
        metadata = {
            "backend_id": self.backend_id,
            "task": request.task,
            "valid": False,
            "reason": "mock_invalid_midi",
        }
        (path.with_suffix(".metadata.json")).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return artifact_record("midi", path, metadata=metadata)

    def _write_empty_midi(self, request: ModelGenerationRequest):
        path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        track.append(
            mido.MetaMessage("track_name", name=f"{self.backend_id}:{request.task}:empty", time=0)
        )
        midi_file.tracks.append(track)
        path.parent.mkdir(parents=True, exist_ok=True)
        midi_file.save(path)
        return artifact_record(
            "midi",
            path,
            metadata={
                "backend_id": self.backend_id,
                "task": request.task,
                "valid": True,
                "empty": True,
                "track_id": request.track_id,
                "bars": request.bars or [],
            },
        )
