from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import mido

from model_backends.artifact import artifact_path, artifact_record
from model_backends.base import ModelCapabilities, ModelGenerationRequest, ModelGenerationResult
from model_backends.errors import (
    ModelBackendUnavailableError,
    ModelGenerationError,
    UnsupportedModelTaskError,
)


class _MidiGptApi(NamedTuple):
    Score: type[Any]
    InferenceEngine: type[Any]
    GenerationRequest: type[Any]
    InferenceConfig: type[Any]
    TrackPrompt: type[Any]


class MidiGptBackend:
    backend_id = "midigpt"
    backend_version = "0.2.0"
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
        model_dim: int = 8,
        top_p: float = 0.95,
        mask_mode: str = "attention",
        polyphony_hard_limit: int = 4,
        device: str | None = None,
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.install_hint = install_hint or 'pip install "midigpt[inference]"'
        self.model_dim = model_dim
        self.top_p = top_p
        self.mask_mode = mask_mode
        self.polyphony_hard_limit = polyphony_hard_limit
        self.device = _device_arg(device or os.environ.get("AI_DEVICE"))
        self._api: _MidiGptApi | None = None
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
                        "target_track_only": True,
                        "full_score_midi_path": str(_full_score_output_path(output_path)),
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
        api = self._load_api()
        try:
            kwargs = {"device": self.device} if self.device is not None else {}
            self._engine = api.InferenceEngine.from_pretrained(self.model_name, **kwargs)
        except Exception as exc:
            raise ModelGenerationError(
                f"Could not load MIDI-GPT model {self.model_name!r}"
            ) from exc
        return self._engine

    def _load_api(self) -> _MidiGptApi:
        if self._api is not None:
            return self._api
        try:
            midigpt_module = importlib.import_module("midigpt")
            inference_module = importlib.import_module("midigpt.inference")
            self._api = _MidiGptApi(
                Score=midigpt_module.Score,
                InferenceEngine=inference_module.InferenceEngine,
                GenerationRequest=inference_module.GenerationRequest,
                InferenceConfig=inference_module.InferenceConfig,
                TrackPrompt=inference_module.TrackPrompt,
            )
        except (ImportError, AttributeError) as exc:
            raise ModelBackendUnavailableError(
                f"MIDI-GPT inference API is unavailable. Install hint: {self.install_hint}"
            ) from exc
        return self._api

    def _run_engine(
        self,
        engine: Any,
        *,
        request: ModelGenerationRequest,
        output_path: Path,
    ) -> Any:
        api = self._load_api()
        context_midi_path = _context_midi_path(request)
        if not context_midi_path.exists():
            raise ModelGenerationError(f"MIDI-GPT context MIDI not found: {context_midi_path}")

        try:
            score = api.Score.from_midi(str(context_midi_path))
            track_index = resolve_track_index(
                score,
                request.track_id,
                context_midi_path=context_midi_path,
            )
            bar_indices = _bar_indices(request.bars)
            score_tracks = _score_tracks(score)
            generation_request = api.GenerationRequest(
                tracks=_track_prompts(
                    api,
                    track_count=len(score_tracks),
                    target_track_index=track_index,
                    bar_indices=bar_indices,
                ),
                config=api.InferenceConfig(
                    temperature=request.temperature,
                    seed=request.seed if request.seed is not None else -1,
                    top_p=self.top_p,
                    model_dim=_model_dim_for_bars(request.bars, default=self.model_dim),
                    mask_mode=self.mask_mode,
                    polyphony_hard_limit=self.polyphony_hard_limit,
                ),
            )
            result = engine.session(score, generation_request).run()
            full_score_output_path = _full_score_output_path(output_path)
            full_score_output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_midi(str(full_score_output_path))
            _extract_target_track_midi(
                full_score_output_path,
                output_path,
                target_track_index=track_index,
            )
        except ModelGenerationError:
            raise
        except Exception as exc:
            raise ModelGenerationError(f"MIDI-GPT generation failed: {exc}") from exc

        return output_path

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


def resolve_track_index(
    score: Any,
    track_id: str | None,
    *,
    context_midi_path: Path | None = None,
) -> int:
    if not track_id:
        raise ModelGenerationError("MIDI-GPT requires a target track_id")

    normalized = _normalize_identifier(track_id)
    tracks = _score_tracks(score)
    for index, track in enumerate(tracks):
        identifiers = {_normalize_identifier(value) for value in _track_identifiers(track)}
        if normalized in identifiers:
            return index

    if context_midi_path is not None:
        fallback = _resolve_track_index_from_context_midi(
            context_midi_path,
            normalized_track_id=normalized,
            score_track_count=len(tracks),
        )
        if fallback is not None:
            return fallback

    available = sorted(
        {
            identifier
            for track in tracks
            for identifier in _track_identifiers(track)
            if identifier
        }
    )
    raise ModelGenerationError(
        f"Could not resolve MIDI-GPT track_id {track_id!r}. "
        f"Available score tracks: {available}"
    )


def _context_midi_path(request: ModelGenerationRequest) -> Path:
    value = request.metadata.get("context_midi_path")
    if not value:
        raise ModelGenerationError("MIDI-GPT requires metadata.context_midi_path")
    return Path(str(value))


def _bar_indices(bars: list[int] | None) -> list[int]:
    if not bars:
        raise ModelGenerationError("MIDI-GPT requires one or more target bars")
    invalid = [bar for bar in bars if int(bar) < 1]
    if invalid:
        raise ModelGenerationError(f"MIDI-GPT bars must be 1-based positive integers: {invalid}")
    return [int(bar) - 1 for bar in bars]


def _track_prompts(
    api: _MidiGptApi,
    *,
    track_count: int,
    target_track_index: int,
    bar_indices: list[int],
) -> list[Any]:
    prompts = []
    for index in range(track_count):
        if index == target_track_index:
            prompts.append(api.TrackPrompt(id=index, bars=bar_indices))
        else:
            prompts.append(api.TrackPrompt(id=index, bars=[], ignore=True))
    return prompts


def _model_dim_for_bars(bars: list[int] | None, *, default: int) -> int:
    if not bars:
        return default
    return max(default, max(int(bar) for bar in bars))


def _score_tracks(score: Any) -> list[Any]:
    tracks = getattr(score, "tracks", None)
    if tracks is None and isinstance(score, dict):
        tracks = score.get("tracks")
    if tracks is None:
        raise ModelGenerationError("MIDI-GPT Score has no tracks collection")
    return list(tracks)


def _track_identifiers(track: Any) -> list[str]:
    identifiers: list[str] = []
    for attr in ("id", "track_id", "name", "track_name", "instrument", "part_name"):
        value = _track_value(track, attr)
        if value is not None:
            identifiers.append(str(value))

    metadata = _track_value(track, "metadata")
    if isinstance(metadata, dict):
        for key in ("id", "track_id", "name", "track_name", "instrument"):
            value = metadata.get(key)
            if value is not None:
                identifiers.append(str(value))
    return identifiers


def _track_value(track: Any, key: str) -> Any:
    if isinstance(track, dict):
        return track.get(key)
    return getattr(track, key, None)


def _resolve_track_index_from_context_midi(
    context_midi_path: Path,
    *,
    normalized_track_id: str,
    score_track_count: int,
) -> int | None:
    try:
        midi_file = mido.MidiFile(context_midi_path)
    except Exception:
        return None

    midi_tracks = list(enumerate(midi_file.tracks))
    non_conductor_tracks = [
        (index, track)
        for index, track in midi_tracks
        if _normalize_identifier(_midi_track_name(track)) != "conductor"
    ]
    for midi_index, track in non_conductor_tracks:
        if _normalize_identifier(_midi_track_name(track)) != normalized_track_id:
            continue
        if score_track_count == len(non_conductor_tracks):
            return non_conductor_tracks.index((midi_index, track))
        if score_track_count == len(midi_tracks):
            return midi_index
    return None


def _full_score_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.full{output_path.suffix}")


def _extract_target_track_midi(
    source_path: Path,
    output_path: Path,
    *,
    target_track_index: int,
) -> None:
    midi_file = mido.MidiFile(source_path)
    if target_track_index < 0 or target_track_index >= len(midi_file.tracks):
        raise ModelGenerationError(
            f"MIDI-GPT output has {len(midi_file.tracks)} tracks; "
            f"target track index {target_track_index} is missing"
        )

    target_only = mido.MidiFile(type=1, ticks_per_beat=midi_file.ticks_per_beat)
    target_track = mido.MidiTrack()
    target_track.append(mido.MetaMessage("track_name", name="midigpt:target", time=0))
    for message in midi_file.tracks[target_track_index]:
        if getattr(message, "type", None) == "track_name":
            continue
        target_track.append(message.copy())
    target_only.tracks.append(target_track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_only.save(output_path)


def _midi_track_name(track: mido.MidiTrack) -> str:
    for message in track:
        if getattr(message, "type", None) == "track_name":
            return str(message.name)
    return ""


def _normalize_identifier(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _device_arg(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return None
    return normalized
