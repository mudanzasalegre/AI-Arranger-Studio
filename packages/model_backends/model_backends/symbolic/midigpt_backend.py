from __future__ import annotations

import importlib
import importlib.util
import inspect
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
    backend_version = "0.3.0"
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
        max_attempts: int = 3,
        temperature_schedule: list[float] | None = None,
        top_p_schedule: list[float] | None = None,
        default_inference: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        defaults = default_inference or {}
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.install_hint = install_hint or 'pip install "midigpt[inference]"'
        self.model_dim = int(defaults.get("model_dim", model_dim))
        self.top_p = float(defaults.get("top_p", top_p))
        self.mask_mode = str(defaults.get("mask_mode", mask_mode))
        self.polyphony_hard_limit = int(
            defaults.get("polyphony_hard_limit", polyphony_hard_limit)
        )
        self.max_attempts = int(defaults.get("max_attempts", max_attempts))
        self.temperature_schedule = _float_schedule(
            defaults.get("temperature_schedule", temperature_schedule),
            fallback=[1.0, 0.85, 0.7],
        )
        self.top_p_schedule = _float_schedule(
            defaults.get("top_p_schedule", top_p_schedule),
            fallback=[self.top_p, 0.9, 0.85],
        )
        self.device = _device_arg(device or os.environ.get("AI_DEVICE"))
        self._api: _MidiGptApi | None = None
        self._engine: Any | None = None
        self._last_generation_trace: dict[str, Any] = {}

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
        generation_trace = dict(self._last_generation_trace)
        full_score_paths = list(generation_trace.get("full_score_midi_paths") or [])
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
                        "full_score_midi_path": (
                            full_score_paths[0]
                            if full_score_paths
                            else str(_full_score_output_path(output_path))
                        ),
                        "full_score_midi_paths": full_score_paths,
                        "midigpt_trace": generation_trace,
                    },
                )
            ],
            confidence=None,
            raw_metadata={
                "backend_version": self.backend_version,
                "model_name": self.model_name,
                "reproducibility": "best_effort",
                "midigpt_trace": generation_trace,
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
                project=request.project,
                context_midi_path=context_midi_path,
                metadata=request.metadata,
            )
            bar_indices = _bar_indices(request.bars)
            chunks = _chunk_bar_indices(bar_indices)
            requested_attributes = _attributes_for_request(request)
            applied_attributes = _compatible_track_attributes(
                engine,
                score,
                target_track_index=track_index,
                attributes=requested_attributes,
            )
            trace = self._generation_trace_template(
                request=request,
                target_track_index=track_index,
                bar_indices=bar_indices,
                chunks=chunks,
                requested_attributes=requested_attributes,
                applied_attributes=applied_attributes,
                density_hard_limit=_density_hard_limit(
                    request,
                    requested_attributes=requested_attributes,
                    applied_attributes=applied_attributes,
                ),
            )
            last_error: Exception | None = None
            for attempt_index, (temperature, top_p) in enumerate(
                self._attempt_schedule(request),
                start=1,
            ):
                try:
                    chunk_outputs = self._run_chunks(
                        api,
                        engine,
                        score,
                        request=request,
                        output_path=output_path,
                        target_track_index=track_index,
                        chunks=chunks,
                        attempt_index=attempt_index,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    _combine_target_chunk_midis(chunk_outputs, output_path)
                    note_count = _midi_note_count(output_path)
                    if note_count <= 0:
                        raise ModelGenerationError(
                            "MIDI-GPT produced no target notes after extraction"
                        )
                    full_score_paths = [
                        str(
                            _chunk_full_score_output_path(
                                output_path,
                                attempt_index=attempt_index,
                                chunk_index=chunk_index,
                            )
                        )
                        for chunk_index, _chunk in enumerate(chunks, start=1)
                    ]
                    _validate_target_midi_duration(
                        output_path,
                        target_bars=request.bars or [],
                        project=request.project,
                    )
                    trace["attempts"].append(
                        {
                            "attempt": attempt_index,
                            "status": "ok",
                            "temperature": temperature,
                            "top_p": top_p,
                            "note_count": note_count,
                            "chunk_outputs": [str(path) for path in chunk_outputs],
                            "full_score_midi_paths": full_score_paths,
                        }
                    )
                    trace["status"] = "ok"
                    trace["note_count"] = note_count
                    trace["chunk_outputs"] = [str(path) for path in chunk_outputs]
                    trace["full_score_midi_paths"] = full_score_paths
                    self._last_generation_trace = trace
                    break
                except Exception as exc:
                    last_error = exc
                    trace["attempts"].append(
                        {
                            "attempt": attempt_index,
                            "status": "fail",
                            "temperature": temperature,
                            "top_p": top_p,
                            "error": str(exc),
                        }
                    )
            else:
                self._last_generation_trace = trace
                raise ModelGenerationError(
                    f"MIDI-GPT generation failed after {len(trace['attempts'])} attempts: "
                    f"{last_error}"
                ) from last_error
        except ModelGenerationError:
            raise
        except Exception as exc:
            raise ModelGenerationError(f"MIDI-GPT generation failed: {exc}") from exc

        return output_path

    def _run_chunks(
        self,
        api: _MidiGptApi,
        engine: Any,
        score: Any,
        *,
        request: ModelGenerationRequest,
        output_path: Path,
        target_track_index: int,
        chunks: list[list[int]],
        attempt_index: int,
        temperature: float,
        top_p: float,
    ) -> list[Path]:
        score_tracks = _score_tracks(score)
        requested_attributes = _attributes_for_request(request)
        applied_attributes = _compatible_track_attributes(
            engine,
            score,
            target_track_index=target_track_index,
            attributes=requested_attributes,
        )
        density_hard_limit = _density_hard_limit(
            request,
            requested_attributes=requested_attributes,
            applied_attributes=applied_attributes,
        )
        chunk_outputs: list[Path] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_full_path = _chunk_full_score_output_path(
                output_path,
                attempt_index=attempt_index,
                chunk_index=chunk_index,
            )
            chunk_target_path = _chunk_target_output_path(
                output_path,
                attempt_index=attempt_index,
                chunk_index=chunk_index,
            )
            config_kwargs = {
                "temperature": temperature,
                "seed": request.seed if request.seed is not None else -1,
                "top_p": top_p,
                "model_dim": _model_dim_for_bar_indices(chunk, default=self.model_dim),
                "mask_mode": str(request.metadata.get("mask_mode") or self.mask_mode),
                "polyphony_hard_limit": self._polyphony_limit(request),
            }
            if density_hard_limit is not None:
                config_kwargs["density_hard_limit"] = density_hard_limit
            inference_config = _construct_supported(
                api.InferenceConfig,
                **config_kwargs,
            )
            generation_request = _construct_supported(
                api.GenerationRequest,
                tracks=_track_prompts(
                    api,
                    track_count=len(score_tracks),
                    target_track_index=target_track_index,
                    bar_indices=chunk,
                    task=request.task,
                    attributes=applied_attributes,
                ),
                config=inference_config,
            )
            result = engine.session(score, generation_request).run()
            chunk_full_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_midi(str(chunk_full_path))
            _extract_target_track_midi(
                chunk_full_path,
                chunk_target_path,
                target_track_index=target_track_index,
                target_bar_indices=chunk,
                project=request.project,
            )
            chunk_outputs.append(chunk_target_path)
        return chunk_outputs

    def _attempt_schedule(self, request: ModelGenerationRequest) -> list[tuple[float, float]]:
        temperatures = _float_schedule(
            request.metadata.get("temperature_schedule"),
            fallback=self.temperature_schedule,
        )
        top_ps = _float_schedule(
            request.metadata.get("top_p_schedule"),
            fallback=self.top_p_schedule,
        )
        if request.metadata.get("temperature") is not None:
            temperatures[0] = float(request.metadata["temperature"])
        if request.metadata.get("top_p") is not None:
            top_ps[0] = float(request.metadata["top_p"])
        pairs: list[tuple[float, float]] = []
        for index in range(max(1, self.max_attempts)):
            pairs.append(
                (
                    float(temperatures[min(index, len(temperatures) - 1)]),
                    float(top_ps[min(index, len(top_ps) - 1)]),
                )
            )
        return pairs[: max(1, self.max_attempts)]

    def _polyphony_limit(self, request: ModelGenerationRequest) -> int:
        configured = request.metadata.get("polyphony_hard_limit")
        if configured is not None:
            return int(configured)
        role = _normalized_text(
            request.metadata.get("target_role")
            or (request.role_intent or {}).get("role")
            or ""
        )
        instrument = _normalized_text(
            request.metadata.get("target_instrument")
            or (request.role_intent or {}).get("instrument")
            or ""
        )
        if "drum" in role or "drum" in instrument:
            return 8
        if "piano" in role or "comping" in role or "piano" in instrument:
            return 8
        if "bass" in role or "bass" in instrument:
            return 2
        if role or instrument:
            return min(self.polyphony_hard_limit, 4)
        return self.polyphony_hard_limit

    def _generation_trace_template(
        self,
        *,
        request: ModelGenerationRequest,
        target_track_index: int,
        bar_indices: list[int],
        chunks: list[list[int]],
        requested_attributes: dict[str, int],
        applied_attributes: dict[str, int],
        density_hard_limit: int | None,
    ) -> dict[str, Any]:
        return {
            "status": "pending",
            "api": "engine.session(score, generation_request).run",
            "model_name": self.model_name,
            "task": request.task,
            "track_id": request.track_id,
            "target_track_index": target_track_index,
            "target_bars_1_based": request.bars or [],
            "target_bars_0_based": bar_indices,
            "chunks_0_based": chunks,
            "attributes": applied_attributes,
            "attributes_requested": requested_attributes,
            "attributes_applied": applied_attributes,
            "density_hard_limit": density_hard_limit,
            "autoregressive": request.task == "generate_track",
            "attempts": [],
        }

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
    project: dict[str, Any] | None = None,
    context_midi_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    if not track_id:
        raise ModelGenerationError("MIDI-GPT requires a target track_id")

    normalized = _normalize_identifier(track_id)
    tracks = _score_tracks(score)
    for index, track in enumerate(tracks):
        identifiers = {_normalize_identifier(value) for value in _track_identifiers(track)}
        if normalized in identifiers:
            return index

    project_track = _project_track(project, track_id)
    if project_track is not None:
        project_identifiers = _project_track_identifiers(project_track)
        for index, track in enumerate(tracks):
            identifiers = {_normalize_identifier(value) for value in _track_identifiers(track)}
            if identifiers & project_identifiers:
                return index

    if context_midi_path is not None:
        fallback = _resolve_track_index_from_context_midi(
            context_midi_path,
            normalized_track_id=normalized,
            score_track_count=len(tracks),
        )
        if fallback is not None:
            return fallback

    metadata = metadata or {}
    context_track_map = metadata.get("context_track_map")
    if isinstance(context_track_map, dict):
        mapped = context_track_map.get(track_id)
        if mapped is not None and context_midi_path is not None:
            fallback = _resolve_track_index_from_context_midi(
                context_midi_path,
                normalized_track_id=_normalize_identifier(str(mapped)),
                score_track_count=len(tracks),
            )
            if fallback is not None:
                return fallback

    if project_track is not None:
        program = _midi_program_for_project_track(project_track)
        if program is not None:
            matches = [
                index
                for index, track in enumerate(tracks)
                if _track_value(track, "instrument") == program
            ]
            if len(matches) == 1:
                return matches[0]
        ordered_index = _project_track_order(project, track_id)
        if ordered_index is not None and 0 <= ordered_index < len(tracks):
            return ordered_index

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
    task: str,
    attributes: dict[str, int],
) -> list[Any]:
    prompts = []
    for index in range(track_count):
        if index == target_track_index:
            prompts.append(
                _construct_supported(
                    api.TrackPrompt,
                    id=index,
                    bars=bar_indices,
                    autoregressive=task == "generate_track",
                    attributes=attributes,
                    bar_attributes={bar: attributes for bar in bar_indices},
                )
            )
        else:
            prompts.append(_construct_supported(api.TrackPrompt, id=index, bars=[], ignore=True))
    return prompts


def _model_dim_for_bar_indices(bar_indices: list[int], *, default: int) -> int:
    if not bar_indices:
        return default
    span = max(bar_indices) - min(bar_indices) + 1
    if span <= 4:
        return 4
    return 8


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


def _project_track(project: dict[str, Any] | None, track_id: str) -> dict[str, Any] | None:
    if not isinstance(project, dict):
        return None
    tracks = project.get("tracks")
    if not isinstance(tracks, list):
        return None
    for track in tracks:
        if isinstance(track, dict) and track.get("id") == track_id:
            return track
    return None


def _project_track_order(project: dict[str, Any] | None, track_id: str) -> int | None:
    if not isinstance(project, dict):
        return None
    tracks = project.get("tracks")
    if not isinstance(tracks, list):
        return None
    ordered = [track for track in tracks if isinstance(track, dict)]
    for index, track in enumerate(ordered):
        if track.get("id") == track_id:
            return index
    return None


def _project_track_identifiers(track: dict[str, Any]) -> set[str]:
    values = [
        track.get("id"),
        track.get("name"),
        track.get("instrument"),
        track.get("role"),
    ]
    metadata = track.get("metadata")
    if isinstance(metadata, dict):
        values.extend(
            metadata.get(key)
            for key in (
                "id",
                "track_id",
                "name",
                "track_name",
                "instrument",
                "model_context_track_name",
            )
        )
    return {_normalize_identifier(str(value)) for value in values if value}


def _midi_program_for_project_track(track: dict[str, Any]) -> int | None:
    metadata = track.get("metadata")
    if isinstance(metadata, dict) and metadata.get("midi_program") is not None:
        return int(metadata["midi_program"])
    instrument = str(track.get("instrument") or "")
    return _INSTRUMENT_MIDI_PROGRAMS.get(instrument)


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


def _chunk_full_score_output_path(
    output_path: Path,
    *,
    attempt_index: int,
    chunk_index: int,
) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.attempt{attempt_index}.chunk{chunk_index}.full{output_path.suffix}"
    )


def _chunk_target_output_path(
    output_path: Path,
    *,
    attempt_index: int,
    chunk_index: int,
) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.attempt{attempt_index}.chunk{chunk_index}{output_path.suffix}"
    )


def _extract_target_track_midi(
    source_path: Path,
    output_path: Path,
    *,
    target_track_index: int,
    target_bar_indices: list[int] | None = None,
    project: dict[str, Any] | None = None,
) -> None:
    midi_file = mido.MidiFile(source_path)
    if target_track_index < 0 or target_track_index >= len(midi_file.tracks):
        raise ModelGenerationError(
            f"MIDI-GPT output has {len(midi_file.tracks)} tracks; "
            f"target track index {target_track_index} is missing"
        )

    target_only = mido.MidiFile(type=1, ticks_per_beat=midi_file.ticks_per_beat)
    target_track = _target_track_window(
        midi_file.tracks[target_track_index],
        target_bar_indices=target_bar_indices or [],
        ticks_per_beat=midi_file.ticks_per_beat,
        project=project,
    )
    target_only.tracks.append(target_track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_only.save(output_path)


def _target_track_window(
    source_track: mido.MidiTrack,
    *,
    target_bar_indices: list[int],
    ticks_per_beat: int,
    project: dict[str, Any] | None,
) -> mido.MidiTrack:
    target_track = mido.MidiTrack()
    target_track.append(mido.MetaMessage("track_name", name="midigpt:target", time=0))
    notes = _track_notes(source_track)
    if not target_bar_indices:
        return _track_from_notes(target_track, notes)

    windows = _target_windows(
        target_bar_indices,
        ticks_per_beat=ticks_per_beat,
        project=project,
    )
    shifted_notes: list[dict[str, int]] = []
    output_cursor = 0
    for bar_index, start_tick, end_tick, duration_ticks in windows:
        for note in notes:
            if start_tick <= note["start_tick"] < end_tick:
                shifted_start = output_cursor + max(0, note["start_tick"] - start_tick)
                shifted_end = output_cursor + min(
                    duration_ticks,
                    max(1, note["end_tick"] - start_tick),
                )
                if shifted_end <= shifted_start:
                    shifted_end = shifted_start + max(1, int(ticks_per_beat / 4))
                shifted_notes.append(
                    {
                        **note,
                        "start_tick": shifted_start,
                        "end_tick": shifted_end,
                        "source_bar_index": bar_index,
                    }
                )
        output_cursor += duration_ticks

    if shifted_notes:
        return _track_from_notes(target_track, shifted_notes)
    return _track_from_notes(target_track, notes)


def _target_windows(
    target_bar_indices: list[int],
    *,
    ticks_per_beat: int,
    project: dict[str, Any] | None,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for bar_index in target_bar_indices:
        start_beat = _beats_before_bar(project, bar_index + 1)
        duration_beats = _bar_duration_beats(project, bar_index + 1)
        start_tick = int(round(start_beat * ticks_per_beat))
        duration_ticks = int(round(duration_beats * ticks_per_beat))
        windows.append((bar_index, start_tick, start_tick + duration_ticks, duration_ticks))
    return windows


def _beats_before_bar(project: dict[str, Any] | None, bar_number: int) -> float:
    return sum(_bar_duration_beats(project, current) for current in range(1, bar_number))


def _bar_duration_beats(project: dict[str, Any] | None, bar_number: int) -> float:
    meter = "4/4"
    if isinstance(project, dict):
        generation_spec = project.get("generation_spec")
        if isinstance(generation_spec, dict) and generation_spec.get("meter"):
            meter = str(generation_spec["meter"])
        meter_map = project.get("meter_map")
        if isinstance(meter_map, list):
            for marker in sorted(
                (item for item in meter_map if isinstance(item, dict)),
                key=lambda item: int(item.get("bar", 1)),
            ):
                if int(marker.get("bar", 1)) <= bar_number and marker.get("meter"):
                    meter = str(marker["meter"])
    try:
        numerator, denominator = meter.split("/", maxsplit=1)
        return int(numerator) * (4 / int(denominator))
    except (ValueError, ZeroDivisionError):
        return 4.0


def _track_notes(track: mido.MidiTrack) -> list[dict[str, int]]:
    notes: list[dict[str, int]] = []
    active: dict[tuple[int, int], tuple[int, int]] = {}
    absolute_tick = 0
    for message in track:
        absolute_tick += int(getattr(message, "time", 0))
        if getattr(message, "type", None) == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)] = (absolute_tick, message.velocity)
        elif getattr(message, "type", None) in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            if key not in active:
                continue
            start_tick, velocity = active.pop(key)
            notes.append(
                {
                    "channel": int(message.channel),
                    "note": int(message.note),
                    "velocity": int(velocity),
                    "start_tick": int(start_tick),
                    "end_tick": int(max(start_tick + 1, absolute_tick)),
                }
            )
    notes.sort(key=lambda item: (item["start_tick"], item["note"]))
    return notes


def _track_from_notes(
    target_track: mido.MidiTrack,
    notes: list[dict[str, int]],
) -> mido.MidiTrack:
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        events.append(
            (
                int(note["start_tick"]),
                0,
                mido.Message(
                    "note_on",
                    channel=int(note["channel"]),
                    note=int(note["note"]),
                    velocity=max(1, min(127, int(note["velocity"]))),
                    time=0,
                ),
            )
        )
        events.append(
            (
                int(note["end_tick"]),
                1,
                mido.Message(
                    "note_off",
                    channel=int(note["channel"]),
                    note=int(note["note"]),
                    velocity=0,
                    time=0,
                ),
            )
        )
    current = 0
    for absolute_tick, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        target_track.append(message.copy(time=max(0, absolute_tick - current)))
        current = absolute_tick
    return target_track


def _combine_target_chunk_midis(chunk_paths: list[Path], output_path: Path) -> None:
    if len(chunk_paths) == 1:
        shutil.copy2(chunk_paths[0], output_path)
        return

    combined = mido.MidiFile(type=1, ticks_per_beat=480)
    combined_track = mido.MidiTrack()
    combined_track.append(mido.MetaMessage("track_name", name="midigpt:target", time=0))
    cursor = 0
    all_notes: list[dict[str, int]] = []
    for chunk_path in chunk_paths:
        midi_file = mido.MidiFile(chunk_path)
        if midi_file.ticks_per_beat:
            combined.ticks_per_beat = midi_file.ticks_per_beat
        notes = _track_notes(midi_file.tracks[0]) if midi_file.tracks else []
        all_notes.extend(
            {
                **note,
                "start_tick": int(note["start_tick"]) + cursor,
                "end_tick": int(note["end_tick"]) + cursor,
            }
            for note in notes
        )
        cursor += _midi_length_ticks(midi_file)
    combined.tracks.append(_track_from_notes(combined_track, all_notes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def _midi_length_ticks(midi_file: mido.MidiFile) -> int:
    max_tick = 0
    for track in midi_file.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(getattr(message, "time", 0))
        max_tick = max(max_tick, absolute_tick)
    return max_tick


def _midi_note_count(path: Path) -> int:
    midi_file = mido.MidiFile(path)
    return sum(
        1
        for track in midi_file.tracks
        for message in track
        if getattr(message, "type", None) == "note_on" and message.velocity > 0
    )


def _validate_target_midi_duration(
    output_path: Path,
    *,
    target_bars: list[int],
    project: dict[str, Any] | None,
) -> None:
    midi_file = mido.MidiFile(output_path)
    if not target_bars:
        return
    expected_ticks = int(
        round(
            sum(_bar_duration_beats(project, bar) for bar in target_bars)
            * midi_file.ticks_per_beat
        )
    )
    actual_ticks = _midi_length_ticks(midi_file)
    if actual_ticks > expected_ticks + midi_file.ticks_per_beat:
        raise ModelGenerationError(
            "MIDI-GPT target extraction exceeded requested bar duration: "
            f"actual_ticks={actual_ticks}, expected_ticks={expected_ticks}"
        )


def _midi_track_name(track: mido.MidiTrack) -> str:
    for message in track:
        if getattr(message, "type", None) == "track_name":
            return str(message.name)
    return ""


def _normalize_identifier(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _device_arg(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return None
    return normalized


def _float_schedule(value: Any, *, fallback: list[float]) -> list[float]:
    if value is None:
        return list(fallback)
    if isinstance(value, list | tuple):
        values = [float(item) for item in value]
        return values or list(fallback)
    return [float(value)]


def _attributes_for_request(request: ModelGenerationRequest) -> dict[str, int]:
    density = request.density or (request.role_intent or {}).get("density") or "medium"
    return {"note_density": _density_to_note_density(density)}


def _compatible_track_attributes(
    engine: Any,
    score: Any,
    *,
    target_track_index: int,
    attributes: dict[str, int],
) -> dict[str, int]:
    analyzer = getattr(engine, "_analyzer", None) or getattr(engine, "analyzer", None)
    if analyzer is None:
        return dict(attributes)

    attr_sizes = _safe_analyzer_mapping(analyzer, "attribute_sizes")
    attr_levels = _safe_analyzer_mapping(analyzer, "attribute_levels")
    attr_track_types = _safe_analyzer_mapping(analyzer, "attribute_track_types")
    tracks = _score_tracks(score)
    target_track = tracks[target_track_index] if target_track_index < len(tracks) else None
    is_drum_track = _is_drum_track(target_track)

    compatible: dict[str, int] = {}
    for key, value in attributes.items():
        if attr_sizes and key not in attr_sizes:
            continue
        if attr_levels.get(key, "track") != "track":
            continue
        required_track_type = attr_track_types.get(key, "both")
        if required_track_type == "drum" and not is_drum_track:
            continue
        if required_track_type == "melodic" and is_drum_track:
            continue
        max_size = attr_sizes.get(key)
        if max_size is not None:
            value = max(0, min(int(value), int(max_size) - 1))
        compatible[key] = int(value)
    return compatible


def _density_hard_limit(
    request: ModelGenerationRequest,
    *,
    requested_attributes: dict[str, int],
    applied_attributes: dict[str, int],
) -> int | None:
    configured = request.metadata.get("density_hard_limit")
    if configured is not None:
        return int(configured)
    requested_density = requested_attributes.get("note_density")
    if requested_density is None or "note_density" in applied_attributes:
        return None
    return max(1, int(requested_density))


def _safe_analyzer_mapping(analyzer: Any, method_name: str) -> dict[str, Any]:
    method = getattr(analyzer, method_name, None)
    if not callable(method):
        return {}
    try:
        value = method()
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _is_drum_track(track: Any) -> bool:
    if track is None:
        return False
    track_type = _normalized_text(_track_value(track, "track_type") or _track_value(track, "type"))
    if "drum" in track_type:
        return True
    instrument = _track_value(track, "instrument")
    try:
        return int(instrument) >= 128
    except (TypeError, ValueError):
        return False


def _density_to_note_density(value: Any) -> int:
    if isinstance(value, int | float):
        scaled = float(value) * 10 if float(value) <= 1 else float(value)
        return max(1, min(10, int(round(scaled))))
    normalized = _normalized_text(value).replace("-", "_")
    mapping = {
        "low": 2,
        "medium_low": 3,
        "medium": 5,
        "medium_high": 7,
        "high": 8,
    }
    return mapping.get(normalized, 5)


def _chunk_bar_indices(bar_indices: list[int], *, max_span: int = 8) -> list[list[int]]:
    ordered = sorted({int(bar) for bar in bar_indices})
    if not ordered:
        return []
    groups: list[list[int]] = []
    current = [ordered[0]]
    for bar in ordered[1:]:
        if bar == current[-1] + 1 and (bar - current[0] + 1) <= max_span:
            current.append(bar)
            continue
        groups.append(current)
        current = [bar]
    groups.append(current)
    return groups


def _construct_supported(callable_obj: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return callable_obj(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return callable_obj(**supported)


_INSTRUMENT_MIDI_PROGRAMS = {
    "piano": 0,
    "double_bass": 32,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "trumpet_bflat": 56,
    "trombone": 57,
    "tuba": 58,
    "clarinet_bflat": 71,
    "flute": 73,
}
