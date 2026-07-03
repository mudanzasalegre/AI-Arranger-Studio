from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mido

from model_backends.artifact import artifact_path, artifact_record
from model_backends.base import (
    ModelCapabilities,
    ModelGenerationRequest,
    ModelGenerationResult,
    ModelTask,
)
from model_backends.custom_role.base import CustomRoleModelBackend
from model_backends.custom_role.loader import (
    CUSTOM_ROLE_MODEL_VERSION,
    CustomRoleModelSpec,
    canonical_custom_role,
    inspect_custom_role_model,
)
from model_backends.errors import (
    ModelBackendUnavailableError,
    ModelGenerationError,
    UnsupportedModelTaskError,
)

CUSTOM_ROLE_TASKS: set[ModelTask] = {
    "generate_track",
    "infill_bars",
    "generate_variation",
}
TICKS_PER_BEAT = 480
ROLE_PROGRAMS = {
    "melody": 65,
    "walking_bass": 32,
    "piano_comping": 0,
    "horn_responses": 56,
    "drums": 0,
}
ROLE_CHANNELS = {
    "melody": 0,
    "walking_bass": 1,
    "piano_comping": 2,
    "horn_responses": 3,
    "drums": 9,
}


class StatisticalCustomRoleBackend(CustomRoleModelBackend):
    """Small local custom-role backend backed by trained token n-gram checkpoints."""

    backend_version = CUSTOM_ROLE_MODEL_VERSION

    def __init__(
        self,
        *,
        backend_id: str,
        role: str,
        checkpoint_dir: str | Path,
        model_file: str = "model.json",
        tokenizer_file: str = "tokenizer.json",
        config_file: str = "config.yaml",
        training_manifest_file: str = "training_manifest.yaml",
        license_report_file: str = "license_report.json",
        metrics_file: str = "metrics.json",
        output_dir: str | Path = "outputs/model_artifacts/raw",
        **_: Any,
    ) -> None:
        self.backend_id = backend_id
        self.role = canonical_custom_role(role)
        self.output_dir = Path(output_dir)
        self.spec = CustomRoleModelSpec(
            backend_id=backend_id,
            role=self.role,
            checkpoint_dir=str(checkpoint_dir),
            model_file=model_file,
            tokenizer_file=tokenizer_file,
            config_file=config_file,
            training_manifest_file=training_manifest_file,
            license_report_file=license_report_file,
            metrics_file=metrics_file,
        )
        self.inspection = inspect_custom_role_model(self.spec)
        self.capabilities = ModelCapabilities(
            symbolic_midi=True,
            multitrack=False,
            bar_infill=True,
            track_generation=True,
            text_prompt=False,
            json_planning=False,
            token_output=True,
            supports_training=True,
            commercial_use=self.inspection.commercial_use,
        )
        self._model: dict[str, Any] | None = None

    def is_available(self) -> bool:
        return self.inspection.available

    @property
    def unavailable_reason(self) -> str:
        return self.inspection.unavailable_reason

    @property
    def registry_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "checkpoint_dir": self.inspection.checkpoint_dir,
            "model_path": self.inspection.model_path,
            "tokenizer_path": self.inspection.tokenizer_path,
            "training_manifest_path": self.inspection.training_manifest_path,
            "license_report_path": self.inspection.license_report_path,
            "metrics_path": self.inspection.metrics_path,
            "commercial_allowed": self.inspection.commercial_allowed,
            "dataset_count": self.inspection.dataset_count,
            "rejected_source_count": self.inspection.rejected_source_count,
            "metrics": self.inspection.metrics,
            "warnings": self.inspection.warnings,
        }

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task not in CUSTOM_ROLE_TASKS:
            raise UnsupportedModelTaskError(
                f"{self.backend_id} does not support task {request.task}"
            )
        if not self.inspection.available:
            raise ModelBackendUnavailableError(
                f"Custom role model unavailable: {self.inspection.unavailable_reason}"
            )

        requested_role = _requested_role(request)
        if requested_role != self.role:
            raise ModelGenerationError(
                f"{self.backend_id} only supports role {self.role}; got {requested_role}"
            )
        export_mode = str(request.metadata.get("export_mode") or "private")
        if export_mode == "commercial" and not self.inspection.commercial_allowed:
            raise ModelBackendUnavailableError(
                f"{self.backend_id} is not allowed for export_mode=commercial"
            )

        model = self._load_model()
        if canonical_custom_role(str(model.get("role", ""))) != self.role:
            raise ModelGenerationError(
                f"{self.backend_id} checkpoint role mismatch: {model.get('role')!r}"
            )

        seed = int(request.seed if request.seed is not None else model.get("seed", 0) or 0)
        bars = [int(bar) for bar in (request.bars or [1])]
        target_tokens = _generate_tokens(
            model,
            seed=seed,
            max_tokens=_max_tokens_for_request(request),
            prefix=_prefix_for_request(request),
        )
        score = _score_tokens(model, target_tokens)

        token_path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=seed,
            suffix=".tokens.json",
            request_id=request.request_id,
        )
        midi_path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        _write_role_midi(
            midi_path,
            role=self.role,
            bars=bars,
            seed=seed,
            tokens=target_tokens,
            density=str((request.role_intent or {}).get("density") or request.density or "medium"),
        )
        payload = {
            "schema_version": CUSTOM_ROLE_MODEL_VERSION,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "generation_source": "statistical_custom_role_model",
            "role": self.role,
            "task": request.task,
            "target_tokens": target_tokens,
            "model_score": score,
            "role_intent": request.role_intent or {},
            "bars": bars,
            "track_id": request.track_id,
            "export_mode": export_mode,
            "model": {
                "model_type": model.get("model_type"),
                "model_id": model.get("model_id"),
                "checkpoint_dir": self.inspection.checkpoint_dir,
                "training_manifest_path": self.inspection.training_manifest_path,
                "license_report_path": self.inspection.license_report_path,
                "metrics_path": self.inspection.metrics_path,
                "commercial_allowed": self.inspection.commercial_allowed,
            },
        }
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        confidence = _confidence_from_score(score)
        artifact_metadata = {
            "backend_id": self.backend_id,
            "role": self.role,
            "task": request.task,
            "valid": True,
            "track_id": request.track_id,
            "bars": bars,
            "commercial_allowed": self.inspection.commercial_allowed,
            "model_type": model.get("model_type"),
            "model_id": model.get("model_id"),
            "generation_source": "statistical_custom_role_model",
        }
        return ModelGenerationResult(
            backend_id=self.backend_id,
            task=request.task,
            artifacts=[
                artifact_record(
                    "midi",
                    midi_path,
                    metadata={
                        **artifact_metadata,
                        "note_count": _note_count(midi_path),
                    },
                ),
                artifact_record(
                    "tokens",
                    token_path,
                    metadata={**artifact_metadata, "token_count": len(target_tokens)},
                ),
            ],
            confidence=confidence,
            warnings=list(self.inspection.warnings),
            raw_metadata={
                "backend_version": self.backend_version,
                "checkpoint_dir": self.inspection.checkpoint_dir,
                "tokenizer_path": self.inspection.tokenizer_path,
                "training_manifest_path": self.inspection.training_manifest_path,
                "license_report_path": self.inspection.license_report_path,
                "metrics_path": self.inspection.metrics_path,
                "commercial_allowed": self.inspection.commercial_allowed,
                "metrics": self.inspection.metrics,
            },
        )

    def _load_model(self) -> dict[str, Any]:
        if self._model is None:
            payload = json.loads(Path(self.inspection.model_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ModelGenerationError("Statistical custom role model must be a JSON object")
            self._model = payload
        return self._model


def _requested_role(request: ModelGenerationRequest) -> str:
    role_intent = request.role_intent or {}
    raw_role = (
        role_intent.get("role")
        or request.metadata.get("target_role")
        or request.metadata.get("role")
        or ""
    )
    return canonical_custom_role(str(raw_role))


def _prefix_for_request(request: ModelGenerationRequest) -> list[str] | None:
    role = _requested_role(request)
    if not role:
        return None
    return ["BOS", f"ROLE={role}"]


def _max_tokens_for_request(request: ModelGenerationRequest) -> int:
    bars = request.bars or [1]
    density = str((request.role_intent or {}).get("density") or request.density or "medium")
    density_factor = {"low": 8, "medium": 12, "medium_high": 16, "high": 20}.get(density, 12)
    return max(24, min(128, 8 + len(bars) * density_factor))


def _generate_tokens(
    model: dict[str, Any],
    *,
    seed: int,
    max_tokens: int,
    prefix: list[str] | None,
) -> list[str]:
    order = int(model.get("order") or 2)
    context_size = max(1, order - 1)
    next_token_counts = _next_token_counts(model)
    tokens = list(prefix or model.get("start_context") or ["BOS"])
    if not tokens:
        tokens = ["BOS"]
    while len(tokens) < max_tokens:
        context = _context_key(tokens[-context_size:])
        choices = next_token_counts.get(context)
        if not choices:
            choices = next_token_counts.get(_context_key(["BOS"]))
        if not choices:
            choices = next_token_counts.get(_context_key(["BOS", "BOS"]))
        if not choices:
            break
        next_token = _weighted_choice(choices, seed=seed, step=len(tokens), context=context)
        tokens.append(next_token)
        if next_token == "EOS":
            break
    if tokens[-1:] != ["EOS"] and len(tokens) < max_tokens:
        tokens.append("EOS")
    return tokens[:max_tokens]


def _score_tokens(model: dict[str, Any], tokens: list[str]) -> dict[str, float]:
    order = int(model.get("order") or 2)
    context_size = max(1, order - 1)
    padded = ["BOS"] * context_size + tokens + ["EOS"]
    vocabulary = model.get("vocabulary")
    vocabulary_size = max(1, len(vocabulary) if isinstance(vocabulary, list) else 1)
    context_counts = {
        str(key): int(value)
        for key, value in dict(model.get("context_counts") or {}).items()
    }
    next_token_counts = _next_token_counts(model)
    nll = 0.0
    count = 0
    for index in range(context_size, len(padded)):
        context = _context_key(padded[index - context_size : index])
        next_token = padded[index]
        next_counts = next_token_counts.get(context, {})
        numerator = next_counts.get(next_token, 0) + 1
        denominator = context_counts.get(context, 0) + vocabulary_size
        nll -= math.log(numerator / denominator)
        count += 1
    avg_nll = nll / count if count else 0.0
    return {
        "token_count": float(count),
        "negative_log_likelihood": round(avg_nll, 8),
        "perplexity": round(math.exp(avg_nll), 8),
    }


def _next_token_counts(model: dict[str, Any]) -> dict[str, dict[str, int]]:
    raw = model.get("next_token_counts")
    if not isinstance(raw, dict):
        return {}
    return {
        str(context): {
            str(token): int(count)
            for token, count in dict(counts).items()
            if int(count) > 0
        }
        for context, counts in raw.items()
        if isinstance(counts, dict)
    }


def _weighted_choice(
    choices: dict[str, int],
    *,
    seed: int,
    step: int,
    context: str,
) -> str:
    ordered = sorted((token, count) for token, count in choices.items() if count > 0)
    if not ordered:
        return "EOS"
    total = sum(count for _, count in ordered)
    ticket = int(_stable_hash(f"{seed}:{step}:{context}")[:12], 16) % total
    cumulative = 0
    for token, count in ordered:
        cumulative += count
        if ticket < cumulative:
            return token
    return ordered[-1][0]


def _write_role_midi(
    path: Path,
    *,
    role: str,
    bars: list[int],
    seed: int,
    tokens: list[str],
    density: str,
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Global", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(132), time=0))
    midi.tracks.append(meta)

    channel = ROLE_CHANNELS.get(role, 0)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=f"{role} statistical custom role", time=0))
    if channel != 9:
        track.append(
            mido.Message(
                "program_change",
                program=ROLE_PROGRAMS.get(role, 0),
                channel=channel,
                time=0,
            )
        )
    events = _role_events(role, bars=bars, seed=seed, tokens=tokens, density=density)
    current = 0
    for absolute_tick, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=max(0, absolute_tick - current)))
        current = absolute_tick
    midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def _role_events(
    role: str,
    *,
    bars: list[int],
    seed: int,
    tokens: list[str],
    density: str,
) -> list[tuple[int, int, mido.Message]]:
    variation = int(_stable_hash("|".join(tokens) + str(seed))[:4], 16) % 7
    events: list[tuple[int, int, mido.Message]] = []
    for offset, _bar in enumerate(bars):
        base = offset * 4.0
        if role == "walking_bass":
            pitches = [36, 40, 43, 47]
            events.extend(
                _notes(
                    [(base + i, 0.92, pitches[(i + variation) % 4]) for i in range(4)],
                    1,
                    82,
                )
            )
        elif role == "piano_comping":
            chord_a = [48 + variation % 3, 55, 60, 63]
            chord_b = [50 + variation % 3, 57, 62, 65]
            events.extend(_notes([(base + 0.5, 0.75, pitch) for pitch in chord_a], 2, 72))
            events.extend(_notes([(base + 2.5, 0.75, pitch) for pitch in chord_b], 2, 70))
        elif role == "horn_responses":
            pitches = [65 + variation % 5, 69 + variation % 5, 72 + variation % 5]
            events.extend(_notes([(base + 2.0, 0.5, pitch) for pitch in pitches], 3, 86))
            if density in {"medium_high", "high"}:
                events.extend(_notes([(base + 3.25, 0.35, pitch + 2) for pitch in pitches], 3, 82))
        elif role == "drums":
            drum_hits = [
                (base + 0.0, 0.12, 36),
                (base + 1.0, 0.12, 42),
                (base + 2.0, 0.12, 38),
                (base + 3.0, 0.12, 51),
            ]
            if density in {"medium_high", "high"}:
                drum_hits.extend([(base + 3.5, 0.12, 38), (base + 3.75, 0.12, 42)])
            events.extend(_notes(drum_hits, 9, 92))
        else:
            contour = [60, 63, 67, 70]
            events.extend(
                _notes(
                    [(base + i, 0.75, contour[(i + variation) % 4]) for i in range(4)],
                    0,
                    86,
                )
            )
    return events


def _notes(
    notes: list[tuple[float, float, int]],
    channel: int,
    velocity: int,
) -> list[tuple[int, int, mido.Message]]:
    events: list[tuple[int, int, mido.Message]] = []
    for start, duration, pitch in notes:
        start_tick = round(start * TICKS_PER_BEAT)
        end_tick = round((start + duration) * TICKS_PER_BEAT)
        events.append(
            (
                start_tick,
                0,
                mido.Message("note_on", channel=channel, note=pitch, velocity=velocity),
            )
        )
        events.append(
            (
                end_tick,
                1,
                mido.Message("note_off", channel=channel, note=pitch, velocity=0),
            )
        )
    return events


def _note_count(path: Path) -> int:
    midi = mido.MidiFile(path)
    return sum(
        1
        for track in midi.tracks
        for message in track
        if not message.is_meta
        and message.type == "note_on"
        and int(getattr(message, "velocity", 0)) > 0
    )


def _confidence_from_score(score: dict[str, float]) -> float:
    perplexity = max(1.0, float(score.get("perplexity", 1.0)))
    return round(max(0.35, min(0.92, 1.0 / (1.0 + perplexity / 16.0))), 3)


def _context_key(tokens: list[str]) -> str:
    return "\u241f".join(tokens)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
