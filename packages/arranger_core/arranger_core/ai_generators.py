from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from arranger_core.model_contract import (
    DeterministicRoleModelBackend,
    DeterministicWalkingBassBackend,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    parse_note_token,
)
from arranger_core.schema import Bar, NoteEvent, RestEvent, Track, meter_to_quarter_beats

RoleModelMode = Literal["external_model", "custom_model"]
ROLE_MODEL_MODES_AVAILABLE = ["rule_based", "retrieval", "external_model", "custom_model"]
DRUM_BY_PITCH = {
    "C2": "kick",
    "D2": "snare",
    "F#2": "closed_hihat",
    "G#2": "hihat_pedal",
    "A2": "low_tom",
    "B2": "mid_tom",
    "D3": "high_tom",
    "C#3": "crash",
    "D#3": "ride",
}


class AIWalkingBassGenerator:
    role = "walking_bass"

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        model_mode: RoleModelMode = "custom_model",
    ) -> None:
        self.backend = backend or DeterministicWalkingBassBackend()
        self.model_mode = model_mode

    def generate(self, context: Any) -> Track:
        return _generate_model_track(
            context,
            backend=self.backend,
            generator_name=self.__class__.__name__,
            model_mode=self.model_mode,
            request_role=self.role,
            track_id="double_bass",
            instrument="double_bass",
            track_role=self.role,
            annotation_role_key="bass_role",
        )


class AIPianoCompingGenerator:
    role = "comping"

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        model_mode: RoleModelMode = "custom_model",
    ) -> None:
        self.backend = backend or DeterministicRoleModelBackend()
        self.model_mode = model_mode

    def generate(self, context: Any) -> Track:
        return _generate_model_track(
            context,
            backend=self.backend,
            generator_name=self.__class__.__name__,
            model_mode=self.model_mode,
            request_role=self.role,
            track_id="piano",
            instrument="piano",
            track_role=self.role,
            annotation_role_key="piano_role",
            name="Piano",
        )


class AIMelodyGenerator:
    role = "melody"

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        model_mode: RoleModelMode = "custom_model",
    ) -> None:
        self.backend = backend or DeterministicRoleModelBackend()
        self.model_mode = model_mode

    def generate(self, context: Any) -> Track:
        instrument_id = _lead_instrument_id(context)
        return self.generate_for_instrument(context, instrument_id)

    def generate_for_instrument(self, context: Any, instrument_id: str) -> Track:
        track_id = instrument_id if instrument_id != "piano" else "piano_melody"
        return _generate_model_track(
            context,
            backend=self.backend,
            generator_name=self.__class__.__name__,
            model_mode=self.model_mode,
            request_role=self.role,
            track_id=track_id,
            instrument=instrument_id,
            track_role=self.role,
            annotation_role_key="melodic_role",
            name=_display_name(context, instrument_id),
        )


class AIHornResponseGenerator:
    role = "horn_response"

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        model_mode: RoleModelMode = "custom_model",
    ) -> None:
        self.backend = backend or DeterministicRoleModelBackend()
        self.model_mode = model_mode

    def generate(self, context: Any) -> list[Track]:
        instruments = [
            instrument_id
            for instrument_id in context.instrument_ids
            if _is_horn(context, instrument_id)
        ]
        return [
            self.generate_for_instrument(context, instrument_id, harmony_index=index)
            for index, instrument_id in enumerate(instruments, start=1)
        ]

    def generate_for_instrument(
        self,
        context: Any,
        instrument_id: str,
        *,
        harmony_index: int,
    ) -> Track:
        return _generate_model_track(
            context,
            backend=self.backend,
            generator_name=self.__class__.__name__,
            model_mode=self.model_mode,
            request_role=self.role,
            track_id=instrument_id,
            instrument=instrument_id,
            track_role=self.role,
            annotation_role_key="horn_role",
            name=_display_name(context, instrument_id),
            request_metadata={"harmony_index": harmony_index},
            track_metadata={"harmony_index": harmony_index},
        )


class AIDrumsGenerator:
    role = "drums"

    def __init__(
        self,
        backend: ModelBackend | None = None,
        *,
        model_mode: RoleModelMode = "custom_model",
    ) -> None:
        self.backend = backend or DeterministicRoleModelBackend()
        self.model_mode = model_mode

    def generate(self, context: Any) -> Track:
        return _generate_model_track(
            context,
            backend=self.backend,
            generator_name=self.__class__.__name__,
            model_mode=self.model_mode,
            request_role=self.role,
            track_id="drum_kit",
            instrument="drum_kit",
            track_role=self.role,
            annotation_role_key="drums_role",
        )


def _generate_model_track(
    context: Any,
    *,
    backend: ModelBackend,
    generator_name: str,
    model_mode: RoleModelMode,
    request_role: str,
    track_id: str,
    instrument: str,
    track_role: str,
    annotation_role_key: str,
    name: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    track_metadata: dict[str, Any] | None = None,
) -> Track:
    request = _model_request(
        context,
        role=request_role,
        instrument=instrument,
        track_id=track_id,
        model_mode=model_mode,
        metadata=request_metadata,
    )
    response = backend.generate(request)
    return _track_from_response(
        context,
        response=response,
        backend=backend,
        generator_name=generator_name,
        model_mode=model_mode,
        track_id=track_id,
        instrument=instrument,
        track_role=track_role,
        annotation_role_key=annotation_role_key,
        name=name,
        track_metadata=track_metadata,
    )


def _model_request(
    context: Any,
    *,
    role: str,
    instrument: str,
    track_id: str,
    model_mode: RoleModelMode,
    metadata: dict[str, Any] | None = None,
) -> ModelRequest:
    beats_by_bar = [
        meter_to_quarter_beats(context.project.meter_at_bar(bar_number))
        for bar_number in range(1, context.project.bar_count + 1)
    ]
    return ModelRequest(
        request_id=f"{context.project.project_id}:{track_id}:{role}",
        role=role,
        style=context.spec.style,
        key=context.spec.key,
        meter=context.spec.meter,
        tempo=context.spec.tempo,
        form=context.spec.form,
        seed=context.spec.seed,
        chord_context=_chord_context(context),
        controls={
            "bar_count": context.project.bar_count,
            "instrument": instrument,
            "track_id": track_id,
            "beats_per_bar": beats_by_bar,
            "role_model_mode": model_mode,
            "output_contract": "NOTE_tokens",
        },
        metadata={"project_id": context.project.project_id, **(metadata or {})},
    )


def _track_from_response(
    context: Any,
    *,
    response: ModelResponse,
    backend: ModelBackend,
    generator_name: str,
    model_mode: RoleModelMode,
    track_id: str,
    instrument: str,
    track_role: str,
    annotation_role_key: str,
    name: str | None,
    track_metadata: dict[str, Any] | None = None,
) -> Track:
    events_by_bar = _events_by_bar(
        response.target_tokens,
        backend.name,
        role=track_role,
        role_key=annotation_role_key,
    )
    bars: list[Bar] = []
    for bar_number in range(1, context.project.bar_count + 1):
        bar_duration = meter_to_quarter_beats(context.project.meter_at_bar(bar_number))
        note_events = events_by_bar.get(bar_number, [])
        bars.append(
            Bar(
                number=bar_number,
                events=_complete_bar_events(note_events, bar_duration),
                metadata={
                    "model_backend": backend.name,
                    "model_backend_version": backend.version,
                    "model_token_count": len(note_events),
                    "role_model_mode": model_mode,
                },
            )
        )

    return Track(
        id=track_id,
        instrument=instrument,
        role=track_role,
        name=name,
        bars=bars,
        metadata={
            "generator": generator_name,
            "role_model_interface": "custom_role_model_v0",
            "role_model_mode": model_mode,
            "modes_available": ROLE_MODEL_MODES_AVAILABLE,
            "model_backend": backend.name,
            "model_backend_version": backend.version,
            "model_response_metadata": response.metadata,
            "target_token_count": len(response.target_tokens),
            **(track_metadata or {}),
        },
    )


def _chord_context(context: Any) -> list[str]:
    chords_by_bar: dict[int, list[Any]] = context.chords_by_bar
    chord_context: list[str] = []
    previous = "C"
    for bar_number in range(1, context.project.bar_count + 1):
        chords = chords_by_bar.get(bar_number, [])
        if chords:
            previous = str(chords[0].symbol)
        chord_context.append(previous)
    return chord_context


def _events_by_bar(
    tokens: list[str],
    backend_name: str,
    *,
    role: str,
    role_key: str,
) -> dict[int, list[NoteEvent]]:
    events_by_bar: dict[int, list[NoteEvent]] = defaultdict(list)
    for token in tokens:
        model_note = parse_note_token(token)
        if model_note is None:
            continue
        annotations = {
            role_key: "model_generated",
            "source": "model_backend",
            "role_model_role": role,
            "model_backend": backend_name,
            "model_token": token,
        }
        drum_name = DRUM_BY_PITCH.get(model_note.pitch)
        if drum_name:
            annotations["drum"] = drum_name
        events_by_bar[model_note.bar].append(
            NoteEvent(
                pitch=model_note.pitch,
                start=model_note.start,
                duration=model_note.duration,
                velocity=max(1, min(127, model_note.velocity)),
                annotations=annotations,
            )
        )
    return {
        bar_number: sorted(events, key=lambda event: (event.start, event.pitch))
        for bar_number, events in events_by_bar.items()
    }


def _complete_bar_events(
    note_events: list[NoteEvent],
    bar_duration: float,
) -> list[NoteEvent | RestEvent]:
    valid_notes = [
        event
        for event in note_events
        if event.start >= 0 and event.duration > 0 and event.start < bar_duration
    ]
    trimmed_notes = [
        event.model_copy(update={"duration": min(event.duration, bar_duration - event.start)})
        for event in valid_notes
    ]
    occupied = sorted(
        {(event.start, event.start + event.duration) for event in trimmed_notes},
        key=lambda item: item[0],
    )
    rests: list[RestEvent] = []
    cursor = 0.0
    for start, end in occupied:
        if start > cursor:
            rests.append(RestEvent(start=cursor, duration=start - cursor))
        cursor = max(cursor, end)
    if cursor < bar_duration:
        rests.append(RestEvent(start=cursor, duration=bar_duration - cursor))
    return sorted([*trimmed_notes, *rests], key=lambda event: (event.start, event.duration))


def _lead_instrument_id(context: Any) -> str:
    for instrument_id in context.instrument_ids:
        if _is_horn(context, instrument_id):
            return instrument_id
    return "piano" if "piano" in context.instrument_ids else context.instrument_ids[0]


def _is_horn(context: Any, instrument_id: str) -> bool:
    try:
        instrument = context.instrument_catalog.get(instrument_id)
    except KeyError:
        return False
    return instrument.breath_required and instrument.family in {"woodwind", "brass"}


def _display_name(context: Any, instrument_id: str) -> str:
    try:
        return context.instrument_catalog.get(instrument_id).display_name
    except KeyError:
        return instrument_id
