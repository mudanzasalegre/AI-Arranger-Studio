from __future__ import annotations

from collections import defaultdict
from typing import Any

from arranger_core.model_contract import (
    DeterministicWalkingBassBackend,
    ModelBackend,
    ModelRequest,
    parse_note_token,
)
from arranger_core.schema import Bar, NoteEvent, RestEvent, Track, meter_to_quarter_beats


class AIWalkingBassGenerator:
    role = "walking_bass"

    def __init__(self, backend: ModelBackend | None = None) -> None:
        self.backend = backend or DeterministicWalkingBassBackend()

    def generate(self, context: Any) -> Track:
        request = ModelRequest(
            request_id=f"{context.project.project_id}:walking_bass",
            role=self.role,
            style=context.spec.style,
            key=context.spec.key,
            meter=context.spec.meter,
            tempo=context.spec.tempo,
            form=context.spec.form,
            seed=context.spec.seed,
            chord_context=_chord_context(context),
            controls={
                "bar_count": context.project.bar_count,
                "instrument": "double_bass",
                "beats_per_bar": [
                    meter_to_quarter_beats(context.project.meter_at_bar(bar_number))
                    for bar_number in range(1, context.project.bar_count + 1)
                ],
            },
            metadata={"project_id": context.project.project_id},
        )
        response = self.backend.generate(request)
        events_by_bar = _events_by_bar(response.target_tokens, self.backend.name)

        bars: list[Bar] = []
        for bar_number in range(1, context.project.bar_count + 1):
            bar_duration = meter_to_quarter_beats(context.project.meter_at_bar(bar_number))
            note_events = events_by_bar.get(bar_number, [])
            bars.append(
                Bar(
                    number=bar_number,
                    events=_complete_bar_events(note_events, bar_duration),
                    metadata={
                        "model_backend": self.backend.name,
                        "model_backend_version": self.backend.version,
                        "model_token_count": len(note_events),
                    },
                )
            )

        return Track(
            id="double_bass",
            instrument="double_bass",
            role=self.role,
            bars=bars,
            metadata={
                "generator": "AIWalkingBassGenerator",
                "model_backend": self.backend.name,
                "model_backend_version": self.backend.version,
                "model_response_metadata": response.metadata,
                "target_token_count": len(response.target_tokens),
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
) -> dict[int, list[NoteEvent]]:
    events_by_bar: dict[int, list[NoteEvent]] = defaultdict(list)
    for token in tokens:
        model_note = parse_note_token(token)
        if model_note is None:
            continue
        events_by_bar[model_note.bar].append(
            NoteEvent(
                pitch=model_note.pitch,
                start=model_note.start,
                duration=model_note.duration,
                velocity=max(1, min(127, model_note.velocity)),
                annotations={
                    "bass_role": "model_generated",
                    "model_backend": backend_name,
                    "model_token": token,
                },
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
