from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.chords import ChordParser
from arranger_core.music_theory import midi_to_note, note_to_midi

MODEL_CONTRACT_VERSION = "0.1.0"


class ModelContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelRequest(ModelContractModel):
    schema_version: str = MODEL_CONTRACT_VERSION
    request_id: str = ""
    role: str
    style: str = "unknown"
    key: str = "C minor"
    meter: str = "4/4"
    tempo: int = Field(default=120, ge=1, le=320)
    form: str = "unknown"
    seed: int = 0
    chord_context: list[str] = Field(default_factory=list)
    previous_tokens: list[str] = Field(default_factory=list)
    controls: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(ModelContractModel):
    schema_version: str = MODEL_CONTRACT_VERSION
    role: str
    target_tokens: list[str] = Field(default_factory=list)
    backend_name: str = ""
    backend_version: str = ""
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    name: str
    version: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


@runtime_checkable
class RoleModelGenerator(Protocol):
    role: str
    backend: ModelBackend

    def generate(self, context: Any) -> Any:
        ...


@dataclass(frozen=True)
class ModelNote:
    bar: int
    start: float
    duration: float
    pitch: str
    velocity: int = 72


def note_token(
    *,
    bar: int,
    start: float,
    duration: float,
    pitch: str,
    velocity: int = 72,
) -> str:
    return (
        "NOTE"
        f"|bar={bar}"
        f"|start={start:g}"
        f"|duration={duration:g}"
        f"|pitch={pitch}"
        f"|velocity={velocity}"
    )


def parse_note_token(token: str) -> ModelNote | None:
    if not token.startswith("NOTE|"):
        return None
    fields: dict[str, str] = {}
    for chunk in token.split("|")[1:]:
        if "=" not in chunk:
            return None
        key, value = chunk.split("=", maxsplit=1)
        fields[key] = value
    try:
        return ModelNote(
            bar=int(fields["bar"]),
            start=float(fields["start"]),
            duration=float(fields["duration"]),
            pitch=fields["pitch"],
            velocity=int(fields.get("velocity", 72)),
        )
    except (KeyError, TypeError, ValueError):
        return None


class DeterministicWalkingBassBackend:
    """Local backend for smoke tests until a trained symbolic model exists."""

    name = "deterministic-walking-bass-placeholder"
    version = MODEL_CONTRACT_VERSION

    def __init__(self, *, chord_parser: ChordParser | None = None) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.role != "walking_bass":
            raise ValueError(f"{self.name} only supports walking_bass requests")

        bar_count = int(request.controls.get("bar_count") or len(request.chord_context) or 1)
        beats_by_bar = request.controls.get("beats_per_bar", [])
        chords = request.chord_context or ["Cm7"]
        anchor = note_to_midi("C2")
        tokens: list[str] = []
        for bar_number in range(1, bar_count + 1):
            chord_symbol = chords[(bar_number - 1) % len(chords)]
            beat_count = _beat_count_for_bar(beats_by_bar, bar_number)
            intervals = _bass_intervals_for_chord(chord_symbol, beat_count, self.chord_parser)
            root_pc = _root_pc(chord_symbol, self.chord_parser)
            for beat_index, interval in enumerate(intervals):
                pitch, anchor = _nearest_bass_pitch((root_pc + interval) % 12, anchor)
                tokens.append(
                    note_token(
                        bar=bar_number,
                        start=float(beat_index),
                        duration=1.0,
                        pitch=pitch,
                        velocity=74 if beat_index == 0 else 68,
                    )
                )

        return ModelResponse(
            role=request.role,
            target_tokens=tokens,
            backend_name=self.name,
            backend_version=self.version,
            metadata={
                "contract": "symbolic_note_tokens",
                "bar_count": bar_count,
                "source": "deterministic_placeholder",
            },
        )


def _beat_count_for_bar(raw_beats_by_bar: Any, bar_number: int) -> int:
    if isinstance(raw_beats_by_bar, list) and bar_number <= len(raw_beats_by_bar):
        try:
            return max(1, round(float(raw_beats_by_bar[bar_number - 1])))
        except (TypeError, ValueError):
            return 4
    return 4


def _bass_intervals_for_chord(
    chord_symbol: str,
    beat_count: int,
    parser: ChordParser,
) -> list[int]:
    try:
        parsed = parser.parse(chord_symbol)
        third = 3 if parsed.quality in {"minor_triad", "half_diminished"} else 4
        seventh = 10 if any(interval % 12 == 10 for interval in parsed.chord_tone_intervals) else 11
    except ValueError:
        third = 4
        seventh = 10
    return [0, third, 7, seventh][:beat_count]


def _root_pc(chord_symbol: str, parser: ChordParser) -> int:
    try:
        return parser.parse(chord_symbol).root_pc
    except ValueError:
        return 0


def _nearest_bass_pitch(target_pc: int, anchor_midi: int) -> tuple[str, int]:
    low = note_to_midi("E1")
    high = note_to_midi("C4")
    candidates = [
        midi_note
        for midi_note in range(low, high + 1)
        if midi_note % 12 == target_pc % 12
    ]
    selected = min(candidates, key=lambda midi_note: abs(midi_note - anchor_midi))
    return midi_to_note(selected, prefer_sharps=False), selected
