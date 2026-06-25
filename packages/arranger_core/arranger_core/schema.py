from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION}
DEFAULT_METER = "4/4"
FLOAT_TOLERANCE = 1e-6


class SchemaVersionError(ValueError):
    """Raised when a project payload uses an unsupported schema version."""


class BarDurationValidationError(ValueError):
    """Raised when one or more bars do not fill their expected duration."""

    def __init__(self, issues: list[BarDurationIssue]) -> None:
        self.issues = issues
        joined = "; ".join(issue.message for issue in issues)
        super().__init__(joined)


def _validate_schema_version(value: str) -> str:
    if value not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"Unsupported schema version {value!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return value


def meter_to_quarter_beats(meter: str) -> float:
    """Convert a meter like 4/4 or 6/8 to quarter-note beats per bar."""

    try:
        numerator_text, denominator_text = meter.split("/", maxsplit=1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError as exc:
        raise ValueError(f"Invalid meter {meter!r}; expected '<numerator>/<denominator>'") from exc

    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Invalid meter {meter!r}; numerator and denominator must be positive")

    return numerator * (4 / denominator)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GenerationSpec(StrictBaseModel):
    schema_version: str = SCHEMA_VERSION
    prompt: str = ""
    style: str = "hard_bop"
    substyle: str | None = None
    tempo: int = Field(default=132, ge=1, le=320)
    key: str = "C minor"
    meter: str = DEFAULT_METER
    form: str = "minor_blues_12"
    ensemble: str = "jazz_sextet"
    duration_bars: int | None = Field(default=None, ge=0)
    density: str = "medium"
    mood: str | None = None
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    instruments: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return _validate_schema_version(value)

    @field_validator("meter")
    @classmethod
    def meter_is_parseable(cls, value: str) -> str:
        meter_to_quarter_beats(value)
        return value


class TempoMark(StrictBaseModel):
    bar: int = Field(ge=1)
    bpm: int = Field(ge=1, le=320)


class KeyMark(StrictBaseModel):
    bar: int = Field(ge=1)
    key: str


class MeterMark(StrictBaseModel):
    bar: int = Field(ge=1)
    meter: str

    @field_validator("meter")
    @classmethod
    def meter_is_parseable(cls, value: str) -> str:
        meter_to_quarter_beats(value)
        return value


class Section(StrictBaseModel):
    name: str
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_bar_must_not_precede_start_bar(self) -> Self:
        if self.end_bar < self.start_bar:
            raise ValueError("Section end_bar must be greater than or equal to start_bar")
        return self

    @property
    def duration_bars(self) -> int:
        return self.end_bar - self.start_bar + 1


class ChordSymbol(StrictBaseModel):
    symbol: str
    bar: int | None = Field(default=None, ge=1)
    beat: float = Field(default=1.0, ge=0.0)
    duration: float | None = Field(default=None, gt=0.0)
    root: str | None = None
    quality: str | None = None
    bass: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimedEvent(StrictBaseModel):
    start: float = Field(default=0.0, ge=0.0)
    duration: float = Field(gt=0.0)
    voice: int = Field(default=1, ge=1)
    staff: int | None = Field(default=None, ge=1)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.duration


class NoteEvent(TimedEvent):
    type: Literal["note"] = "note"
    pitch: str
    velocity: int = Field(default=80, ge=1, le=127)
    tied_to_next: bool = False
    articulations: list[str] = Field(default_factory=list)
    dynamic: str | None = None


class RestEvent(TimedEvent):
    type: Literal["rest"] = "rest"


NotationEvent = NoteEvent | RestEvent


class BarDurationIssue(StrictBaseModel):
    severity: Literal["error"] = "error"
    track_id: str
    bar_number: int
    voice: int
    expected_beats: float
    actual_start: float | None = None
    actual_end: float | None = None
    message: str


class Bar(StrictBaseModel):
    number: int = Field(ge=1)
    meter: str | None = None
    chords: list[ChordSymbol] = Field(default_factory=list)
    events: list[NotationEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("meter")
    @classmethod
    def meter_is_parseable(cls, value: str | None) -> str | None:
        if value is not None:
            meter_to_quarter_beats(value)
        return value

    def expected_duration(self, default_meter: str = DEFAULT_METER) -> float:
        return meter_to_quarter_beats(self.meter or default_meter)

    def duration_issues(
        self,
        *,
        track_id: str,
        default_meter: str = DEFAULT_METER,
    ) -> list[BarDurationIssue]:
        expected_beats = self.expected_duration(default_meter)
        intervals_by_voice: dict[int, list[tuple[float, float]]] = defaultdict(list)

        for event in self.events:
            intervals_by_voice[event.voice].append((event.start, event.end))

        issues: list[BarDurationIssue] = []
        for voice, intervals in intervals_by_voice.items():
            issues.extend(
                _duration_issues_for_voice(
                    track_id=track_id,
                    bar_number=self.number,
                    voice=voice,
                    expected_beats=expected_beats,
                    intervals=intervals,
                )
            )

        return issues

    def assert_duration_valid(
        self,
        *,
        track_id: str = "<bar>",
        default_meter: str = DEFAULT_METER,
    ) -> None:
        issues = self.duration_issues(track_id=track_id, default_meter=default_meter)
        if issues:
            raise BarDurationValidationError(issues)


class Track(StrictBaseModel):
    id: str
    instrument: str
    role: str
    name: str | None = None
    channel: int | None = Field(default=None, ge=1, le=16)
    bars: list[Bar] = Field(default_factory=list)
    events: list[NotationEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def bar_count(self) -> int:
        return len(self.bars)


class ArrangementProject(StrictBaseModel):
    schema_version: str = SCHEMA_VERSION
    project_id: str = Field(default_factory=lambda: f"project-{uuid4().hex[:12]}")
    metadata: dict[str, Any] = Field(default_factory=dict)
    generation_spec: GenerationSpec | None = None
    tempo_map: list[TempoMark] = Field(default_factory=list)
    key_map: list[KeyMark] = Field(default_factory=list)
    meter_map: list[MeterMark] = Field(default_factory=list)
    form: list[Section] = Field(default_factory=list)
    chord_grid: list[ChordSymbol] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    export_manifest: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return _validate_schema_version(value)

    @property
    def bar_count(self) -> int:
        counts = [track.bar_count for track in self.tracks]
        counts.extend(section.end_bar for section in self.form)
        return max(counts, default=0)

    def meter_at_bar(self, bar_number: int) -> str:
        if bar_number < 1:
            raise ValueError("bar_number must be >= 1")

        meter = self.generation_spec.meter if self.generation_spec else DEFAULT_METER
        for marker in sorted(self.meter_map, key=lambda item: item.bar):
            if marker.bar <= bar_number:
                meter = marker.meter
            else:
                break
        return meter

    def validate_bar_durations(self) -> list[BarDurationIssue]:
        issues: list[BarDurationIssue] = []
        for track in self.tracks:
            for bar in track.bars:
                issues.extend(
                    bar.duration_issues(
                        track_id=track.id,
                        default_meter=self.meter_at_bar(bar.number),
                    )
                )
        return issues

    def assert_bar_durations_valid(self) -> None:
        issues = self.validate_bar_durations()
        if issues:
            raise BarDurationValidationError(issues)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> Self:
        if isinstance(data, bytes | bytearray):
            data = data.decode("utf-8")
        return cls.model_validate_json(data)

    def save_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def load_project_json(path: str | Path) -> ArrangementProject:
    return ArrangementProject.load_json(path)


def save_project_json(project: ArrangementProject, path: str | Path, *, indent: int = 2) -> Path:
    return project.save_json(path, indent=indent)


def _duration_issues_for_voice(
    *,
    track_id: str,
    bar_number: int,
    voice: int,
    expected_beats: float,
    intervals: list[tuple[float, float]],
) -> list[BarDurationIssue]:
    if not intervals:
        return []

    issues: list[BarDurationIssue] = []
    merged_intervals = _merge_intervals(intervals)

    first_start = merged_intervals[0][0]
    if first_start > FLOAT_TOLERANCE:
        issues.append(
            _gap_issue(
                track_id=track_id,
                bar_number=bar_number,
                voice=voice,
                expected_beats=expected_beats,
                gap_start=0.0,
                gap_end=first_start,
            )
        )

    previous_end = min(merged_intervals[0][1], expected_beats)
    for start, end in merged_intervals:
        if end > expected_beats + FLOAT_TOLERANCE:
            issues.append(
                BarDurationIssue(
                    track_id=track_id,
                    bar_number=bar_number,
                    voice=voice,
                    expected_beats=expected_beats,
                    actual_start=start,
                    actual_end=end,
                    message=(
                        f"Track {track_id!r} bar {bar_number} voice {voice} exceeds "
                        f"{expected_beats:g} beats ending at {end:g}"
                    ),
                )
            )

        if start > previous_end + FLOAT_TOLERANCE:
            issues.append(
                _gap_issue(
                    track_id=track_id,
                    bar_number=bar_number,
                    voice=voice,
                    expected_beats=expected_beats,
                    gap_start=previous_end,
                    gap_end=start,
                )
            )

        previous_end = max(previous_end, min(end, expected_beats))

    if previous_end < expected_beats - FLOAT_TOLERANCE:
        issues.append(
            _gap_issue(
                track_id=track_id,
                bar_number=bar_number,
                voice=voice,
                expected_beats=expected_beats,
                gap_start=previous_end,
                gap_end=expected_beats,
            )
        )

    return issues


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = []

    for start, end in ordered:
        if not merged or start > merged[-1][1] + FLOAT_TOLERANCE:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))

    return merged


def _gap_issue(
    *,
    track_id: str,
    bar_number: int,
    voice: int,
    expected_beats: float,
    gap_start: float,
    gap_end: float,
) -> BarDurationIssue:
    return BarDurationIssue(
        track_id=track_id,
        bar_number=bar_number,
        voice=voice,
        expected_beats=expected_beats,
        actual_start=gap_start,
        actual_end=gap_end,
        message=(
            f"Track {track_id!r} bar {bar_number} voice {voice} has a duration gap "
            f"from beat {gap_start:g} to {gap_end:g}"
        ),
    )


def project_from_mapping(data: dict[str, Any]) -> ArrangementProject:
    return ArrangementProject.model_validate(data)


def project_to_mapping(project: ArrangementProject) -> dict[str, Any]:
    return json.loads(project.to_json(indent=2))
