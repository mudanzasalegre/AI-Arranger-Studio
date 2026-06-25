from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.schema import ArrangementProject, NoteEvent, Track
from arranger_core.song_planner import GrooveMap

PERFORMANCE_MAP_VERSION = "0.1.0"
PerformanceSource = Literal["rule_based", "imported_model", "normalized_model"]

_EXPRESSIVE_TIMING_KEYS = {
    "performance_microtiming_ms",
    "humanized_timing_ms",
    "model_microtiming_ms",
    "source_microtiming_ms",
    "expressive_timing_ms",
}


class PerformanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VelocityRange(PerformanceModel):
    low: int = Field(ge=1, le=127)
    high: int = Field(ge=1, le=127)
    accent: int = 0


class PerformanceMap(PerformanceModel):
    schema_version: str = PERFORMANCE_MAP_VERSION
    feel: str
    meter: str
    swing_ratio: float = Field(ge=0.5, le=0.75)
    microtiming_ms: dict[str, int] = Field(default_factory=dict)
    velocity_ranges: dict[str, VelocityRange] = Field(default_factory=dict)
    performance_sources: list[PerformanceSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PerformanceMapper:
    def apply(
        self,
        project: ArrangementProject,
        *,
        seed: int | None = None,
        default_source: PerformanceSource = "rule_based",
    ) -> ArrangementProject:
        performance_project = project.model_copy(deep=True)
        performance_map = build_performance_map(performance_project)
        rng = random.Random((seed if seed is not None else _project_seed(project)) + 17001)
        sources_seen: set[PerformanceSource] = set()
        normalized_model_notes = 0
        preserved_model_timing = 0

        for track in performance_project.tracks:
            for bar in track.bars:
                for event in bar.events:
                    if not isinstance(event, NoteEvent):
                        continue
                    result = self._apply_event(
                        event,
                        track=track,
                        bar_number=bar.number,
                        rng=rng,
                        performance_map=performance_map,
                        default_source=default_source,
                    )
                    sources_seen.add(result["source"])
                    if result.get("normalized_model"):
                        normalized_model_notes += 1
                    if result.get("preserved_model_timing"):
                        preserved_model_timing += 1

        performance_map.performance_sources = sorted(sources_seen)
        performance_project.metadata = {
            **performance_project.metadata,
            "humanized": True,
            "performance_applied": True,
            "performance_map": performance_map.model_dump(mode="json"),
            "performance_summary": {
                "normalized_model_notes": normalized_model_notes,
                "preserved_model_timing_notes": preserved_model_timing,
                "sources": sorted(sources_seen),
            },
        }
        return performance_project

    def _apply_event(
        self,
        event: NoteEvent,
        *,
        track: Track,
        bar_number: int,
        rng: random.Random,
        performance_map: PerformanceMap,
        default_source: PerformanceSource,
    ) -> dict[str, Any]:
        if event.annotations.get("performance_applied") is True:
            source = _stored_performance_source(event, default_source)
            return {"source": source}

        source = _source_for_event(event, track=track, default_source=default_source)
        expressive_timing = _has_expressive_timing(event)
        role = _role_key(track)
        beat_accent = _beat_accent(event.start)
        timing_status = "generated"
        preserved_model_timing = False

        if source == "imported_model":
            performance_source: PerformanceSource = "normalized_model"
            previous_velocity = event.velocity
            event.velocity = _normalize_model_velocity(
                event.velocity,
                role=role,
                beat_accent=beat_accent,
                performance_map=performance_map,
            )
            if expressive_timing:
                microtiming_ms = _existing_microtiming_ms(event)
                timing_status = "preserved_imported"
                preserved_model_timing = True
            else:
                microtiming_ms = _microtiming_ms(
                    event,
                    role=role,
                    rng=rng,
                    performance_map=performance_map,
                    imported=True,
                )
            event.annotations["model_velocity_before_normalization"] = previous_velocity
            event.annotations["performance_normalized_from"] = "imported_model"
        else:
            performance_source = "rule_based"
            previous_velocity = event.velocity
            delta = _rule_based_velocity_delta(
                event,
                role=role,
                rng=rng,
                beat_accent=beat_accent,
                performance_map=performance_map,
            )
            event.velocity = max(1, min(127, event.velocity + delta))
            microtiming_ms = (
                _existing_microtiming_ms(event)
                if expressive_timing
                else _microtiming_ms(
                    event,
                    role=role,
                    rng=rng,
                    performance_map=performance_map,
                    imported=False,
                )
            )
            event.annotations["humanized_velocity_delta"] = event.velocity - previous_velocity

        event.annotations["performance_applied"] = True
        event.annotations["performance_map_version"] = performance_map.schema_version
        event.annotations["performance_source"] = performance_source
        event.annotations["performance_timing_status"] = timing_status
        event.annotations["performance_bar"] = bar_number
        event.annotations["performance_velocity_delta"] = event.velocity - previous_velocity
        event.annotations["performance_microtiming_ms"] = microtiming_ms
        event.annotations["humanized_timing_ms"] = microtiming_ms
        return {
            "source": performance_source,
            "normalized_model": performance_source == "normalized_model",
            "preserved_model_timing": preserved_model_timing,
        }


def build_performance_map(project: ArrangementProject) -> PerformanceMap:
    groove = _project_groove_map(project)
    feel = groove.feel if groove else _fallback_feel(project)
    meter = groove.meter if groove else project.meter_at_bar(1) if project.bar_count else "4/4"
    swing_ratio = groove.swing_ratio if groove else _fallback_swing_ratio(project)
    return PerformanceMap(
        feel=feel,
        meter=meter,
        swing_ratio=swing_ratio,
        microtiming_ms={
            "drums": 6,
            "walking_bass": 8,
            "comping": 12,
            "melody": 10,
            "horn_response": 8,
            "imported_model": 6,
        },
        velocity_ranges={
            "drums": VelocityRange(low=38, high=104, accent=6),
            "walking_bass": VelocityRange(low=58, high=86, accent=5),
            "comping": VelocityRange(low=50, high=82, accent=3),
            "melody": VelocityRange(low=62, high=98, accent=5),
            "horn_response": VelocityRange(low=60, high=100, accent=7),
            "unknown": VelocityRange(low=48, high=96, accent=3),
        },
        notes=[
            "Microtiming is applied at MIDI render time, not to notated starts.",
            "Imported model material is velocity-normalized before final export.",
        ],
    )


def _project_groove_map(project: ArrangementProject) -> GrooveMap | None:
    raw_song_plan = project.metadata.get("song_plan")
    if not isinstance(raw_song_plan, dict):
        return None
    raw_groove = raw_song_plan.get("groove_map")
    if not isinstance(raw_groove, dict):
        return None
    try:
        return GrooveMap.model_validate(raw_groove)
    except Exception:
        return None


def _fallback_feel(project: ArrangementProject) -> str:
    spec = project.generation_spec
    if spec is None:
        return "swing"
    if spec.meter == "3/4" or spec.style == "jazz_waltz":
        return "waltz"
    if spec.style == "bossa_nova":
        return "bossa"
    if spec.style == "funk_jazz":
        return "straight_eighth"
    if spec.style == "jazz_ballad":
        return "slow_swing"
    return "swing"


def _fallback_swing_ratio(project: ArrangementProject) -> float:
    spec = project.generation_spec
    if spec is None:
        return 0.61
    if spec.style in {"bossa_nova", "funk_jazz"}:
        return 0.5
    if spec.tempo < 90:
        return 0.66
    if spec.tempo > 180:
        return 0.57
    return 0.61


def _project_seed(project: ArrangementProject) -> int:
    if project.generation_spec is not None:
        return project.generation_spec.seed
    return int(project.metadata.get("seed", 0) or 0)


def _source_for_event(
    event: NoteEvent,
    *,
    track: Track,
    default_source: PerformanceSource,
) -> PerformanceSource:
    if _event_is_imported_model(event, track):
        return "imported_model"
    stored = event.annotations.get("performance_source")
    if stored in {"rule_based", "imported_model", "normalized_model"}:
        return stored  # type: ignore[return-value]
    return default_source


def _stored_performance_source(
    event: NoteEvent,
    default_source: PerformanceSource,
) -> PerformanceSource:
    stored = event.annotations.get("performance_source")
    if stored in {"rule_based", "imported_model", "normalized_model"}:
        return stored  # type: ignore[return-value]
    return default_source


def _event_is_imported_model(event: NoteEvent, track: Track) -> bool:
    source = str(event.annotations.get("source") or track.metadata.get("source") or "")
    return (
        source in {"model_artifact", "text2midi_sketch"}
        or event.annotations.get("imported_model") is True
        or bool(event.annotations.get("artifact_id"))
        or bool(track.metadata.get("artifact_id"))
    )


def _has_expressive_timing(event: NoteEvent) -> bool:
    return any(key in event.annotations for key in _EXPRESSIVE_TIMING_KEYS)


def _existing_microtiming_ms(event: NoteEvent) -> int:
    for key in _EXPRESSIVE_TIMING_KEYS:
        value = event.annotations.get(key)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return 0


def _role_key(track: Track) -> str:
    if track.role == "drums" or track.instrument == "drum_kit":
        return "drums"
    if track.role in {"walking_bass", "bass"}:
        return "walking_bass"
    if track.role in {"comping", "piano"} or track.instrument == "piano":
        return "comping"
    if track.role == "horn_response":
        return "horn_response"
    if track.role == "melody":
        return "melody"
    return "unknown"


def _beat_accent(start: float) -> int:
    rounded = round(start % 4.0, 3)
    if abs(rounded) < 1e-6:
        return 2
    if abs(rounded - round(rounded)) < 1e-6:
        return 1
    return 0


def _normalize_model_velocity(
    velocity: int,
    *,
    role: str,
    beat_accent: int,
    performance_map: PerformanceMap,
) -> int:
    velocity_range = performance_map.velocity_ranges.get(
        role,
        performance_map.velocity_ranges["unknown"],
    )
    target = max(velocity_range.low, min(velocity_range.high, int(velocity)))
    if beat_accent:
        target = min(velocity_range.high, target + velocity_range.accent * beat_accent // 2)
    return max(1, min(127, target))


def _rule_based_velocity_delta(
    event: NoteEvent,
    *,
    role: str,
    rng: random.Random,
    beat_accent: int,
    performance_map: PerformanceMap,
) -> int:
    velocity_range = performance_map.velocity_ranges.get(
        role,
        performance_map.velocity_ranges["unknown"],
    )
    span = {
        "drums": 5,
        "walking_bass": 4,
        "comping": 3,
        "melody": 4,
        "horn_response": 5,
    }.get(role, 3)
    accent_delta = velocity_range.accent if beat_accent == 2 else velocity_range.accent // 2
    if beat_accent == 0:
        accent_delta = 0
    delta = rng.randint(-span, span) + accent_delta
    projected = event.velocity + delta
    if projected < velocity_range.low:
        delta += velocity_range.low - projected
    elif projected > velocity_range.high:
        delta -= projected - velocity_range.high
    return delta


def _microtiming_ms(
    event: NoteEvent,
    *,
    role: str,
    rng: random.Random,
    performance_map: PerformanceMap,
    imported: bool,
) -> int:
    jitter_key = "imported_model" if imported else role
    jitter = performance_map.microtiming_ms.get(jitter_key, 6)
    random_offset = rng.randint(-jitter, jitter)
    swing_offset = _swing_offset_ms(event.start, performance_map=performance_map)
    return int(max(-45, min(95, random_offset + swing_offset)))


def _swing_offset_ms(start: float, *, performance_map: PerformanceMap) -> int:
    if performance_map.swing_ratio <= 0.5:
        return 0
    beat_position = start % 1.0
    if abs(beat_position - 0.5) > 0.02:
        return 0
    # 120 BPM is used as a stable notational baseline; export converts ms to ticks.
    return round((performance_map.swing_ratio - 0.5) * 500)
