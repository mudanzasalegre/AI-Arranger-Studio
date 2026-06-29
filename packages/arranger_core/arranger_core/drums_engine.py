from __future__ import annotations

from statistics import pstdev
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.music_theory import note_to_midi
from arranger_core.retrieval import retrieval_trace, retrieve_pattern
from arranger_core.schema import ArrangementProject, Bar, GenerationSpec, NoteEvent, Track
from arranger_core.song_planner import GrooveMap, SectionPlan, SongPlan

DRUM_ENGINE_VERSION = "0.1.0"
DrumsEngineMode = Literal["rule_based", "retrieval"]
DrumsGrooveStyle = Literal["swing", "bossa", "funk", "waltz", "ballad", "straight_eighth"]
DrumsSource = Literal["rule_based", "retrieval", "fallback_rule_based"]

DRUM_PITCHES = {
    "kick": "C2",
    "snare": "D2",
    "closed_hihat": "F#2",
    "hihat_pedal": "G#2",
    "low_tom": "A2",
    "mid_tom": "B2",
    "high_tom": "D3",
    "crash": "C#3",
    "ride": "D#3",
}
ALLOWED_DRUM_NAMES = frozenset(DRUM_PITCHES)


class DrumsEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DrumLedgerEntry(DrumsEngineModel):
    bar_number: int
    mode: DrumsEngineMode
    source: DrumsSource
    groove_style: DrumsGrooveStyle
    energy: float
    density: float
    fill: bool = False
    setup: bool = False
    break_bar: bool = False
    horn_hit_support: bool = False
    source_pattern_id: str | None = None
    instruments: list[str] = Field(default_factory=list)
    signature: str


class DrumPatternLedger(DrumsEngineModel):
    schema_version: str = DRUM_ENGINE_VERSION
    entries: list[DrumLedgerEntry] = Field(default_factory=list)

    def add(self, entry: DrumLedgerEntry) -> None:
        self.entries.append(entry)


class DrumsValidationReport(DrumsEngineModel):
    status: Literal["pass", "fail"]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class DrumsEngine:
    def generate(self, context: Any) -> Track:
        mode = _drums_mode(context)
        song_plan = _song_plan(context)
        groove_map = _groove_map(context)
        groove_style = _groove_style(context, groove_map)
        pattern = _select_pattern(context) if mode == "retrieval" else None
        if mode == "retrieval" and pattern is None:
            mode = "rule_based"

        bars: list[Bar] = []
        ledger = DrumPatternLedger()
        for bar_number in range(1, context.project.bar_count + 1):
            bar_duration = _bar_duration(context.project, bar_number)
            grid = _grid_for_duration(bar_duration)
            section = _section_for_bar(song_plan, bar_number)
            energy = _bar_energy(song_plan, section, bar_number)
            density = _drum_density(context, section, energy)
            fill = _is_fill_bar(groove_map, context.project, bar_number)
            setup = _is_setup_bar(groove_map, bar_number)
            break_bar = _is_break_bar(groove_map, bar_number)
            horn_hit = _is_horn_hit_bar(groove_map, bar_number)

            events: list[NoteEvent] = []
            source: DrumsSource = "rule_based"
            if pattern is not None and not fill and not setup and not break_bar:
                events = _events_from_pattern(
                    pattern,
                    bar_duration=bar_duration,
                    groove_style=groove_style,
                    grid=grid,
                    energy=energy,
                )
                if events:
                    source = "retrieval"

            if not events:
                events = _rule_based_events(
                    bar_number,
                    grid=grid,
                    bar_duration=bar_duration,
                    groove_style=groove_style,
                    energy=energy,
                    density=density,
                    fill=fill,
                    setup=setup,
                    break_bar=break_bar,
                    horn_hit=horn_hit,
                    groove_map=groove_map,
                )
                if pattern is not None and source == "retrieval":
                    source = "fallback_rule_based"

            metadata = {
                "fill": fill,
                "setup": setup,
                "break": break_bar,
                "horn_hit_support": horn_hit,
                "feel": groove_style,
                "energy": round(energy, 3),
                "density": round(density, 3),
                "drums_engine_mode": mode,
                "drums_source": source,
                "learned_pattern_id": pattern.get("id")
                if pattern and source == "retrieval"
                else None,
            }
            bar = Bar(number=bar_number, events=events, metadata=metadata)
            bars.append(bar)
            ledger.add(
                DrumLedgerEntry(
                    bar_number=bar_number,
                    mode=mode,
                    source=source,
                    groove_style=groove_style,
                    energy=round(energy, 3),
                    density=round(density, 3),
                    fill=fill,
                    setup=setup,
                    break_bar=break_bar,
                    horn_hit_support=horn_hit,
                    source_pattern_id=pattern.get("id")
                    if pattern and source == "retrieval"
                    else None,
                    instruments=sorted(_drum_names(events)),
                    signature=_bar_signature(events),
                )
            )

        track = Track(
            id="drum_kit",
            instrument="drum_kit",
            role="drums",
            bars=bars,
            metadata={
                "generator": "DrumsEngine",
                "drums_engine_version": DRUM_ENGINE_VERSION,
                "drums_engine_mode": mode,
                "groove": groove_style,
                "learned_pattern_id": pattern.get("id") if pattern else None,
                "retrieval_trace": retrieval_trace(pattern),
            },
        )
        validation = self.validate_track(context.project, track)
        return track.model_copy(
            update={
                "metadata": {
                    **track.metadata,
                    "drums_validation": validation.model_dump(mode="json"),
                    "drum_pattern_ledger": ledger.model_dump(mode="json"),
                    "groove_map": groove_map.model_dump(mode="json") if groove_map else None,
                }
            }
        )

    def validate_track(self, project: ArrangementProject, track: Track) -> DrumsValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        bars = track.bars
        notes = [event for bar in bars for event in bar.events if isinstance(event, NoteEvent)]

        if not notes:
            errors.append(_issue("no_drum_notes", "Drum track has no notes", track_id=track.id))

        unsupported = [
            (bar.number, event.pitch)
            for bar in bars
            for event in bar.events
            if isinstance(event, NoteEvent)
            and note_to_midi(event.pitch)
            not in {note_to_midi(pitch) for pitch in DRUM_PITCHES.values()}
        ]
        for bar_number, pitch in unsupported:
            errors.append(
                _issue(
                    "unsupported_drum_pitch",
                    f"Unsupported drum pitch {pitch}",
                    track_id=track.id,
                    bar_number=bar_number,
                )
            )

        fill_bars = [bar.number for bar in bars if bar.metadata.get("fill")]
        setup_bars = [bar.number for bar in bars if bar.metadata.get("setup")]
        horn_hit_bars = [bar.number for bar in bars if bar.metadata.get("horn_hit_support")]
        signatures = [
            _bar_signature([event for event in bar.events if isinstance(event, NoteEvent)])
            for bar in bars
        ]
        unique_signatures = len(set(signatures))
        density_by_bar = [
            sum(1 for event in bar.events if isinstance(event, NoteEvent)) for bar in bars
        ]
        density_variance = pstdev(density_by_bar) if len(density_by_bar) > 1 else 0.0

        if bars and not fill_bars:
            warnings.append(_issue("missing_fills", "Drum track has no fills", track_id=track.id))
        if len(bars) >= 8 and len(set(signatures)) < 3:
            warnings.append(
                _issue(
                    "flat_drum_language",
                    "Drum track repeats too few bar patterns",
                    track_id=track.id,
                    details={"unique_signatures": unique_signatures},
                )
            )
        if fill_bars and not setup_bars and len(bars) >= 8:
            warnings.append(
                _issue(
                    "missing_setups",
                    "Drum fills are not prepared by setup bars",
                    track_id=track.id,
                )
            )

        metrics = {
            "note_count": len(notes),
            "bar_count": len(bars),
            "fill_bars": fill_bars,
            "fill_bar_count": len(fill_bars),
            "setup_bars": setup_bars,
            "setup_bar_count": len(setup_bars),
            "horn_hit_bars": horn_hit_bars,
            "horn_hit_support_count": len(horn_hit_bars),
            "unique_bar_signatures": unique_signatures,
            "density_stdev": round(density_variance, 3),
            "drum_pitch_count": len(_drum_names(notes)),
        }
        return DrumsValidationReport(
            status="fail" if errors else "pass",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )


def _rule_based_events(
    bar_number: int,
    *,
    grid: tuple[float, ...],
    bar_duration: float,
    groove_style: DrumsGrooveStyle,
    energy: float,
    density: float,
    fill: bool,
    setup: bool,
    break_bar: bool,
    horn_hit: bool,
    groove_map: GrooveMap | None,
) -> list[NoteEvent]:
    if fill:
        return _fill_events(
            bar_number,
            grid=grid,
            groove_style=groove_style,
            energy=energy,
        )

    if groove_style == "bossa":
        events = _bossa_events(grid, energy=energy, density=density)
    elif groove_style == "funk":
        events = _funk_events(grid, energy=energy, density=density)
    elif groove_style == "waltz":
        events = _waltz_events(bar_duration, energy=energy, setup=setup, break_bar=break_bar)
    elif groove_style == "ballad":
        events = _ballad_events(grid, energy=energy, density=density, break_bar=break_bar)
    elif groove_style == "straight_eighth":
        events = _straight_eighth_events(grid, energy=energy, density=density)
    else:
        events = _swing_events(
            bar_number,
            grid=grid,
            energy=energy,
            density=density,
            break_bar=break_bar,
            kick_lock_beats=tuple(groove_map.kick_lock_beats if groove_map else (0.0, 2.0)),
        )

    if setup:
        events.extend(_setup_events(grid, groove_style=groove_style, energy=energy))
    if horn_hit:
        events.extend(_horn_hit_support_events(groove_style=groove_style, energy=energy))
    return _dedupe_events(events)


def _swing_events(
    bar_number: int,
    *,
    grid: tuple[float, ...],
    energy: float,
    density: float,
    break_bar: bool,
    kick_lock_beats: tuple[float, ...],
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    ride_velocity = _velocity(64, 84, energy)
    for start in grid:
        events.append(
            _drum_note(
                "ride",
                start=start,
                velocity=ride_velocity if start % 1.0 == 0 else ride_velocity - 7,
            )
        )
    for start in (1.0, 3.0):
        if start in grid:
            events.append(
                _drum_note(
                    "hihat_pedal",
                    start=start,
                    velocity=_velocity(62, 82, energy),
                    annotations={"timekeeping": True},
                )
            )
    for start in kick_lock_beats:
        if start in grid and (not break_bar or start == 0.0):
            events.append(
                _drum_note(
                    "kick",
                    start=start,
                    velocity=_velocity(38, 60, energy),
                    annotations={"kick_lock": True},
                )
            )

    comping_cells = (
        (1.5, 3.5),
        (0.5, 2.5),
        (1.5, 2.5, 3.5),
        (0.5, 1.5, 3.0),
    )
    comping = comping_cells[(bar_number - 1) % len(comping_cells)]
    comping_limit = 1 if density < 0.62 else 2 if density < 0.82 else 3
    if not break_bar:
        for start in comping[:comping_limit]:
            if start in grid:
                events.append(
                    _drum_note(
                        "snare",
                        start=start,
                        velocity=_velocity(36, 58, energy),
                        annotations={"comping": True},
                    )
                )
    return events


def _bossa_events(
    grid: tuple[float, ...],
    *,
    energy: float,
    density: float,
) -> list[NoteEvent]:
    events = [
        _drum_note(
            "closed_hihat",
            start=start,
            velocity=_velocity(46, 62, energy),
            annotations={"timekeeping": True},
        )
        for start in grid
    ]
    for start in (0.0, 1.5, 2.0, 3.5):
        if start in grid:
            events.append(
                _drum_note(
                    "kick",
                    start=start,
                    velocity=_velocity(48, 72, energy),
                    annotations={"bossa": True},
                )
            )
    for start in (1.0, 3.0):
        if start in grid:
            events.append(
                _drum_note(
                    "snare",
                    start=start,
                    velocity=_velocity(42, 64, energy),
                    annotations={"cross_stick": True},
                )
            )
    if density > 0.72 and 2.5 in grid:
        events.append(
            _drum_note(
                "snare", start=2.5, velocity=_velocity(30, 46, energy), annotations={"ghost": True}
            )
        )
    return events


def _funk_events(
    grid: tuple[float, ...],
    *,
    energy: float,
    density: float,
) -> list[NoteEvent]:
    events = [
        _drum_note(
            "closed_hihat",
            start=start,
            velocity=_velocity(64, 92, energy),
            annotations={"timekeeping": True},
        )
        for start in grid
    ]
    for start in (0.0, 1.5, 2.5):
        if start in grid:
            events.append(
                _drum_note(
                    "kick",
                    start=start,
                    velocity=_velocity(72, 100, energy),
                    annotations={"kick_lock": True},
                )
            )
    for start in (1.0, 3.0):
        if start in grid:
            events.append(
                _drum_note(
                    "snare",
                    start=start,
                    velocity=_velocity(78, 106, energy),
                    annotations={"backbeat": True},
                )
            )
    if density > 0.75 and 3.5 in grid:
        events.append(
            _drum_note(
                "kick", start=3.5, velocity=_velocity(54, 78, energy), annotations={"pickup": True}
            )
        )
    return events


def _straight_eighth_events(
    grid: tuple[float, ...],
    *,
    energy: float,
    density: float,
) -> list[NoteEvent]:
    events = [
        _drum_note(
            "ride", start=start, velocity=_velocity(58, 76, energy), annotations={"straight": True}
        )
        for start in grid
    ]
    for start in (0.0, 2.0):
        if start in grid:
            events.append(
                _drum_note(
                    "kick",
                    start=start,
                    velocity=_velocity(52, 72, energy),
                    annotations={"kick_lock": True},
                )
            )
    for start in (1.0, 3.0):
        if start in grid:
            events.append(
                _drum_note(
                    "snare",
                    start=start,
                    velocity=_velocity(48, 70, energy),
                    annotations={"comping": True},
                )
            )
    if density > 0.8 and 2.5 in grid:
        events.append(
            _drum_note(
                "snare", start=2.5, velocity=_velocity(36, 54, energy), annotations={"ghost": True}
            )
        )
    return events


def _waltz_events(
    bar_duration: float,
    *,
    energy: float,
    setup: bool,
    break_bar: bool,
) -> list[NoteEvent]:
    grid = _grid_for_duration(bar_duration)
    events = [
        _drum_note(
            "ride", start=start, velocity=_velocity(52, 72, energy), annotations={"waltz": True}
        )
        for start in grid
    ]
    events.append(
        _drum_note(
            "kick", start=0.0, velocity=_velocity(46, 64, energy), annotations={"downbeat": True}
        )
    )
    if not break_bar:
        events.append(
            _drum_note(
                "hihat_pedal",
                start=1.0,
                velocity=_velocity(48, 66, energy),
                annotations={"timekeeping": True},
            )
        )
        events.append(
            _drum_note(
                "snare",
                start=2.0,
                velocity=_velocity(38, 58, energy),
                annotations={"waltz_comp": True},
            )
        )
    if setup and 2.5 in grid:
        events.append(
            _drum_note(
                "snare", start=2.5, velocity=_velocity(60, 82, energy), annotations={"setup": True}
            )
        )
    return events


def _ballad_events(
    grid: tuple[float, ...],
    *,
    energy: float,
    density: float,
    break_bar: bool,
) -> list[NoteEvent]:
    events = [
        _drum_note(
            "ride",
            start=start,
            velocity=_velocity(40, 58, energy),
            annotations={"brush_like": True},
        )
        for start in grid
    ]
    for start in (1.0, 3.0):
        if start in grid:
            events.append(
                _drum_note(
                    "hihat_pedal",
                    start=start,
                    velocity=_velocity(38, 52, energy),
                    annotations={"timekeeping": True},
                )
            )
    if not break_bar:
        events.append(
            _drum_note(
                "kick",
                start=0.0,
                velocity=_velocity(34, 48, energy),
                annotations={"feathered": True},
            )
        )
        if density > 0.45 and 2.0 in grid:
            events.append(
                _drum_note(
                    "snare",
                    start=2.0,
                    velocity=_velocity(30, 44, energy),
                    annotations={"soft_comp": True},
                )
            )
    return events


def _fill_events(
    bar_number: int,
    *,
    grid: tuple[float, ...],
    groove_style: DrumsGrooveStyle,
    energy: float,
) -> list[NoteEvent]:
    if groove_style == "bossa":
        cycle = ("snare", "kick", "snare", "high_tom")
    elif groove_style == "funk":
        cycle = ("snare", "kick", "snare", "low_tom", "snare", "kick")
    elif groove_style == "waltz":
        cycle = ("snare", "low_tom", "mid_tom", "snare", "high_tom")
    else:
        cycles = (
            ("snare", "low_tom", "snare", "mid_tom", "snare", "high_tom"),
            ("snare", "snare", "low_tom", "mid_tom", "snare", "high_tom"),
            ("low_tom", "snare", "mid_tom", "snare", "high_tom", "snare"),
        )
        cycle = cycles[(bar_number - 1) % len(cycles)]
    events: list[NoteEvent] = []
    for index, start in enumerate(grid):
        drum_name = cycle[index % len(cycle)]
        if start == grid[-1]:
            drum_name = "crash"
        events.append(
            _drum_note(
                drum_name,
                start=start,
                velocity=_velocity(70, 104 if drum_name == "crash" else 92, energy),
                annotations={"fill": True, "bar": bar_number},
            )
        )
    return events


def _setup_events(
    grid: tuple[float, ...],
    *,
    groove_style: DrumsGrooveStyle,
    energy: float,
) -> list[NoteEvent]:
    starts = [start for start in (2.5, 3.0, 3.5) if start in grid]
    if not starts and len(grid) >= 2:
        starts = [grid[-2]]
    drums = ("snare", "low_tom", "snare") if groove_style != "funk" else ("kick", "snare", "kick")
    return [
        _drum_note(
            drums[index % len(drums)],
            start=start,
            velocity=_velocity(62, 88, energy),
            annotations={"setup": True},
        )
        for index, start in enumerate(starts)
    ]


def _horn_hit_support_events(
    *,
    groove_style: DrumsGrooveStyle,
    energy: float,
) -> list[NoteEvent]:
    drum_name = "crash" if groove_style in {"swing", "waltz", "ballad"} else "snare"
    return [
        _drum_note(
            "kick",
            start=0.0,
            velocity=_velocity(62, 88, energy),
            annotations={"horn_hit_support": True},
        ),
        _drum_note(
            drum_name,
            start=0.0,
            velocity=_velocity(70, 100, energy),
            annotations={"horn_hit_support": True},
        ),
    ]


def _events_from_pattern(
    pattern: dict[str, Any],
    *,
    bar_duration: float,
    groove_style: DrumsGrooveStyle,
    grid: tuple[float, ...],
    energy: float,
) -> list[NoteEvent]:
    if bar_duration != 4.0:
        return []
    payload = _learned_payload(pattern)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []

    events = _timekeeping_events(groove_style, grid=grid, energy=energy)
    learned_count = 0
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        try:
            start = _quantized_half_beat(float(raw_event["beat"]))
            midi_pitch = int(raw_event["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        drum_name = _drum_name_for_midi(midi_pitch)
        if drum_name is None or start not in grid:
            continue
        learned_count += 1
        events.append(
            _drum_note(
                drum_name,
                start=start,
                velocity=int(raw_event.get("velocity", _velocity(58, 82, energy)) or 72),
                annotations={
                    "drum": "learned",
                    "learned_drum": drum_name,
                    "learned_pattern_id": pattern.get("id"),
                },
            )
        )
    if learned_count == 0:
        return []
    return _dedupe_events(events)


def _timekeeping_events(
    groove_style: DrumsGrooveStyle,
    *,
    grid: tuple[float, ...],
    energy: float,
) -> list[NoteEvent]:
    drum_name = "closed_hihat" if groove_style in {"bossa", "funk"} else "ride"
    return [
        _drum_note(
            drum_name,
            start=start,
            velocity=_velocity(48, 74, energy),
            annotations={"timekeeping": True, "retrieval_backbone": True},
        )
        for start in grid
    ]


def _drum_note(
    drum_name: str,
    *,
    start: float,
    velocity: int,
    annotations: dict[str, object] | None = None,
) -> NoteEvent:
    return NoteEvent(
        pitch=DRUM_PITCHES[drum_name],
        start=start,
        duration=0.5,
        velocity=max(1, min(127, velocity)),
        annotations={"drum": drum_name, **(annotations or {})},
    )


def _dedupe_events(events: list[NoteEvent]) -> list[NoteEvent]:
    priority = {
        "crash": 0,
        "snare": 1,
        "kick": 2,
        "hihat_pedal": 3,
        "closed_hihat": 4,
        "ride": 5,
    }
    best: dict[tuple[float, str], NoteEvent] = {}
    for event in events:
        drum_name = str(event.annotations.get("drum", ""))
        key = (event.start, drum_name)
        current = best.get(key)
        if current is None or event.velocity > current.velocity:
            best[key] = event
    return sorted(
        best.values(),
        key=lambda event: (
            event.start,
            priority.get(str(event.annotations.get("drum", "")), 20),
            event.pitch,
        ),
    )


def _drums_mode(context: Any) -> DrumsEngineMode:
    raw = context.spec.constraints.get("drums_engine_mode") or context.spec.constraints.get(
        "drums_mode"
    )
    if raw in {"rule_based", "retrieval"}:
        return raw
    if (
        context.spec.constraints.get("drums_retrieval", True) is not False
        and _select_pattern(context) is not None
    ):
        return "retrieval"
    return "rule_based"


def _song_plan(context: Any) -> SongPlan | None:
    song_plan = getattr(context, "song_plan", None)
    return song_plan if isinstance(song_plan, SongPlan) else None


def _groove_map(context: Any) -> GrooveMap | None:
    song_plan = _song_plan(context)
    if song_plan is not None:
        return song_plan.groove_map
    raw = context.project.metadata.get("song_plan")
    if isinstance(raw, dict) and isinstance(raw.get("groove_map"), dict):
        try:
            return GrooveMap.model_validate(raw["groove_map"])
        except Exception:
            return None
    return None


def _groove_style(context: Any, groove_map: GrooveMap | None) -> DrumsGrooveStyle:
    raw = context.spec.constraints.get("drums_groove") or context.spec.constraints.get("feel")
    if raw in {"swing", "bossa", "funk", "waltz", "ballad", "straight_eighth"}:
        return "funk" if raw == "funk" else raw
    feel = groove_map.feel if groove_map else ""
    if feel == "bossa":
        return "bossa"
    if feel in {"straight_eighth", "funk"} or context.spec.style == "funk_jazz":
        return "funk"
    if feel == "waltz" or context.spec.meter == "3/4" or context.spec.style == "jazz_waltz":
        return "waltz"
    if feel == "slow_swing" or context.spec.style == "jazz_ballad":
        return "ballad"
    return "swing"


def _section_for_bar(song_plan: SongPlan | None, bar_number: int) -> SectionPlan | None:
    if song_plan is None:
        return None
    for section in song_plan.sections:
        if section.start_bar <= bar_number <= section.end_bar:
            return section
    return None


def _bar_energy(song_plan: SongPlan | None, section: SectionPlan | None, bar_number: int) -> float:
    if section is None:
        return 0.58
    phrase_lift = 0.04 if bar_number == section.end_bar else 0.0
    return max(0.0, min(1.0, section.energy + phrase_lift))


def _drum_density(context: Any, section: SectionPlan | None, energy: float) -> float:
    if section is not None:
        raw = section.role_densities.get("drums")
        if raw is not None:
            return max(0.0, min(1.0, float(raw)))
    density = context.spec.density
    base = {"low": 0.45, "medium": 0.66, "high": 0.86}.get(str(density), 0.66)
    return max(0.0, min(1.0, base + energy * 0.12))


def _is_fill_bar(
    groove_map: GrooveMap | None,
    project: ArrangementProject,
    bar_number: int,
) -> bool:
    if groove_map is not None and bar_number in set(groove_map.fill_bars):
        return True
    if bar_number == project.bar_count:
        return True
    return bar_number % 4 == 0


def _is_setup_bar(groove_map: GrooveMap | None, bar_number: int) -> bool:
    return groove_map is not None and bar_number in set(groove_map.setup_bars)


def _is_break_bar(groove_map: GrooveMap | None, bar_number: int) -> bool:
    return groove_map is not None and bar_number in set(groove_map.break_bars)


def _is_horn_hit_bar(groove_map: GrooveMap | None, bar_number: int) -> bool:
    return groove_map is not None and bar_number in set(groove_map.horn_hit_bars)


def _select_pattern(context: Any) -> dict[str, Any] | None:
    return retrieve_pattern(
        context,
        category="drum_grooves",
        role="drums",
        instrument="drum_kit",
        density=context.spec.density,
    )


def _grid_for_duration(bar_duration: float) -> tuple[float, ...]:
    steps = max(1, round(bar_duration / 0.5))
    return tuple(round(index * 0.5, 3) for index in range(steps))


def _bar_duration(project: ArrangementProject, bar_number: int) -> float:
    from arranger_core.schema import meter_to_quarter_beats

    return meter_to_quarter_beats(project.meter_at_bar(bar_number))


def _velocity(low: int, high: int, energy: float) -> int:
    return round(low + (high - low) * max(0.0, min(1.0, energy)))


def _drum_name_for_midi(midi_pitch: int) -> str | None:
    for drum_name, pitch in DRUM_PITCHES.items():
        if note_to_midi(pitch) == midi_pitch:
            return drum_name
    return None


def _quantized_half_beat(value: float) -> float:
    return round(round(value * 2) / 2, 3)


def _drum_names(events: list[NoteEvent]) -> set[str]:
    return {str(event.annotations.get("drum")) for event in events if event.annotations.get("drum")}


def _bar_signature(events: list[NoteEvent]) -> str:
    return "|".join(
        f"{event.start:g}:{event.annotations.get('drum', event.pitch)}"
        for event in sorted(events, key=lambda item: (item.start, item.pitch))
    )


def _learned_payload(pattern: dict[str, Any] | None) -> dict[str, Any]:
    if pattern is None:
        return {}
    payload = pattern.get("payload")
    return payload if isinstance(payload, dict) else {}


def _issue(
    code: str,
    message: str,
    *,
    track_id: str | None = None,
    bar_number: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "track_id": track_id,
        "bar_number": bar_number,
        "details": details or {},
    }


def generate_drums_track(
    spec: GenerationSpec,
    project: ArrangementProject,
    *,
    context: Any,
) -> Track:
    _ = spec, project
    return DrumsEngine().generate(context)
