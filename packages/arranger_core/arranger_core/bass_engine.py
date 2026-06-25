from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.music_theory import midi_to_note, note_to_midi, pitch_class, pitch_class_name
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    NoteEvent,
    Track,
    meter_to_quarter_beats,
)

BASS_ENGINE_VERSION = "0.1.0"
BassEngineMode = Literal["rule_based", "retrieval", "statistical", "ai_infill"]
BassLineStyle = Literal[
    "walking_bass",
    "two_feel",
    "pedal_modal",
    "bossa_bass",
    "waltz_bass",
]
BassSource = Literal[
    "rule_based",
    "retrieval",
    "statistical",
    "ai_infill",
    "fallback_rule_based",
]


class BassEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BassChordInfo(BassEngineModel):
    symbol: str
    root: str
    root_pc: int
    quality: str
    chord_tone_pcs: tuple[int, ...]
    guide_tone_pcs: tuple[int, ...]
    tension_pcs: tuple[int, ...]
    prefer_sharps: bool


class BassLineLedgerEntry(BassEngineModel):
    bar_number: int
    chord: str
    next_chord: str
    mode: BassEngineMode
    style: BassLineStyle
    source: BassSource
    starts: list[float]
    contour: list[int]
    source_pattern_id: str | None = None
    accepted: bool = True
    rejection_reason: str | None = None


class BassLineLedger(BassEngineModel):
    schema_version: str = BASS_ENGINE_VERSION
    entries: list[BassLineLedgerEntry] = Field(default_factory=list)

    def add(self, entry: BassLineLedgerEntry) -> None:
        self.entries.append(entry)


class BassValidationReport(BassEngineModel):
    status: Literal["pass", "fail"]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class BassAiBackend(Protocol):
    def generate_bass_track(
        self,
        *,
        project: ArrangementProject,
        base_track: Track,
        context: Any,
    ) -> Track:
        ...


class BassEngine:
    def __init__(
        self,
        *,
        chord_parser: ChordParser | None = None,
        ai_backend: BassAiBackend | None = None,
    ) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()
        self.ai_backend = ai_backend

    def generate(self, context: Any) -> Track:
        mode = _bass_mode(context)
        style = _bass_style(context)
        selected_mode = mode
        fallback_reason: str | None = None

        base_track = self._rule_based_track(context, style=style)
        track = base_track
        source: BassSource = "rule_based"
        source_pattern = None

        if mode == "retrieval":
            source_pattern = _select_pattern(context, "walking_bass_cells", "walking_bass")
            retrieved = self._retrieval_track(context, style=style, pattern=source_pattern)
            if retrieved is None:
                selected_mode = "rule_based"
                fallback_reason = "retrieval_pattern_unavailable"
            else:
                report = self.validate_track(context.project, retrieved)
                if report.status == "pass":
                    track = retrieved
                    source = "retrieval"
                else:
                    selected_mode = "rule_based"
                    fallback_reason = "retrieval_validation_failed"
        elif mode == "statistical":
            statistical = self._statistical_track(context, style=style)
            report = self.validate_track(context.project, statistical)
            if report.status == "pass":
                track = statistical
                source = "statistical"
            else:
                selected_mode = "rule_based"
                fallback_reason = "statistical_validation_failed"
        elif mode == "ai_infill":
            ai_result = self._ai_track(context, base_track)
            if ai_result["track"] is not None:
                track = ai_result["track"]
                source = "ai_infill"
            else:
                selected_mode = "rule_based"
                fallback_reason = str(ai_result["fallback_reason"])

        validation = self.validate_track(context.project, track)
        if validation.status == "fail" and track is not base_track:
            track = base_track
            selected_mode = "rule_based"
            source = "fallback_rule_based"
            fallback_reason = "bass_validation_failed"
            validation = self.validate_track(context.project, track)

        ledger = _build_ledger(
            context,
            track,
            mode=selected_mode,
            style=style,
            source=source,
            source_pattern=source_pattern if source == "retrieval" else None,
            fallback_reason=fallback_reason,
        )
        return _finalize_track(
            track,
            context=context,
            mode=selected_mode,
            style=style,
            source=source,
            validation=validation,
            ledger=ledger,
            source_pattern=source_pattern if source == "retrieval" else None,
            fallback_reason=fallback_reason,
        )

    def validate_track(
        self,
        project: ArrangementProject,
        track: Track,
    ) -> BassValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        notes_by_bar = {
            bar.number: [
                event
                for event in sorted(bar.events, key=lambda item: item.start)
                if isinstance(event, NoteEvent)
            ]
            for bar in track.bars
        }
        all_notes = [event for events in notes_by_bar.values() for event in events]

        if not all_notes:
            errors.append(_issue("no_bass_notes", "Bass track has no notes"))

        low = note_to_midi("E1")
        high = note_to_midi("C4")
        for bar_number, events in notes_by_bar.items():
            for event in events:
                midi_note = note_to_midi(event.pitch)
                if midi_note < low or midi_note > high:
                    errors.append(
                        _issue(
                            "bass_range",
                            f"{event.pitch} outside double bass practical range",
                            track_id=track.id,
                            bar_number=bar_number,
                        )
                    )

        duration_issues = ArrangementProject(
            project_id=f"{project.project_id}-bass-validation",
            generation_spec=project.generation_spec,
            tempo_map=project.tempo_map,
            key_map=project.key_map,
            meter_map=project.meter_map,
            form=project.form,
            chord_grid=project.chord_grid,
            tracks=[track],
        ).validate_bar_durations()
        for duration_issue in duration_issues:
            errors.append(
                _issue(
                    "bass_duration",
                    duration_issue.message,
                    track_id=track.id,
                    bar_number=duration_issue.bar_number,
                )
            )

        chords_by_bar = _chords_by_bar(project.chord_grid)
        root_hits = 0
        checked_downbeats = 0
        supported_notes = 0
        checked_harmony = 0
        for bar in track.bars:
            first_note = notes_by_bar.get(bar.number, [None])[0]
            chord = _active_chord(chords_by_bar, bar.number, 0.0)
            chord_info = _parse_chord(self.chord_parser, chord.symbol)
            if first_note is not None:
                checked_downbeats += 1
                if note_to_midi(first_note.pitch) % 12 == chord_info.root_pc:
                    root_hits += 1
            for event in notes_by_bar.get(bar.number, []):
                checked_harmony += 1
                if _is_bass_harmonically_supported(event, chord_info):
                    supported_notes += 1

        root_ratio = root_hits / checked_downbeats if checked_downbeats else 0.0
        if checked_downbeats and root_ratio < 0.9:
            errors.append(
                _issue(
                    "bass_downbeat_root",
                    f"Bass hits roots on too few downbeats ({root_ratio:.2f})",
                    track_id=track.id,
                    details={"ratio": round(root_ratio, 3)},
                )
            )

        support_ratio = supported_notes / checked_harmony if checked_harmony else 0.0
        if checked_harmony and support_ratio < 0.5:
            errors.append(
                _issue(
                    "bass_harmony",
                    f"Bass harmonic support is too low ({support_ratio:.2f})",
                    track_id=track.id,
                    details={"ratio": round(support_ratio, 3)},
                )
            )

        leaps = _melodic_leaps(track)
        max_leap = max(leaps, default=0)
        erratic_leaps = [leap for leap in leaps if leap > 14]
        if erratic_leaps:
            errors.append(
                _issue(
                    "bass_erratic_contour",
                    "Bass line contains erratic leaps",
                    track_id=track.id,
                    details={"max_leap": max_leap, "erratic_leap_count": len(erratic_leaps)},
                )
            )

        approach_ratio = _approach_resolution_ratio(
            project,
            track,
            chord_parser=self.chord_parser,
        )
        if approach_ratio is not None and approach_ratio < 0.65:
            warnings.append(
                _issue(
                    "bass_approach_resolution",
                    f"Bass approaches next roots weakly ({approach_ratio:.2f})",
                    track_id=track.id,
                    details={"ratio": round(approach_ratio, 3)},
                )
            )

        metrics = {
            "note_count": len(all_notes),
            "root_on_downbeat_ratio": round(root_ratio, 3),
            "chord_support_ratio": round(support_ratio, 3),
            "approach_resolution_ratio": (
                round(approach_ratio, 3) if approach_ratio is not None else None
            ),
            "max_leap_semitones": max_leap,
            "average_leap_semitones": (
                round(sum(leaps) / len(leaps), 3) if leaps else 0.0
            ),
            "erratic_leap_count": len(erratic_leaps),
            "bar_count": len(track.bars),
        }
        return BassValidationReport(
            status="fail" if errors else "pass",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def _rule_based_track(self, context: Any, *, style: BassLineStyle) -> Track:
        return _build_bass_track(
            context,
            mode="rule_based",
            style=style,
            source="rule_based",
            chord_parser=self.chord_parser,
            pattern=None,
            statistical=False,
        )

    def _retrieval_track(
        self,
        context: Any,
        *,
        style: BassLineStyle,
        pattern: dict[str, Any] | None,
    ) -> Track | None:
        if pattern is None:
            return None
        return _build_bass_track(
            context,
            mode="retrieval",
            style=style,
            source="retrieval",
            chord_parser=self.chord_parser,
            pattern=pattern,
            statistical=False,
        )

    def _statistical_track(self, context: Any, *, style: BassLineStyle) -> Track:
        return _build_bass_track(
            context,
            mode="statistical",
            style=style,
            source="statistical",
            chord_parser=self.chord_parser,
            pattern=None,
            statistical=True,
        )

    def _ai_track(self, context: Any, base_track: Track) -> dict[str, Any]:
        if self.ai_backend is None:
            return {"track": None, "fallback_reason": "ai_backend_unavailable"}
        try:
            track = self.ai_backend.generate_bass_track(
                project=context.project,
                base_track=base_track,
                context=context,
            )
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            return {"track": None, "fallback_reason": f"ai_backend_error:{exc}"}

        report = self.validate_track(context.project, track)
        if report.status != "pass":
            return {
                "track": None,
                "fallback_reason": "ai_validation_failed",
                "validation": report.model_dump(mode="json"),
            }
        return {"track": track, "fallback_reason": None, "validation": report}


def _build_bass_track(
    context: Any,
    *,
    mode: BassEngineMode,
    style: BassLineStyle,
    source: BassSource,
    chord_parser: ChordParser,
    pattern: dict[str, Any] | None,
    statistical: bool,
) -> Track:
    chords_by_bar = _chords_by_bar(context.project.chord_grid)
    previous_midi = note_to_midi("C2")
    bars: list[Bar] = []
    for bar_number in range(1, context.project.bar_count + 1):
        bar_duration = _bar_duration(context.project, bar_number)
        chord = _active_chord(chords_by_bar, bar_number, 0.0)
        next_chord = _first_chord_after(chords_by_bar, bar_number, context.project.bar_count)
        chord_info = _parse_chord(chord_parser, chord.symbol)
        next_info = _parse_chord(chord_parser, next_chord.symbol)
        line = _bass_line_for_bar(
            chord_info,
            next_info,
            bar_number=bar_number,
            bar_duration=bar_duration,
            previous_midi=previous_midi,
            style=style,
            pattern=pattern,
            statistical=statistical,
        )
        previous_midi = line[-1].midi_note if line else previous_midi
        bars.append(
            Bar(
                number=bar_number,
                events=[
                    NoteEvent(
                        pitch=item.pitch,
                        start=item.start,
                        duration=item.duration,
                        velocity=item.velocity,
                        articulations=item.articulations,
                        annotations={
                            "bass_role": item.role,
                            "source_chord": chord.symbol,
                            "next_chord": next_chord.symbol,
                            "bass_engine_mode": mode,
                            "bass_line_style": style,
                            "bass_source": source,
                            "target_next_root_pc": next_info.root_pc,
                            "learned_pattern_id": (
                                pattern.get("id") if pattern is not None else None
                            ),
                        },
                    )
                    for item in line
                ],
                metadata={
                    "source_chord": chord.symbol,
                    "next_chord": next_chord.symbol,
                    "bass_line_style": style,
                    "bass_engine_mode": mode,
                    "learned_pattern_id": pattern.get("id") if pattern else None,
                },
            )
        )
    return Track(
        id="double_bass",
        instrument="double_bass",
        role="walking_bass",
        bars=bars,
        metadata={
            "generator": "BassEngine",
            "bass_engine_version": BASS_ENGINE_VERSION,
            "bass_engine_mode": mode,
            "bass_line_style": style,
            "learned_pattern_id": pattern.get("id") if pattern else None,
        },
    )


class _BassNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pitch: str
    midi_note: int
    start: float
    duration: float
    velocity: int
    role: str
    articulations: list[str] = Field(default_factory=list)


def _bass_line_for_bar(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    *,
    bar_number: int,
    bar_duration: float,
    previous_midi: int,
    style: BassLineStyle,
    pattern: dict[str, Any] | None,
    statistical: bool,
) -> list[_BassNote]:
    if pattern is not None and style == "walking_bass" and bar_duration == 4.0:
        line = _line_from_pattern(
            chord_info,
            next_info,
            pattern,
            previous_midi=previous_midi,
        )
        if line:
            return line

    if style == "bossa_bass" and bar_duration == 4.0:
        specs = _bossa_specs(chord_info, next_info, bar_number)
    elif style == "two_feel" and bar_duration == 4.0:
        specs = _two_feel_specs(chord_info, next_info, bar_number)
    elif style == "pedal_modal" and bar_duration == 4.0:
        specs = _pedal_modal_specs(chord_info, next_info, bar_number)
    elif style == "waltz_bass" or round(bar_duration) == 3:
        specs = _waltz_specs(chord_info, next_info, bar_duration)
    else:
        specs = _walking_specs(chord_info, next_info, bar_duration, statistical=statistical)
    return _specs_to_notes(
        specs,
        previous_midi=previous_midi,
        prefer_sharps=chord_info.prefer_sharps,
    )


def _walking_specs(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    bar_duration: float,
    *,
    statistical: bool,
) -> list[tuple[int, float, float, str, int, list[str]]]:
    beats = max(1, round(bar_duration))
    if beats == 1:
        return [(chord_info.root_pc, 0.0, bar_duration, "root", 74, ["tenuto"])]
    third = _pc_at_interval(chord_info, (3, 4, 10))
    fifth = _pc_at_interval(chord_info, (7, 5))
    seventh = _pc_at_interval(chord_info, (10, 11, 6))
    approach = _approach_pc(
        next_info.root_pc,
        previous_pc=seventh if not statistical else fifth,
        prefer_from_below=not statistical,
    )
    pool = [chord_info.root_pc, third, fifth, approach]
    if statistical:
        pool = _statistical_pool(chord_info, next_info)
    return [
        (
            pool[min(index, len(pool) - 1)],
            float(index),
            min(1.0, bar_duration - index),
            _bass_role(index, beats),
            76 if index == 0 else 69,
            ["tenuto"] if index == 0 else [],
        )
        for index in range(beats)
    ]


def _two_feel_specs(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    bar_number: int,
) -> list[tuple[int, float, float, str, int, list[str]]]:
    second = (
        _approach_pc(next_info.root_pc, previous_pc=chord_info.root_pc)
        if bar_number % 2 == 0
        else _pc_at_interval(chord_info, (7, 5))
    )
    second_role = "approach_next_root" if bar_number % 2 == 0 else "fifth"
    return [
        (chord_info.root_pc, 0.0, 2.0, "root", 72, ["tenuto"]),
        (second, 2.0, 2.0, second_role, 66, []),
    ]


def _pedal_modal_specs(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    bar_number: int,
) -> list[tuple[int, float, float, str, int, list[str]]]:
    color = _pc_at_interval(chord_info, (7, 14, 10, 3))
    approach = (
        chord_info.root_pc
        if next_info.root_pc == chord_info.root_pc
        else _approach_pc(next_info.root_pc, previous_pc=color)
    )
    middle = color if bar_number % 2 else chord_info.root_pc
    return [
        (chord_info.root_pc, 0.0, 1.0, "pedal_root", 73, ["tenuto"]),
        (middle, 1.0, 1.0, "modal_color", 65, []),
        (chord_info.root_pc, 2.0, 1.0, "pedal_root", 70, []),
        (approach, 3.0, 1.0, "approach_next_root", 66, []),
    ]


def _bossa_specs(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    bar_number: int,
) -> list[tuple[int, float, float, str, int, list[str]]]:
    fifth = _pc_at_interval(chord_info, (7, 5))
    last = (
        _approach_pc(next_info.root_pc, previous_pc=fifth, prefer_from_below=bar_number % 2 == 1)
        if next_info.root_pc != chord_info.root_pc
        else fifth
    )
    return [
        (chord_info.root_pc, 0.0, 1.5, "root", 72, ["tenuto"]),
        (fifth, 1.5, 1.0, "fifth", 64, []),
        (last, 2.5, 1.5, "approach_next_root", 67, []),
    ]


def _waltz_specs(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    bar_duration: float,
) -> list[tuple[int, float, float, str, int, list[str]]]:
    beats = max(1, round(bar_duration))
    third_or_fifth = _pc_at_interval(chord_info, (7, 3, 4))
    approach = _approach_pc(next_info.root_pc, previous_pc=third_or_fifth)
    pcs = [chord_info.root_pc, third_or_fifth, approach]
    return [
        (
            pcs[min(index, len(pcs) - 1)],
            float(index),
            min(1.0, bar_duration - index),
            _bass_role(index, beats),
            72 if index == 0 else 66,
            ["tenuto"] if index == 0 else [],
        )
        for index in range(beats)
    ]


def _line_from_pattern(
    chord_info: BassChordInfo,
    next_info: BassChordInfo,
    pattern: dict[str, Any],
    *,
    previous_midi: int,
) -> list[_BassNote]:
    payload = _learned_payload(pattern)
    raw_intervals = payload.get("pitch_intervals")
    if not isinstance(raw_intervals, list):
        return []

    intervals: list[int] = []
    for value in raw_intervals[:4]:
        try:
            intervals.append(int(value))
        except (TypeError, ValueError):
            continue
    if len(intervals) < 3:
        return []
    intervals = [0, *intervals[1:4]]
    intervals.extend([7, 10][len(intervals) - 2 :])
    pcs = [(chord_info.root_pc + interval) % 12 for interval in intervals[:4]]
    pcs[-1] = _approach_pc(next_info.root_pc, previous_pc=pcs[-2])
    specs = [
        (
            pc,
            float(index),
            1.0,
            _bass_role(index, 4),
            74 if index == 0 else 68,
            ["tenuto"] if index == 0 else [],
        )
        for index, pc in enumerate(pcs)
    ]
    return _specs_to_notes(
        specs,
        previous_midi=previous_midi,
        prefer_sharps=chord_info.prefer_sharps,
    )


def _specs_to_notes(
    specs: list[tuple[int, float, float, str, int, list[str]]],
    *,
    previous_midi: int,
    prefer_sharps: bool,
) -> list[_BassNote]:
    notes: list[_BassNote] = []
    anchor = previous_midi
    for pc, start, duration, role, velocity, articulations in specs:
        note_name, anchor = _nearest_note_in_range(
            pc,
            low_midi=note_to_midi("E1"),
            high_midi=note_to_midi("C4"),
            anchor_midi=anchor,
            prefer_sharps=prefer_sharps,
        )
        notes.append(
            _BassNote(
                pitch=note_name,
                midi_note=anchor,
                start=start,
                duration=max(0.25, duration),
                velocity=velocity,
                role=role,
                articulations=list(articulations),
            )
        )
    return notes


def _bass_mode(context: Any) -> BassEngineMode:
    raw = context.spec.constraints.get("bass_engine_mode")
    if raw is None:
        raw = context.spec.constraints.get("bass_mode")
    if raw in {"rule_based", "retrieval", "statistical", "ai_infill"}:
        return raw
    if (
        context.spec.constraints.get("bass_retrieval", True) is not False
        and _select_pattern(context, "walking_bass_cells", "walking_bass") is not None
    ):
        return "retrieval"
    return "rule_based"


def _bass_style(context: Any) -> BassLineStyle:
    raw = context.spec.constraints.get("bass_line_style") or context.spec.constraints.get(
        "bass_style"
    )
    if raw in {"walking_bass", "two_feel", "pedal_modal", "bossa_bass", "waltz_bass"}:
        return raw
    feel = context.spec.constraints.get("feel")
    if context.spec.meter == "3/4" or context.spec.style == "jazz_waltz":
        return "waltz_bass"
    if context.spec.style == "bossa_nova" or feel == "bossa":
        return "bossa_bass"
    if context.spec.style == "modal_jazz" or context.spec.form.startswith("modal"):
        return "pedal_modal"
    if context.spec.style == "jazz_ballad" or context.spec.density == "low":
        return "two_feel"
    return "walking_bass"


def _select_pattern(context: Any, category: str, role: str) -> dict[str, Any] | None:
    candidates = [
        pattern
        for pattern in context.learned_patterns.get(category, [])
        if pattern.get("role") == role
    ]
    if not candidates:
        return None
    style_matches = [
        pattern
        for pattern in candidates
        if pattern.get("style") in {context.spec.style, "unknown"}
    ]
    selected_pool = style_matches or candidates
    return selected_pool[context.spec.seed % len(selected_pool)]


def _build_ledger(
    context: Any,
    track: Track,
    *,
    mode: BassEngineMode,
    style: BassLineStyle,
    source: BassSource,
    source_pattern: dict[str, Any] | None,
    fallback_reason: str | None,
) -> BassLineLedger:
    ledger = BassLineLedger()
    chords_by_bar = _chords_by_bar(context.project.chord_grid)
    for bar in track.bars:
        chord = _active_chord(chords_by_bar, bar.number, 0.0)
        next_chord = _first_chord_after(chords_by_bar, bar.number, context.project.bar_count)
        notes = [event for event in bar.events if isinstance(event, NoteEvent)]
        contour = _contour([note_to_midi(event.pitch) for event in notes])
        ledger.add(
            BassLineLedgerEntry(
                bar_number=bar.number,
                chord=chord.symbol,
                next_chord=next_chord.symbol,
                mode=mode,
                style=style,
                source=source,
                starts=[event.start for event in notes],
                contour=contour,
                source_pattern_id=source_pattern.get("id") if source_pattern else None,
                accepted=fallback_reason is None,
                rejection_reason=fallback_reason,
            )
        )
    return ledger


def _finalize_track(
    track: Track,
    *,
    context: Any,
    mode: BassEngineMode,
    style: BassLineStyle,
    source: BassSource,
    validation: BassValidationReport,
    ledger: BassLineLedger,
    source_pattern: dict[str, Any] | None,
    fallback_reason: str | None,
) -> Track:
    return track.model_copy(
        update={
            "metadata": {
                **track.metadata,
                "generator": "BassEngine",
                "bass_engine_version": BASS_ENGINE_VERSION,
                "bass_engine_mode": mode,
                "bass_line_style": style,
                "bass_source": source,
                "bass_validation": validation.model_dump(mode="json"),
                "bass_line_ledger": ledger.model_dump(mode="json"),
                "learned_pattern_id": source_pattern.get("id") if source_pattern else None,
                "fallback_reason": fallback_reason,
                "style": context.spec.style,
            }
        }
    )


def _parse_chord(parser: ChordParser, symbol: str) -> BassChordInfo:
    try:
        parsed = parser.parse(symbol)
    except ValueError:
        root = _fallback_root(symbol)
        root_pc = pitch_class(root)
        return BassChordInfo(
            symbol=symbol,
            root=root,
            root_pc=root_pc,
            quality="major_triad",
            chord_tone_pcs=(root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12),
            guide_tone_pcs=((root_pc + 4) % 12, (root_pc + 10) % 12),
            tension_pcs=(),
            prefer_sharps="#" in root and "b" not in root,
        )
    return BassChordInfo(
        symbol=symbol,
        root=parsed.root,
        root_pc=parsed.root_pc,
        quality=parsed.quality,
        chord_tone_pcs=tuple(parsed.chord_tone_pcs) or (parsed.root_pc,),
        guide_tone_pcs=_guide_tones(parsed),
        tension_pcs=tuple(parsed.tension_pcs),
        prefer_sharps="#" in parsed.root and "b" not in parsed.root,
    )


def _guide_tones(parsed: ParsedChord) -> tuple[int, ...]:
    intervals = [
        interval
        for interval in parsed.chord_tone_intervals
        if interval % 12 in {3, 4, 10, 11}
    ]
    return tuple((parsed.root_pc + interval) % 12 for interval in intervals)


def _fallback_root(chord_symbol: str) -> str:
    for length in (2, 1):
        root = chord_symbol[:length]
        try:
            pitch_class(root)
            return root
        except ValueError:
            continue
    return pitch_class_name(0)


def _pc_at_interval(chord_info: BassChordInfo, intervals: tuple[int, ...]) -> int:
    wanted = {(chord_info.root_pc + interval) % 12 for interval in intervals}
    for pc in (*chord_info.chord_tone_pcs, *chord_info.tension_pcs):
        if pc in wanted:
            return pc
    return chord_info.chord_tone_pcs[0]


def _approach_pc(
    target_pc: int,
    *,
    previous_pc: int,
    prefer_from_below: bool = True,
) -> int:
    below = (target_pc - 1) % 12
    above = (target_pc + 1) % 12
    candidates = (below, above) if prefer_from_below else (above, below)
    return min(
        candidates,
        key=lambda pc: (_pc_distance(pc, previous_pc), 0 if pc == candidates[0] else 1),
    )


def _pc_distance(first: int, second: int) -> int:
    diff = abs((first - second) % 12)
    return min(diff, 12 - diff)


def _statistical_pool(chord_info: BassChordInfo, next_info: BassChordInfo) -> list[int]:
    guide = _pc_at_interval(chord_info, (3, 4, 10, 11))
    fifth = _pc_at_interval(chord_info, (7, 5))
    approach = _approach_pc(next_info.root_pc, previous_pc=fifth, prefer_from_below=False)
    return [chord_info.root_pc, fifth, guide, approach]


def _bass_role(index: int, beats: int) -> str:
    if index == 0:
        return "root"
    if index == beats - 1:
        return "approach_next_root"
    return "walking"


def _chords_by_bar(chord_grid: list[ChordSymbol]) -> dict[int, list[ChordSymbol]]:
    grouped: dict[int, list[ChordSymbol]] = {}
    for chord in chord_grid:
        if chord.bar is None:
            continue
        grouped.setdefault(chord.bar, []).append(chord)
    for chords in grouped.values():
        chords.sort(key=lambda chord: chord.beat)
    return grouped


def _active_chord(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    start: float,
) -> ChordSymbol:
    chords = chords_by_bar.get(bar_number, [])
    if not chords:
        return ChordSymbol(symbol="C", bar=bar_number, beat=1.0)
    active = chords[0]
    for chord in chords:
        if chord.beat - 1.0 <= start + 1e-6:
            active = chord
        else:
            break
    return active


def _first_chord_after(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    max_bar: int,
) -> ChordSymbol:
    for next_bar in range(bar_number + 1, max_bar + 1):
        chords = chords_by_bar.get(next_bar, [])
        if chords:
            return chords[0]
    return _active_chord(chords_by_bar, bar_number, 0.0)


def _bar_duration(project: ArrangementProject, bar_number: int) -> float:
    return meter_to_quarter_beats(project.meter_at_bar(bar_number))


def _nearest_note_in_range(
    target_pc: int,
    *,
    low_midi: int,
    high_midi: int,
    anchor_midi: int,
    prefer_sharps: bool,
) -> tuple[str, int]:
    candidates = [
        midi_note
        for midi_note in range(low_midi, high_midi + 1)
        if midi_note % 12 == target_pc % 12
    ]
    if not candidates:
        fallback = min(max(anchor_midi, low_midi), high_midi)
        return midi_to_note(fallback, prefer_sharps=prefer_sharps), fallback
    selected = min(candidates, key=lambda midi_note: abs(midi_note - anchor_midi))
    return midi_to_note(selected, prefer_sharps=prefer_sharps), selected


def _is_bass_harmonically_supported(event: NoteEvent, chord_info: BassChordInfo) -> bool:
    pc = note_to_midi(event.pitch) % 12
    if pc in {*chord_info.chord_tone_pcs, *chord_info.tension_pcs}:
        return True
    role = str(event.annotations.get("bass_role") or "")
    return role.startswith("approach")


def _melodic_leaps(track: Track) -> list[int]:
    notes = [
        event
        for bar in track.bars
        for event in sorted(bar.events, key=lambda item: item.start)
        if isinstance(event, NoteEvent)
    ]
    return [
        abs(note_to_midi(current.pitch) - note_to_midi(previous.pitch))
        for previous, current in zip(notes, notes[1:], strict=False)
    ]


def _approach_resolution_ratio(
    project: ArrangementProject,
    track: Track,
    *,
    chord_parser: ChordParser,
) -> float | None:
    chords_by_bar = _chords_by_bar(project.chord_grid)
    checked = 0
    supported = 0
    bar_notes = {
        bar.number: [
            event
            for event in sorted(bar.events, key=lambda item: item.start)
            if isinstance(event, NoteEvent)
        ]
        for bar in track.bars
    }
    for bar in track.bars[:-1]:
        notes = bar_notes.get(bar.number, [])
        if not notes:
            continue
        next_chord = _first_chord_after(chords_by_bar, bar.number, project.bar_count)
        next_info = _parse_chord(chord_parser, next_chord.symbol)
        last_pc = note_to_midi(notes[-1].pitch) % 12
        checked += 1
        if last_pc == next_info.root_pc or _pc_distance(last_pc, next_info.root_pc) <= 2:
            supported += 1
    if checked == 0:
        return None
    return supported / checked


def _contour(pitches: list[int]) -> list[int]:
    return [
        1 if current > previous else -1 if current < previous else 0
        for previous, current in zip(pitches, pitches[1:], strict=False)
    ]


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


def generate_bass_track(
    spec: GenerationSpec,
    project: ArrangementProject,
    *,
    context: Any,
) -> Track:
    _ = spec
    return BassEngine().generate(context)
