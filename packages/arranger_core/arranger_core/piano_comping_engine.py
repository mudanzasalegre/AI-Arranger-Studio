from __future__ import annotations

from statistics import mean
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.music_theory import midi_to_note, note_to_midi, pitch_class, pitch_class_name
from arranger_core.retrieval import retrieval_trace, retrieve_pattern
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    NoteEvent,
    RestEvent,
    Track,
    meter_to_quarter_beats,
)
from arranger_core.song_planner import GrooveMap, SectionPlan, SongPlan

PIANO_COMPING_ENGINE_VERSION = "0.1.0"
PianoCompingMode = Literal["rule_based", "retrieval", "ai_variation"]
PianoVoicingStyle = Literal["shell", "rootless", "quartal", "spread"]
PianoCompingSource = Literal["rule_based", "retrieval", "ai_variation", "fallback_rule_based"]

PRACTICAL_LOW = note_to_midi("C3")
PRACTICAL_HIGH = note_to_midi("C6")
COMFORT_LOW = note_to_midi("E3")
COMFORT_HIGH = note_to_midi("G5")


class PianoCompingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PianoChordInfo(PianoCompingModel):
    symbol: str
    root: str
    root_pc: int
    quality: str
    chord_tone_pcs: tuple[int, ...]
    guide_tone_pcs: tuple[int, ...]
    tension_pcs: tuple[int, ...]
    alteration_pcs: tuple[int, ...]
    prefer_sharps: bool


class PianoCompingLedgerEntry(PianoCompingModel):
    bar_number: int
    chord_symbols: list[str]
    mode: PianoCompingMode
    source: PianoCompingSource
    voicing_style: PianoVoicingStyle
    starts: list[float]
    voicing_sizes: list[int]
    density: float
    section_id: str | None = None
    source_pattern_id: str | None = None
    max_voice_leading_semitones: int = 0
    fill: bool = False


class PianoCompingLedger(PianoCompingModel):
    schema_version: str = PIANO_COMPING_ENGINE_VERSION
    entries: list[PianoCompingLedgerEntry] = Field(default_factory=list)

    def add(self, entry: PianoCompingLedgerEntry) -> None:
        self.entries.append(entry)


class PianoCompingValidationReport(PianoCompingModel):
    status: Literal["pass", "fail"]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class PianoCompingAiBackend(Protocol):
    def generate_piano_variation(
        self,
        *,
        project: ArrangementProject,
        base_track: Track,
        context: Any,
    ) -> Track:
        ...


class PianoCompingEngine:
    def __init__(
        self,
        *,
        chord_parser: ChordParser | None = None,
        ai_backend: PianoCompingAiBackend | None = None,
    ) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()
        self.ai_backend = ai_backend

    def generate(self, context: Any) -> Track:
        mode = _piano_mode(context)
        style = _voicing_style(context)
        pattern = _select_pattern(context) if mode == "retrieval" else None
        selected_mode = mode
        source: PianoCompingSource = "rule_based"
        fallback_reason: str | None = None

        base_track = self._build_track(
            context,
            mode="rule_based",
            source="rule_based",
            style=style,
            pattern=None,
        )
        track = base_track

        if mode == "retrieval":
            if pattern is None:
                selected_mode = "rule_based"
                fallback_reason = "retrieval_pattern_unavailable"
            else:
                retrieved = self._build_track(
                    context,
                    mode="retrieval",
                    source="retrieval",
                    style=style,
                    pattern=pattern,
                )
                report = self.validate_track(context.project, retrieved)
                if report.status == "pass":
                    track = retrieved
                    source = "retrieval"
                else:
                    selected_mode = "rule_based"
                    fallback_reason = "retrieval_validation_failed"
        elif mode == "ai_variation":
            ai_result = self._ai_track(context, base_track)
            if ai_result["track"] is not None:
                track = ai_result["track"]
                source = "ai_variation"
            else:
                selected_mode = "rule_based"
                fallback_reason = str(ai_result["fallback_reason"])

        validation = self.validate_track(context.project, track)
        if validation.status == "fail" and track is not base_track:
            track = base_track
            selected_mode = "rule_based"
            source = "fallback_rule_based"
            fallback_reason = "piano_comping_validation_failed"
            validation = self.validate_track(context.project, track)

        ledger = _build_ledger(
            context,
            track,
            mode=selected_mode,
            source=source,
            style=style,
            pattern=pattern if source == "retrieval" else None,
        )
        return track.model_copy(
            update={
                "metadata": {
                    **track.metadata,
                    "generator": "PianoCompingEngine",
                    "piano_comping_engine_version": PIANO_COMPING_ENGINE_VERSION,
                    "piano_comping_mode": selected_mode,
                    "piano_comping_source": source,
                    "voicing": style,
                    "voicing_style": style,
                    "learned_pattern_id": (
                        pattern.get("id") if source == "retrieval" and pattern else None
                    ),
                    "fallback_reason": fallback_reason,
                    "retrieval_trace": retrieval_trace(pattern),
                    "piano_comping_validation": validation.model_dump(mode="json"),
                    "piano_comping_ledger": ledger.model_dump(mode="json"),
                }
            }
        )

    def validate_track(
        self,
        project: ArrangementProject,
        track: Track,
    ) -> PianoCompingValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        grouped_voicings = _group_voicings(track)
        chords_by_bar = _chords_by_bar(project.chord_grid)
        all_notes = [
            event
            for _bar_number, _start, events in grouped_voicings
            for event in events
        ]

        if not all_notes:
            errors.append(
                _issue(
                    "no_piano_notes",
                    "Piano comping track has no notes",
                    track_id=track.id,
                )
            )

        voicing_sizes: list[int] = []
        spans: list[int] = []
        low_register_notes = 0
        root_doublings = 0
        too_dense_bars: list[int] = []
        voice_leading_distances: list[int] = []
        previous: list[int] | None = None

        for bar_number, start, events in grouped_voicings:
            midi_notes = sorted(note_to_midi(event.pitch) for event in events)
            if not midi_notes:
                continue
            voicing_sizes.append(len(midi_notes))
            span = midi_notes[-1] - midi_notes[0]
            spans.append(span)
            if len(midi_notes) > 5:
                errors.append(
                    _issue(
                        "piano_polyphony",
                        f"Piano voicing has {len(midi_notes)} notes",
                        track_id=track.id,
                        bar_number=bar_number,
                        details={"beat": start + 1},
                    )
                )
            if span > 28:
                errors.append(
                    _issue(
                        "piano_voicing_span",
                        f"Piano voicing spans {span} semitones",
                        track_id=track.id,
                        bar_number=bar_number,
                        details={"beat": start + 1, "span_semitones": span},
                    )
                )
            for midi_note in midi_notes:
                if midi_note < PRACTICAL_LOW or midi_note > PRACTICAL_HIGH:
                    errors.append(
                        _issue(
                            "piano_register",
                            "Piano voicing outside practical comping register",
                            track_id=track.id,
                            bar_number=bar_number,
                            details={"midi_note": midi_note},
                        )
                    )
                if midi_note < COMFORT_LOW:
                    low_register_notes += 1
            root_pc = events[0].annotations.get("root_pc")
            if root_pc is None:
                active_chord = _active_chord(chords_by_bar, bar_number, start)
                root_pc = _parse_chord(self.chord_parser, active_chord.symbol).root_pc
            if root_pc is not None and any(midi_note % 12 == root_pc for midi_note in midi_notes):
                root_doublings += 1
                errors.append(
                    _issue(
                        "piano_root_duplication",
                        "Piano voicing duplicates the root and can collide with bass",
                        track_id=track.id,
                        bar_number=bar_number,
                        details={"beat": start + 1},
                    )
                )
            if previous is not None:
                distance = _voice_leading_distance(previous, midi_notes)
                voice_leading_distances.append(distance)
                if distance > 18:
                    errors.append(
                        _issue(
                            "piano_voice_leading",
                            f"Piano voicing moves {distance} semitones",
                            track_id=track.id,
                            bar_number=bar_number,
                            details={"beat": start + 1, "distance": distance},
                        )
                    )
            previous = midi_notes

        notes_by_bar: dict[int, int] = {}
        for bar in track.bars:
            count = sum(1 for event in bar.events if isinstance(event, NoteEvent))
            notes_by_bar[bar.number] = count
            if count > 12:
                too_dense_bars.append(bar.number)
        if too_dense_bars:
            errors.append(
                _issue(
                    "piano_density",
                    "Piano comping is too dense",
                    track_id=track.id,
                    details={"bars": too_dense_bars[:12]},
                )
            )
        if low_register_notes:
            warnings.append(
                _issue(
                    "piano_low_register",
                    "Piano uses low-register notes that may clutter the bass",
                    track_id=track.id,
                    details={"note_count": low_register_notes},
                )
            )

        metrics = {
            "note_count": len(all_notes),
            "bar_count": len(track.bars),
            "avg_notes_per_bar": (
                round(sum(notes_by_bar.values()) / len(notes_by_bar), 3)
                if notes_by_bar
                else 0.0
            ),
            "max_notes_per_bar": max(notes_by_bar.values(), default=0),
            "avg_voicing_size": round(mean(voicing_sizes), 3) if voicing_sizes else 0.0,
            "max_voicing_size": max(voicing_sizes, default=0),
            "max_span_semitones": max(spans, default=0),
            "max_voice_leading_semitones": max(voice_leading_distances, default=0),
            "avg_voice_leading_semitones": (
                round(mean(voice_leading_distances), 3) if voice_leading_distances else 0.0
            ),
            "root_doublings": root_doublings,
            "low_register_notes": low_register_notes,
        }
        return PianoCompingValidationReport(
            status="fail" if errors else "pass",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def _build_track(
        self,
        context: Any,
        *,
        mode: PianoCompingMode,
        source: PianoCompingSource,
        style: PianoVoicingStyle,
        pattern: dict[str, Any] | None,
    ) -> Track:
        chords_by_bar = _chords_by_bar(context.project.chord_grid)
        song_plan = _song_plan(context)
        groove_map = _groove_map(context)
        bars: list[Bar] = []
        previous_voicing: list[int] | None = None

        for bar_number in range(1, context.project.bar_count + 1):
            bar_duration = _bar_duration(context.project, bar_number)
            section = _section_for_bar(song_plan, bar_number)
            density = _comping_density(context, section)
            starts = _comping_starts(
                bar_number,
                bar_duration=bar_duration,
                context=context,
                density=density,
                groove_map=groove_map,
                section=section,
            )
            fill = _is_phrase_fill(song_plan, groove_map, bar_number)
            note_events: list[NoteEvent] = []
            for index, start in enumerate(starts):
                chord = _active_chord(chords_by_bar, bar_number, start)
                chord_info = _parse_chord(self.chord_parser, chord.symbol)
                voicing = _voicing_from_pattern(
                    chord_info,
                    pattern,
                    previous_voicing=previous_voicing,
                    style=style,
                ) if pattern else []
                if not voicing:
                    voicing = _voicing_for_chord(
                        chord_info,
                        style=style,
                        previous_voicing=previous_voicing,
                        density=density,
                    )
                duration = _event_duration(starts, index, bar_duration, style=style, fill=fill)
                previous_voicing = voicing
                for note in voicing:
                    pitch = midi_to_note(note, prefer_sharps=chord_info.prefer_sharps)
                    note_events.append(
                        NoteEvent(
                            pitch=pitch,
                            start=start,
                            duration=duration,
                            velocity=_velocity_for_density(density, fill=fill),
                            articulations=_articulations(style, fill=fill),
                            annotations={
                                "voicing": style,
                                "voicing_style": style,
                                "source_chord": chord.symbol,
                                "root_pc": chord_info.root_pc,
                                "piano_comping_mode": mode,
                                "piano_comping_source": source,
                                "comping_density": round(density, 3),
                                "register": _register_label(voicing),
                                "piano_fill": fill,
                                "learned_pattern_id": pattern.get("id") if pattern else None,
                            },
                        )
                    )
            bars.append(
                Bar(
                    number=bar_number,
                    events=_with_rests(note_events, bar_duration, voice=1),
                    metadata={
                        "comping_rhythm": starts,
                        "density": round(density, 3),
                        "voicing_style": style,
                        "piano_comping_mode": mode,
                        "piano_comping_source": source,
                        "piano_fill": fill,
                        "section_id": section.id if section else None,
                        "learned_pattern_id": pattern.get("id") if pattern else None,
                    },
                )
            )

        return Track(
            id="piano",
            instrument="piano",
            role="comping",
            bars=bars,
            metadata={
                "generator": "PianoCompingEngine",
                "voicing": style,
                "voicing_style": style,
                "piano_comping_mode": mode,
                "piano_comping_source": source,
                "learned_pattern_id": pattern.get("id") if pattern else None,
            },
        )

    def _ai_track(self, context: Any, base_track: Track) -> dict[str, Any]:
        if self.ai_backend is None:
            return {"track": None, "fallback_reason": "ai_backend_unavailable"}
        try:
            track = self.ai_backend.generate_piano_variation(
                project=context.project,
                base_track=base_track,
                context=context,
            )
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            return {"track": None, "fallback_reason": f"ai_backend_error:{exc}"}
        report = self.validate_track(context.project, track)
        if report.status != "pass":
            return {"track": None, "fallback_reason": "ai_validation_failed"}
        return {"track": track, "fallback_reason": None}


def _voicing_for_chord(
    chord_info: PianoChordInfo,
    *,
    style: PianoVoicingStyle,
    previous_voicing: list[int] | None,
    density: float,
) -> list[int]:
    pcs = _voicing_pcs(chord_info, style=style, density=density)
    preferred_size = 2 if density < 0.48 else 3 if density < 0.78 else 4
    if style == "shell":
        preferred_size = min(preferred_size, 2)
    if style == "spread":
        preferred_size = min(max(preferred_size, 3), 4)
    pcs = pcs[:preferred_size]
    return _voice_lead_pcs(
        pcs,
        chord_info=chord_info,
        previous_voicing=previous_voicing,
        style=style,
    )


def _voicing_pcs(
    chord_info: PianoChordInfo,
    *,
    style: PianoVoicingStyle,
    density: float,
) -> list[int]:
    guide = [pc for pc in chord_info.guide_tone_pcs if pc != chord_info.root_pc]
    tensions = [
        pc
        for pc in (*chord_info.tension_pcs, *_default_tension_pcs(chord_info))
        if pc != chord_info.root_pc
    ]
    tones = [pc for pc in chord_info.chord_tone_pcs if pc != chord_info.root_pc]

    if style == "shell":
        return list(dict.fromkeys([*guide, *tones]))[:2]
    if style == "quartal":
        start_pc = (chord_info.root_pc + (5 if density < 0.7 else 2)) % 12
        quartal = [(start_pc + interval) % 12 for interval in (0, 5, 10, 15)]
        quartal = [pc for pc in quartal if pc != chord_info.root_pc]
        return list(dict.fromkeys([*quartal, *guide, *tensions]))
    if style == "spread":
        return list(dict.fromkeys([*guide[:1], *tensions[:2], *guide[1:], *tones]))[:4]
    return list(dict.fromkeys([*guide, *tensions, *tones]))[:4]


def _voicing_from_pattern(
    chord_info: PianoChordInfo,
    pattern: dict[str, Any] | None,
    *,
    previous_voicing: list[int] | None,
    style: PianoVoicingStyle,
) -> list[int]:
    payload = _learned_payload(pattern)
    raw_relative_notes = payload.get("relative_notes")
    if not isinstance(raw_relative_notes, list):
        return []
    intervals: list[int] = []
    for value in raw_relative_notes[:5]:
        try:
            intervals.append(int(value))
        except (TypeError, ValueError):
            continue
    if len(intervals) < 2:
        return []

    anchor_pc = (chord_info.guide_tone_pcs or chord_info.chord_tone_pcs)[0]
    anchor = _target_center(previous_voicing, style=style)
    _, base_midi = _nearest_note_in_range(
        anchor_pc,
        low_midi=COMFORT_LOW,
        high_midi=COMFORT_HIGH,
        anchor_midi=anchor,
        prefer_sharps=chord_info.prefer_sharps,
    )
    midi_notes: list[int] = []
    for interval in intervals:
        midi_note = base_midi + interval
        while midi_note < COMFORT_LOW:
            midi_note += 12
        while midi_note > COMFORT_HIGH:
            midi_note -= 12
        if midi_note % 12 == chord_info.root_pc or midi_note in midi_notes:
            continue
        midi_notes.append(midi_note)
    if len(midi_notes) < 2:
        return []
    midi_notes = sorted(midi_notes)[:4]
    if midi_notes[-1] - midi_notes[0] > 28:
        return []
    return _smooth_voicing(midi_notes, previous_voicing, style=style)


def _voice_lead_pcs(
    pcs: list[int],
    *,
    chord_info: PianoChordInfo,
    previous_voicing: list[int] | None,
    style: PianoVoicingStyle,
) -> list[int]:
    anchor = _target_center(previous_voicing, style=style)
    notes: list[int] = []
    offsets = _style_offsets(style, len(pcs))
    for pc, offset in zip(pcs, offsets, strict=False):
        _, midi_note = _nearest_note_in_range(
            pc,
            low_midi=COMFORT_LOW,
            high_midi=COMFORT_HIGH,
            anchor_midi=anchor + offset,
            prefer_sharps=chord_info.prefer_sharps,
        )
        while midi_note in notes and midi_note + 12 <= COMFORT_HIGH:
            midi_note += 12
        if midi_note % 12 == chord_info.root_pc:
            continue
        notes.append(midi_note)
    if len(notes) < 2:
        fallback = [
            pc
            for pc in (
                *chord_info.guide_tone_pcs,
                *chord_info.tension_pcs,
                *chord_info.chord_tone_pcs,
            )
            if pc != chord_info.root_pc
        ]
        return _voice_lead_pcs(
            list(dict.fromkeys(fallback))[:3],
            chord_info=chord_info,
            previous_voicing=previous_voicing,
            style="rootless",
        )
    return _smooth_voicing(sorted(notes), previous_voicing, style=style)


def _smooth_voicing(
    notes: list[int],
    previous_voicing: list[int] | None,
    *,
    style: PianoVoicingStyle,
) -> list[int]:
    low = PRACTICAL_LOW if style == "spread" else COMFORT_LOW
    high = PRACTICAL_HIGH if style == "spread" else COMFORT_HIGH
    candidates: list[list[int]] = []
    for shift in (-12, 0, 12):
        shifted = sorted(note + shift for note in notes)
        while shifted and shifted[0] < low:
            shifted = sorted(note + 12 for note in shifted)
        while shifted and shifted[-1] > high:
            shifted = sorted(note - 12 for note in shifted)
        if shifted and low <= shifted[0] and shifted[-1] <= high:
            candidates.append(shifted)
    if not candidates:
        return sorted(notes)
    if previous_voicing is None:
        return min(
            candidates,
            key=lambda item: abs(_center(item) - _target_center(None, style=style)),
        )
    return min(candidates, key=lambda item: _voice_leading_distance(previous_voicing, item))


def _comping_starts(
    bar_number: int,
    *,
    bar_duration: float,
    context: Any,
    density: float,
    groove_map: GrooveMap | None,
    section: SectionPlan | None,
) -> list[float]:
    if bar_duration == 3.0:
        cells = ([0.5, 2.0], [1.0, 2.0], [0.0, 1.5])
    elif context.spec.style == "bossa_nova":
        cells = ([0.5, 1.5, 3.0], [0.0, 2.0, 3.0], [0.5, 2.5])
    elif context.spec.style == "funk_jazz":
        cells = ([0.0, 1.5, 2.5], [0.5, 1.5, 3.0])
    elif context.spec.style == "jazz_ballad":
        cells = ([0.0, 2.0], [1.5], [0.0, 3.0])
    else:
        safe = tuple(groove_map.comping_safe_beats) if groove_map else (0.5, 1.5, 2.5, 3.5)
        cells = (safe[0::2] or safe, safe[1::2] or safe, safe[:3], safe[-3:])
    starts = list(cells[(bar_number - 1) % len(cells)])
    limit = 1 if density < 0.46 else 2 if density < 0.68 else 3
    if section and section.function in {"bridge", "turnaround"}:
        limit = min(3, limit + 1)
    if _is_phrase_fill(_song_plan(context), groove_map, bar_number):
        limit = min(len(starts), max(2, limit))
    return [start for start in starts[:limit] if start < bar_duration]


def _event_duration(
    starts: list[float],
    index: int,
    bar_duration: float,
    *,
    style: PianoVoicingStyle,
    fill: bool,
) -> float:
    max_duration = 1.25 if style == "spread" else 0.75
    if fill:
        max_duration = 0.5
    next_start = starts[index + 1] if index + 1 < len(starts) else bar_duration
    return max(0.25, min(max_duration, next_start - starts[index]))


def _piano_mode(context: Any) -> PianoCompingMode:
    raw = context.spec.constraints.get("piano_comping_mode") or context.spec.constraints.get(
        "piano_mode"
    )
    if raw in {"rule_based", "retrieval", "ai_variation"}:
        return raw
    if (
        context.spec.constraints.get("piano_retrieval", True) is not False
        and _select_pattern(context) is not None
    ):
        return "retrieval"
    return "rule_based"


def _voicing_style(context: Any) -> PianoVoicingStyle:
    raw = context.spec.constraints.get("piano_voicing") or context.spec.constraints.get("voicing")
    if raw in {"shell", "rootless", "quartal", "spread"}:
        return raw
    if context.spec.style == "modal_jazz":
        return "quartal"
    if context.spec.style == "jazz_ballad":
        return "spread"
    if context.spec.density == "low":
        return "shell"
    return "rootless"


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


def _section_for_bar(song_plan: SongPlan | None, bar_number: int) -> SectionPlan | None:
    if song_plan is None:
        return None
    for section in song_plan.sections:
        if section.start_bar <= bar_number <= section.end_bar:
            return section
    return None


def _comping_density(context: Any, section: SectionPlan | None) -> float:
    if section is not None and "comping" in section.role_densities:
        return max(0.0, min(1.0, float(section.role_densities["comping"])))
    base = {"low": 0.42, "medium": 0.62, "high": 0.78}.get(str(context.spec.density), 0.62)
    if context.spec.style == "jazz_ballad":
        base *= 0.72
    return max(0.0, min(1.0, base))


def _is_phrase_fill(
    song_plan: SongPlan | None,
    groove_map: GrooveMap | None,
    bar_number: int,
) -> bool:
    if groove_map is not None and bar_number in set(groove_map.setup_bars):
        return True
    if song_plan is None:
        return False
    return any(phrase.cadence_bar == bar_number + 1 for phrase in song_plan.phrases)


def _select_pattern(context: Any) -> dict[str, Any] | None:
    return retrieve_pattern(
        context,
        category="piano_voicings",
        role="comping",
        instrument="piano",
        density=context.spec.density,
    )


def _build_ledger(
    context: Any,
    track: Track,
    *,
    mode: PianoCompingMode,
    source: PianoCompingSource,
    style: PianoVoicingStyle,
    pattern: dict[str, Any] | None,
) -> PianoCompingLedger:
    ledger = PianoCompingLedger()
    chords_by_bar = _chords_by_bar(context.project.chord_grid)
    song_plan = _song_plan(context)
    previous: list[int] | None = None
    for bar in track.bars:
        groups = [
            events
            for _bar_number, _start, events in _group_voicings(track, bar_numbers={bar.number})
        ]
        starts = []
        sizes = []
        distances = []
        chord_symbols = []
        for events in groups:
            midi_notes = sorted(note_to_midi(event.pitch) for event in events)
            starts.append(events[0].start)
            sizes.append(len(events))
            chord_symbols.append(str(events[0].annotations.get("source_chord", "")))
            if previous is not None:
                distances.append(_voice_leading_distance(previous, midi_notes))
            previous = midi_notes
        section = _section_for_bar(song_plan, bar.number)
        ledger.add(
            PianoCompingLedgerEntry(
                bar_number=bar.number,
                chord_symbols=chord_symbols
                or [_active_chord(chords_by_bar, bar.number, 0.0).symbol],
                mode=mode,
                source=source,
                voicing_style=style,
                starts=starts,
                voicing_sizes=sizes,
                density=float(bar.metadata.get("density", 0.0) or 0.0),
                section_id=section.id if section else None,
                source_pattern_id=pattern.get("id") if pattern else None,
                max_voice_leading_semitones=max(distances, default=0),
                fill=bool(bar.metadata.get("piano_fill")),
            )
        )
    return ledger


def _group_voicings(
    track: Track,
    *,
    bar_numbers: set[int] | None = None,
) -> list[tuple[int, float, list[NoteEvent]]]:
    groups: list[tuple[int, float, list[NoteEvent]]] = []
    for bar in track.bars:
        if bar_numbers is not None and bar.number not in bar_numbers:
            continue
        by_start: dict[tuple[float, float, int], list[NoteEvent]] = {}
        for event in bar.events:
            if isinstance(event, NoteEvent):
                by_start.setdefault((event.start, event.duration, event.voice), []).append(event)
        for (start, _duration, _voice), events in sorted(by_start.items()):
            groups.append((bar.number, start, events))
    return groups


def _parse_chord(parser: ChordParser, symbol: str) -> PianoChordInfo:
    try:
        parsed = parser.parse(symbol)
    except ValueError:
        root = _fallback_root(symbol)
        root_pc = pitch_class(root)
        return PianoChordInfo(
            symbol=symbol,
            root=root,
            root_pc=root_pc,
            quality="major_triad",
            chord_tone_pcs=(root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12),
            guide_tone_pcs=((root_pc + 4) % 12, (root_pc + 10) % 12),
            tension_pcs=(),
            alteration_pcs=(),
            prefer_sharps="#" in root and "b" not in root,
        )
    return PianoChordInfo(
        symbol=symbol,
        root=parsed.root,
        root_pc=parsed.root_pc,
        quality=parsed.quality,
        chord_tone_pcs=tuple(parsed.chord_tone_pcs) or (parsed.root_pc,),
        guide_tone_pcs=_guide_tones(parsed),
        tension_pcs=tuple(parsed.tension_pcs),
        alteration_pcs=tuple(parsed.alteration_pcs),
        prefer_sharps="#" in parsed.root and "b" not in parsed.root,
    )


def _guide_tones(parsed: ParsedChord) -> tuple[int, ...]:
    intervals = [
        interval
        for interval in parsed.chord_tone_intervals
        if interval % 12 in {3, 4, 10, 11}
    ]
    return tuple((parsed.root_pc + interval) % 12 for interval in intervals)


def _default_tension_pcs(chord_info: PianoChordInfo) -> tuple[int, ...]:
    if chord_info.tension_pcs:
        return chord_info.tension_pcs
    if chord_info.quality == "dominant":
        intervals = (14, 21)
    elif chord_info.quality in {"minor_triad", "half_diminished"}:
        intervals = (14, 17)
    elif chord_info.quality == "major_triad":
        intervals = (14, 21)
    else:
        intervals = (14,)
    return tuple((chord_info.root_pc + interval) % 12 for interval in intervals)


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


def _style_offsets(style: PianoVoicingStyle, size: int) -> list[int]:
    if style == "spread":
        return [-10, 0, 7, 14][:size]
    if style == "quartal":
        return [-5, 0, 5, 10][:size]
    return [-5, 0, 5, 10][:size]


def _target_center(
    previous_voicing: list[int] | None,
    *,
    style: PianoVoicingStyle,
) -> int:
    if previous_voicing:
        return round(_center(previous_voicing))
    return {
        "shell": note_to_midi("F3"),
        "rootless": note_to_midi("C4"),
        "quartal": note_to_midi("D4"),
        "spread": note_to_midi("G3"),
    }[style]


def _center(notes: list[int]) -> float:
    return sum(notes) / max(1, len(notes))


def _voice_leading_distance(previous: list[int], current: list[int]) -> int:
    pairs = zip(sorted(previous), sorted(current), strict=False)
    return max((abs(before - after) for before, after in pairs), default=0)


def _with_rests(
    note_events: list[NoteEvent],
    bar_duration: float,
    *,
    voice: int,
) -> list[NoteEvent | RestEvent]:
    intervals = sorted(
        {(event.start, event.start + event.duration) for event in note_events},
        key=lambda item: item[0],
    )
    rests: list[RestEvent] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            rests.append(RestEvent(start=cursor, duration=start - cursor, voice=voice))
        cursor = max(cursor, min(end, bar_duration))
    if cursor < bar_duration:
        rests.append(RestEvent(start=cursor, duration=bar_duration - cursor, voice=voice))
    return sorted([*note_events, *rests], key=lambda event: (event.start, event.duration))


def _velocity_for_density(density: float, *, fill: bool) -> int:
    base = round(56 + density * 18)
    return min(92, base + 8) if fill else base


def _articulations(style: PianoVoicingStyle, *, fill: bool) -> list[str]:
    if fill:
        return ["accent"]
    if style == "spread":
        return ["tenuto"]
    if style == "quartal":
        return ["staccato"]
    return ["tenuto"]


def _register_label(notes: list[int]) -> str:
    center = _center(notes)
    if center < note_to_midi("G3"):
        return "low_mid"
    if center > note_to_midi("C5"):
        return "mid_high"
    return "mid"


def _fallback_root(chord_symbol: str) -> str:
    for length in (2, 1):
        root = chord_symbol[:length]
        try:
            pitch_class(root)
            return root
        except ValueError:
            continue
    return pitch_class_name(0)


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


def generate_piano_comping_track(
    spec: GenerationSpec,
    project: ArrangementProject,
    *,
    context: Any,
) -> Track:
    _ = spec, project
    return PianoCompingEngine().generate(context)
