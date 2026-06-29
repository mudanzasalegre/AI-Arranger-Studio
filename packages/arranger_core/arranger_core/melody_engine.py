from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.catalogs import InstrumentCatalog
from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.lead_sheet import LeadSheetGenerator, MelodyRange
from arranger_core.music_theory import midi_to_note, note_to_midi
from arranger_core.retrieval import retrieval_trace, retrieve_patterns
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
from arranger_core.song_planner import PhrasePlan, SongPlan

MELODY_ENGINE_VERSION = "0.1.0"
MelodyEngineMode = Literal["rule_based", "retrieval", "ai_infill"]
MotifSource = Literal["rule_based", "retrieval", "ai_infill", "fallback_rule_based"]


class MelodyEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MotifLedgerEntry(MelodyEngineModel):
    phrase_id: str
    bars: list[int]
    mode: MelodyEngineMode
    source: MotifSource
    accepted: bool = True
    motif_id: str = "main_motif"
    variation: str = "repeat"
    cadence_bar: int | None = None
    instrument: str
    rhythm: list[float] = Field(default_factory=list)
    contour: list[int] = Field(default_factory=list)
    source_pattern_id: str | None = None
    rejection_reason: str | None = None


class MotifLedger(MelodyEngineModel):
    schema_version: str = MELODY_ENGINE_VERSION
    entries: list[MotifLedgerEntry] = Field(default_factory=list)

    def add(self, entry: MotifLedgerEntry) -> None:
        self.entries.append(entry)


class MelodyValidationReport(MelodyEngineModel):
    status: Literal["pass", "fail"]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MelodyAiInfillBackend(Protocol):
    def generate_melody_infill(
        self,
        *,
        project: ArrangementProject,
        base_track: Track,
        instrument_id: str,
        target_bars: list[int],
        context: Any,
    ) -> Track:
        ...


class MelodyEngine:
    def __init__(
        self,
        *,
        chord_parser: ChordParser | None = None,
        instrument_catalog: InstrumentCatalog | None = None,
        ai_backend: MelodyAiInfillBackend | None = None,
    ) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()
        self.lead_sheet_generator = LeadSheetGenerator(
            chord_parser=self.chord_parser,
            instrument_catalog=self.instrument_catalog,
        )
        self.ai_backend = ai_backend

    def generate_for_instrument(
        self,
        context: Any,
        instrument_id: str,
    ) -> Track:
        mode = _melody_mode(context)
        base_track = self._rule_based_track(context, instrument_id)
        selected_mode = mode
        fallback_reason: str | None = None
        track = base_track

        if mode == "retrieval":
            retrieved = self._retrieval_track(context, base_track, instrument_id)
            report = self.validate_track(context.project, retrieved, instrument_id=instrument_id)
            if report.status == "pass":
                track = retrieved
            else:
                selected_mode = "rule_based"
                fallback_reason = "retrieval_validation_failed"
        elif mode == "ai_infill":
            ai_result = self._ai_infill_track(context, base_track, instrument_id)
            if ai_result["track"] is not None:
                track = ai_result["track"]
            else:
                selected_mode = "rule_based"
                fallback_reason = str(ai_result.get("fallback_reason") or "ai_infill_unavailable")

        validation = self.validate_track(context.project, track, instrument_id=instrument_id)
        if validation.status == "fail" and track is not base_track:
            track = base_track
            selected_mode = "rule_based"
            fallback_reason = "melody_validation_failed"
            validation = self.validate_track(context.project, track, instrument_id=instrument_id)

        ledger = build_motif_ledger(
            context,
            track,
            instrument_id=instrument_id,
            mode=selected_mode,
            fallback_reason=fallback_reason,
        )
        return _finalize_track(
            track,
            context=context,
            instrument_id=instrument_id,
            mode=selected_mode,
            validation=validation,
            ledger=ledger,
            fallback_reason=fallback_reason,
        )

    def validate_track(
        self,
        project: ArrangementProject,
        track: Track,
        *,
        instrument_id: str,
    ) -> MelodyValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        notes = list(_track_notes(track))
        rests = list(_track_rests(track))
        try:
            instrument = self.instrument_catalog.get(instrument_id)
        except KeyError:
            instrument = None

        if not notes:
            errors.append(_issue("no_melody_notes", "Melody track has no notes"))

        range_low = range_high = None
        if instrument is not None:
            range_low = note_to_midi(instrument.sounding_range[0])
            range_high = note_to_midi(instrument.sounding_range[1])
            for bar in track.bars:
                for event in bar.events:
                    if not isinstance(event, NoteEvent):
                        continue
                    midi_note = note_to_midi(event.pitch)
                    if midi_note < range_low or midi_note > range_high:
                        errors.append(
                            _issue(
                                "melody_range",
                                f"{event.pitch} outside {instrument_id} range",
                                track_id=track.id,
                                bar_number=bar.number,
                            )
                        )

        harmony_score = _harmony_score(
            project,
            track,
            chord_parser=self.chord_parser,
        )
        if harmony_score is not None and harmony_score < 0.52:
            errors.append(
                _issue(
                    "melody_harmony",
                    f"Melody harmony support is too low ({harmony_score:.2f})",
                    track_id=track.id,
                )
            )

        breath_rests = [rest for rest in rests if rest.duration >= 0.5]
        if instrument is not None and instrument.breath_required and not breath_rests:
            errors.append(_issue("melody_breathing", "Breath instrument melody has no rests"))

        phrase_lengths = _phrase_lengths(track)
        if any(length not in {2, 4} for length in phrase_lengths):
            warnings.append(
                _issue(
                    "melody_phrase_length",
                    "Melody phrases should resolve in 2 or 4 bar units",
                )
            )

        duration_issues = ArrangementProject(
            project_id=f"{project.project_id}-melody-validation",
            generation_spec=project.generation_spec,
            tempo_map=project.tempo_map,
            key_map=project.key_map,
            meter_map=project.meter_map,
            form=project.form,
            chord_grid=project.chord_grid,
            tracks=[track],
        ).validate_bar_durations()
        for issue in duration_issues:
            errors.append(
                _issue(
                    "melody_duration",
                    issue.message,
                    track_id=track.id,
                    bar_number=issue.bar_number,
                )
            )

        metrics = {
            "note_count": len(notes),
            "breath_rest_count": len(breath_rests),
            "harmony_score": harmony_score,
            "phrase_lengths": phrase_lengths,
            "range": {"low_midi": range_low, "high_midi": range_high},
        }
        return MelodyValidationReport(
            status="fail" if errors else "pass",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def _rule_based_track(self, context: Any, instrument_id: str) -> Track:
        constraints = {**context.spec.constraints, "lead_instrument": instrument_id}
        lead_project = self.lead_sheet_generator.generate(
            context.spec.model_copy(update={"constraints": constraints}),
            project_id=context.project.project_id,
        )
        return _retarget_track(
            lead_project.tracks[0],
            instrument_id=instrument_id,
            catalog=context.instrument_catalog,
        )

    def _retrieval_track(self, context: Any, base_track: Track, instrument_id: str) -> Track:
        patterns = _melodic_patterns(context)
        if not patterns:
            return base_track
        melody_range = _melody_range(context.spec, instrument_id, context.instrument_catalog)
        chords_by_bar = _chords_by_bar(context.project.chord_grid)
        bars_by_number = {bar.number: bar.model_copy(deep=True) for bar in base_track.bars}
        phrase_plans = _phrase_plans(context, context.project)
        selected_pattern_ids: list[str] = []

        for phrase_index, phrase in enumerate(phrase_plans):
            pattern = patterns[phrase_index % len(patterns)]
            target_bar = phrase.start_bar
            if target_bar not in bars_by_number:
                continue
            replacement = _retrieval_bar(
                project=context.project,
                bar_number=target_bar,
                pattern=pattern,
                chords_by_bar=chords_by_bar,
                melody_range=melody_range,
                chord_parser=self.chord_parser,
            )
            if replacement is None:
                continue
            bars_by_number[target_bar] = replacement
            selected_pattern_ids.append(str(pattern.get("id", "")))

        if not selected_pattern_ids:
            return base_track
        return base_track.model_copy(
            update={
                "bars": [bars_by_number[number] for number in sorted(bars_by_number)],
                "metadata": {
                    **base_track.metadata,
                    "melody_retrieval": {
                        "selected_pattern_ids": selected_pattern_ids,
                        "strategy": "phrase_opening_motif_adaptation",
                    },
                },
            },
            deep=True,
        )

    def _ai_infill_track(
        self,
        context: Any,
        base_track: Track,
        instrument_id: str,
    ) -> dict[str, Any]:
        if self.ai_backend is None:
            return {"track": None, "fallback_reason": "ai_backend_not_configured"}
        target_bars = _ai_target_bars(context, context.project)
        try:
            generated = self.ai_backend.generate_melody_infill(
                project=context.project,
                base_track=base_track,
                instrument_id=instrument_id,
                target_bars=target_bars,
                context=context,
            )
        except Exception as exc:
            return {"track": None, "fallback_reason": f"ai_backend_error:{exc}"}
        merged = _merge_target_bars(base_track, generated, target_bars=target_bars)
        report = self.validate_track(context.project, merged, instrument_id=instrument_id)
        if report.status == "fail":
            return {
                "track": None,
                "fallback_reason": "ai_validation_failed",
                "validation": report.model_dump(mode="json"),
            }
        return {"track": merged, "fallback_reason": None, "target_bars": target_bars}


def build_motif_ledger(
    context: Any,
    track: Track,
    *,
    instrument_id: str,
    mode: MelodyEngineMode,
    fallback_reason: str | None = None,
) -> MotifLedger:
    ledger = MotifLedger()
    for phrase in _phrase_plans(context, context.project):
        phrase_bars = list(range(phrase.start_bar, phrase.end_bar + 1))
        notes = [
            event
            for bar in track.bars
            if bar.number in phrase_bars
            for event in bar.events
            if isinstance(event, NoteEvent)
        ]
        source_pattern_id = next(
            (
                str(event.annotations["learned_pattern_id"])
                for event in notes
                if event.annotations.get("learned_pattern_id")
            ),
            None,
        )
        ai_accepted = any(event.annotations.get("melody_ai_infill") for event in notes)
        source: MotifSource
        if fallback_reason:
            source = "fallback_rule_based"
        elif ai_accepted:
            source = "ai_infill"
        elif source_pattern_id:
            source = "retrieval"
        else:
            source = "rule_based"
        ledger.add(
            MotifLedgerEntry(
                phrase_id=phrase.id,
                bars=phrase_bars,
                mode=mode,
                source=source,
                accepted=True,
                motif_id=phrase.motif_id,
                variation=phrase.variation,
                cadence_bar=phrase.cadence_bar,
                instrument=instrument_id,
                rhythm=[round(event.duration, 3) for event in notes[:8]],
                contour=_contour(notes),
                source_pattern_id=source_pattern_id,
                rejection_reason=fallback_reason,
            )
        )
    return ledger


def _finalize_track(
    track: Track,
    *,
    context: Any,
    instrument_id: str,
    mode: MelodyEngineMode,
    validation: MelodyValidationReport,
    ledger: MotifLedger,
    fallback_reason: str | None,
) -> Track:
    phrase_lookup = {
        bar: entry
        for entry in ledger.entries
        for bar in entry.bars
    }
    bars: list[Bar] = []
    for bar in track.bars:
        ledger_entry = phrase_lookup.get(bar.number)
        events = []
        for event in bar.events:
            if isinstance(event, NoteEvent) and ledger_entry is not None:
                event = event.model_copy(
                    update={
                        "annotations": {
                            **event.annotations,
                            "melody_engine": "MelodyEngine2",
                            "melody_mode": mode,
                            "phrase_id": ledger_entry.phrase_id,
                            "motif_id": ledger_entry.motif_id,
                            "motif_variation": ledger_entry.variation,
                            "cadence_bar": ledger_entry.cadence_bar,
                        }
                    }
                )
            events.append(event)
        bars.append(
            bar.model_copy(
                update={
                    "events": events,
                    "metadata": {
                        **bar.metadata,
                        "melody_engine_phrase_id": ledger_entry.phrase_id
                        if ledger_entry
                        else None,
                    },
                }
            )
        )
    return track.model_copy(
        update={
            "bars": bars,
            "metadata": {
                **track.metadata,
                "generator": "MelodyGenerator",
                "source": "MelodyEngine2",
                "melody_engine_version": MELODY_ENGINE_VERSION,
                "melody_engine_mode": mode,
                "modes_available": ["rule_based", "retrieval", "ai_infill"],
                "head": True,
                "motif_ledger": ledger.model_dump(mode="json"),
                "melody_validation": validation.model_dump(mode="json"),
                "fallback_reason": fallback_reason,
                "song_plan_phrase_count": len(_phrase_plans(context, context.project)),
                "range": _track_range_metadata(track, instrument_id, context.instrument_catalog),
            },
        },
        deep=True,
    )


def _melody_mode(context: Any) -> MelodyEngineMode:
    raw_mode = (
        context.spec.constraints.get("melody_mode")
        or context.spec.constraints.get("melody_engine_mode")
    )
    if raw_mode in {"rule_based", "retrieval", "ai_infill"}:
        return raw_mode
    if _melodic_patterns(context) and context.spec.constraints.get("melody_retrieval", True):
        return "retrieval"
    return "rule_based"


def _melodic_patterns(context: Any) -> list[dict[str, Any]]:
    return [
        pattern
        for pattern in retrieve_patterns(
            context,
            category="melodic_motifs",
            role="melody",
            instrument=str(context.spec.constraints.get("lead_instrument", "lead_instrument")),
            density=context.spec.density,
            limit=8,
        )
        if isinstance(pattern.get("payload"), dict)
    ]


def _retrieval_bar(
    *,
    project: ArrangementProject,
    bar_number: int,
    pattern: dict[str, Any],
    chords_by_bar: dict[int, list[ChordSymbol]],
    melody_range: MelodyRange,
    chord_parser: ChordParser,
) -> Bar | None:
    payload = pattern.get("payload")
    if not isinstance(payload, dict):
        return None
    intervals = _int_list(payload.get("relative_degrees"))[:6]
    if not intervals:
        return None
    bar_duration = meter_to_quarter_beats(project.meter_at_bar(bar_number))
    if abs(bar_duration - 4.0) > 1e-6:
        return None
    rhythm = _float_list(payload.get("rhythm"))[: len(intervals)]
    starts = _retrieval_starts(len(intervals), bar_duration)
    chord = _active_chord(chords_by_bar, bar_number, 0.0)
    palette = _palette(chord.symbol, chord_parser)
    previous_midi = melody_range.midpoint
    notes: list[NoteEvent] = []
    for index, interval in enumerate(intervals):
        target_pc = _supported_pc((palette.root_pc + interval) % 12, palette)
        pitch, previous_midi = _nearest_note_in_range(
            target_pc,
            melody_range=melody_range,
            anchor_midi=previous_midi,
            prefer_sharps=_prefer_sharps(palette),
        )
        duration = min(0.75, rhythm[index] if index < len(rhythm) else 0.5)
        start = starts[index]
        if start + duration > bar_duration:
            duration = max(0.25, bar_duration - start)
        notes.append(
            NoteEvent(
                pitch=pitch,
                start=round(start, 3),
                duration=round(duration, 3),
                velocity=84 if index == 0 else 78,
                articulations=["accent"] if index == 0 else ["staccato"],
                annotations={
                    "melodic_role": "retrieved_motif",
                    "source_chord": chord.symbol,
                    "learned_pattern_id": pattern.get("id"),
                    "retrieval_transform": "transpose_to_active_chord",
                    "retrieval_interval": interval,
                    "retrieval_trace": retrieval_trace(pattern),
                },
            )
        )
    return Bar(
        number=bar_number,
        meter=project.meter_at_bar(bar_number),
        events=_fill_rests(notes, bar_duration),
        metadata={
            "melody_retrieval": True,
            "learned_pattern_id": pattern.get("id"),
            "retrieval_trace": retrieval_trace(pattern),
        },
    )


def _phrase_plans(context: Any, project: ArrangementProject) -> list[PhrasePlan]:
    song_plan = getattr(context, "song_plan", None)
    if isinstance(song_plan, SongPlan) and song_plan.phrases:
        return song_plan.phrases
    phrases: list[PhrasePlan] = []
    phrase_size = 4 if project.bar_count % 4 == 0 else 2
    current = 1
    index = 1
    while current <= project.bar_count:
        end_bar = min(project.bar_count, current + phrase_size - 1)
        phrases.append(
            PhrasePlan(
                id=f"phrase_{index:02d}",
                section_id="fallback_section",
                start_bar=current,
                end_bar=end_bar,
                function="complete_phrase" if current == 1 else "answer",
                motif_id="main_motif",
                variation="repeat" if index == 1 else "answer",
                energy=0.5,
                density=0.5,
                cadence_bar=end_bar,
                target_role="melody",
                breath_points=[bar for bar in range(current + 1, end_bar + 1, 2)],
            )
        )
        current = end_bar + 1
        index += 1
    return phrases


def _ai_target_bars(context: Any, project: ArrangementProject) -> list[int]:
    configured = context.spec.constraints.get("melody_ai_bars")
    if isinstance(configured, list):
        bars = sorted({int(value) for value in configured if int(value) >= 1})
        if bars:
            return [bar for bar in bars if bar <= project.bar_count]
    phrases = _phrase_plans(context, project)
    target = next((phrase for phrase in phrases if phrase.variation != "repeat"), phrases[0])
    return list(range(target.start_bar, target.end_bar + 1))


def _merge_target_bars(
    base_track: Track,
    generated_track: Track,
    *,
    target_bars: list[int],
) -> Track:
    generated_by_bar = {bar.number: bar for bar in generated_track.bars}
    merged_bars = []
    for bar in base_track.bars:
        if bar.number in target_bars and bar.number in generated_by_bar:
            replacement = generated_by_bar[bar.number].model_copy(deep=True)
            replacement.events = [
                _mark_ai_event(event)
                for event in replacement.events
            ]
            merged_bars.append(replacement)
        else:
            merged_bars.append(bar)
    return base_track.model_copy(
        update={
            "bars": merged_bars,
            "metadata": {
                **base_track.metadata,
                "ai_infill_target_bars": target_bars,
                "ai_infill_status": "accepted",
            },
        },
        deep=True,
    )


def _mark_ai_event(event: NoteEvent | RestEvent) -> NoteEvent | RestEvent:
    if not isinstance(event, NoteEvent):
        return event
    return event.model_copy(
        update={
            "annotations": {
                **event.annotations,
                "melody_ai_infill": True,
                "source": "model_artifact",
            }
        }
    )


def _retarget_track(track: Track, *, instrument_id: str, catalog: InstrumentCatalog) -> Track:
    track_id = instrument_id if instrument_id != "piano" else "piano_melody"
    return track.model_copy(
        update={
            "id": track_id,
            "instrument": instrument_id,
            "role": "melody",
            "name": _display_name(instrument_id, catalog),
        },
        deep=True,
    )


def _display_name(instrument_id: str, catalog: InstrumentCatalog) -> str:
    try:
        return catalog.get(instrument_id).display_name
    except KeyError:
        return instrument_id


def _melody_range(
    spec: GenerationSpec,
    instrument_id: str,
    catalog: InstrumentCatalog,
) -> MelodyRange:
    configured = (
        spec.constraints.get("melody_range")
        or spec.constraints.get("lead_sheet_range")
        or spec.constraints.get("lead_range")
    )
    if configured is not None:
        low, high = _parse_range_constraint(configured)
    else:
        low, high = catalog.get(instrument_id).comfortable_range
    low_midi = note_to_midi(low)
    high_midi = note_to_midi(high)
    return MelodyRange(low=low, high=high, low_midi=low_midi, high_midi=high_midi)


def _parse_range_constraint(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        low = value.get("low")
        high = value.get("high")
        if isinstance(low, str) and isinstance(high, str):
            return low, high
    if isinstance(value, list | tuple) and len(value) == 2:
        low, high = value
        if isinstance(low, str) and isinstance(high, str):
            return low, high
    if isinstance(value, str):
        for separator in ("-", "..", ":"):
            if separator in value:
                low, high = value.split(separator, maxsplit=1)
                return low.strip(), high.strip()
    raise ValueError("Invalid melody range constraint")


def _track_range_metadata(
    track: Track,
    instrument_id: str,
    catalog: InstrumentCatalog,
) -> dict[str, str]:
    existing = track.metadata.get("range")
    if isinstance(existing, dict) and "low" in existing and "high" in existing:
        return {"low": str(existing["low"]), "high": str(existing["high"])}
    low, high = catalog.get(instrument_id).comfortable_range
    return {"low": low, "high": high}


def _track_notes(track: Track) -> Iterable[NoteEvent]:
    for bar in track.bars:
        for event in bar.events:
            if isinstance(event, NoteEvent):
                yield event


def _track_rests(track: Track) -> Iterable[RestEvent]:
    for bar in track.bars:
        for event in bar.events:
            if isinstance(event, RestEvent):
                yield event


def _phrase_lengths(track: Track) -> list[int]:
    phrase_ids_by_bar = {
        bar.number: bar.metadata.get("melody_engine_phrase_id") or bar.metadata.get("phrase_index")
        for bar in track.bars
    }
    lengths: dict[str, int] = {}
    for phrase_id in phrase_ids_by_bar.values():
        if phrase_id is None:
            continue
        key = str(phrase_id)
        lengths[key] = lengths.get(key, 0) + 1
    return sorted(lengths.values()) or [4]


def _harmony_score(
    project: ArrangementProject,
    track: Track,
    *,
    chord_parser: ChordParser,
) -> float | None:
    chords_by_bar = _chords_by_bar(project.chord_grid)
    checked = 0
    supported = 0
    for bar in track.bars:
        for event in bar.events:
            if not isinstance(event, NoteEvent):
                continue
            chord = _active_chord(chords_by_bar, bar.number, event.start)
            parsed = _safe_parse(chord_parser, chord.symbol)
            if parsed is None:
                continue
            checked += 1
            if _is_supported(event, parsed):
                supported += 1
    if checked == 0:
        return None
    return round(supported / checked, 3)


def _is_supported(event: NoteEvent, parsed: ParsedChord) -> bool:
    pitch_class = note_to_midi(event.pitch) % 12
    allowed = {
        *parsed.chord_tone_pcs,
        *parsed.tension_pcs,
        *parsed.alteration_pcs,
    }
    role = str(event.annotations.get("melodic_role") or "")
    return (
        pitch_class in allowed
        or role.startswith("approach")
        or role in {"cadence", "retrieved_motif"}
    )


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


def _palette(chord_symbol: str, chord_parser: ChordParser) -> Any:
    parsed = _safe_parse(chord_parser, chord_symbol)
    if parsed is None:
        return _FallbackPalette()
    return parsed


class _FallbackPalette:
    root_pc = 0
    chord_tone_pcs = (0, 4, 7)
    guide_tone_pcs = (4, 10)
    tension_pcs: tuple[int, ...] = ()
    alteration_pcs: tuple[int, ...] = ()
    prefer_sharps = False


def _safe_parse(chord_parser: ChordParser, chord_symbol: str) -> ParsedChord | None:
    try:
        return chord_parser.parse(chord_symbol)
    except ValueError:
        return None


def _supported_pc(target_pc: int, palette: Any) -> int:
    allowed = tuple(
        dict.fromkeys(
            [
                *getattr(palette, "chord_tone_pcs", ()),
                *getattr(palette, "tension_pcs", ()),
                *getattr(palette, "guide_tone_pcs", ()),
            ]
        )
    )
    if not allowed or target_pc in allowed:
        return target_pc
    return min(allowed, key=lambda pc: _pc_distance(pc, target_pc))


def _prefer_sharps(palette: Any) -> bool:
    explicit = getattr(palette, "prefer_sharps", None)
    if explicit is not None:
        return bool(explicit)
    root = str(getattr(palette, "root", ""))
    return "#" in root and "b" not in root


def _nearest_note_in_range(
    target_pc: int,
    *,
    melody_range: MelodyRange,
    anchor_midi: int,
    prefer_sharps: bool,
) -> tuple[str, int]:
    candidates = [
        midi_note
        for midi_note in range(melody_range.low_midi, melody_range.high_midi + 1)
        if midi_note % 12 == target_pc % 12
    ]
    if not candidates:
        fallback = min(max(anchor_midi, melody_range.low_midi), melody_range.high_midi)
        return midi_to_note(fallback, prefer_sharps=prefer_sharps), fallback
    selected = min(
        candidates,
        key=lambda midi_note: (
            abs(midi_note - anchor_midi),
            abs(midi_note - melody_range.midpoint),
        ),
    )
    return midi_to_note(selected, prefer_sharps=prefer_sharps), selected


def _fill_rests(notes: list[NoteEvent], expected: float) -> list[NoteEvent | RestEvent]:
    events: list[NoteEvent | RestEvent] = []
    cursor = 0.0
    for note in sorted(notes, key=lambda item: (item.start, item.pitch)):
        if note.start > cursor + 1e-6:
            events.append(RestEvent(start=round(cursor, 3), duration=round(note.start - cursor, 3)))
        events.append(note)
        cursor = max(cursor, note.start + note.duration)
    if cursor < expected - 1e-6:
        events.append(RestEvent(start=round(cursor, 3), duration=round(expected - cursor, 3)))
    return events


def _retrieval_starts(count: int, bar_duration: float) -> list[float]:
    preferred = [0.0, 0.75, 1.5, 2.5, 3.0, 3.5]
    starts = [value for value in preferred if value < bar_duration]
    if len(starts) >= count:
        return starts[:count]
    step = bar_duration / max(1, count + 1)
    return [round(index * step, 3) for index in range(count)]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _contour(notes: list[NoteEvent]) -> list[int]:
    midi_notes = [note_to_midi(event.pitch) for event in notes]
    contour = []
    for left, right in zip(midi_notes, midi_notes[1:], strict=False):
        if right > left:
            contour.append(1)
        elif right < left:
            contour.append(-1)
        else:
            contour.append(0)
    return contour


def _pc_distance(left: int, right: int) -> int:
    direct = abs((left % 12) - (right % 12))
    return min(direct, 12 - direct)


def _issue(
    code: str,
    message: str,
    *,
    track_id: str | None = None,
    bar_number: int | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "track_id": track_id,
        "bar_number": bar_number,
    }
