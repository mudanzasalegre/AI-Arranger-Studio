from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from arranger_core.catalogs import InstrumentCatalog
from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.harmony_engine import generate_harmony_project
from arranger_core.music_theory import midi_to_note, note_to_midi, pitch_class, pitch_class_name
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


@dataclass(frozen=True)
class MelodyRange:
    low: str
    high: str
    low_midi: int
    high_midi: int

    @property
    def midpoint(self) -> int:
        return (self.low_midi + self.high_midi) // 2


@dataclass(frozen=True)
class MelodySlot:
    start: float
    duration: float
    role: str
    articulation: str | None = None
    dynamic: str | None = None


@dataclass(frozen=True)
class ChordPalette:
    symbol: str
    root_pc: int
    chord_tone_pcs: tuple[int, ...]
    guide_tone_pcs: tuple[int, ...]
    tension_pcs: tuple[int, ...]
    prefer_sharps: bool


MOTIF_SLOTS = (
    MelodySlot(0.0, 1.0, "root", "tenuto", "mf"),
    MelodySlot(1.0, 0.5, "third", "staccato"),
    MelodySlot(1.5, 0.5, "approach_upper", "staccato"),
    MelodySlot(2.0, 1.0, "seventh", "accent"),
    MelodySlot(3.0, 1.0, "rest"),
)
VARIATION_SLOTS = (
    MelodySlot(0.0, 0.5, "rest"),
    MelodySlot(0.5, 0.5, "third", "staccato", "mf"),
    MelodySlot(1.0, 1.0, "fifth", "tenuto"),
    MelodySlot(2.0, 0.5, "approach_lower", "staccato"),
    MelodySlot(2.5, 0.5, "seventh", "accent"),
    MelodySlot(3.0, 1.0, "rest"),
)
SEQUENCE_SLOTS = (
    MelodySlot(0.0, 1.0, "fifth", "tenuto", "mf"),
    MelodySlot(1.0, 0.5, "color", "staccato"),
    MelodySlot(1.5, 0.5, "approach_upper", "staccato"),
    MelodySlot(2.0, 1.0, "third", "accent"),
    MelodySlot(3.0, 1.0, "rest"),
)
CADENCE_SLOTS = (
    MelodySlot(0.0, 1.0, "seventh", "tenuto", "mf"),
    MelodySlot(1.0, 2.0, "cadence", "tenuto"),
    MelodySlot(3.0, 1.0, "rest"),
)


class LeadSheetGenerator:
    def __init__(
        self,
        *,
        chord_parser: ChordParser | None = None,
        instrument_catalog: InstrumentCatalog | None = None,
    ) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()

    def generate(
        self,
        spec: GenerationSpec,
        *,
        project_id: str | None = None,
    ) -> ArrangementProject:
        project = generate_harmony_project(
            spec,
            project_id=project_id or f"lead-sheet-{spec.seed}-{spec.form}",
            include_lead_sheet_track=False,
        )
        rng = random.Random(spec.seed)
        instrument_id = _select_lead_instrument(spec, self.instrument_catalog)
        melody_range = _melody_range(spec, instrument_id, self.instrument_catalog)
        phrase_length = _phrase_length_bars(project.bar_count)
        motif_shift = rng.choice((-2, -1, 0, 1, 2))

        track = self._generate_track(
            project=project,
            instrument_id=instrument_id,
            melody_range=melody_range,
            phrase_length=phrase_length,
            motif_shift=motif_shift,
        )
        metadata = {
            **project.metadata,
            "lead_sheet_generator": "rule_based_v0",
            "lead_instrument": instrument_id,
            "melody_range": {"low": melody_range.low, "high": melody_range.high},
            "phrase_length_bars": phrase_length,
            "initial_motif": {
                "rhythm": [slot.duration for slot in MOTIF_SLOTS],
                "roles": [slot.role for slot in MOTIF_SLOTS if slot.role != "rest"],
            },
            "motivic_variations": [
                "rhythmic_displacement",
                "guide_tone_sequence",
                "cadential_answer",
            ],
        }
        return project.model_copy(update={"metadata": metadata, "tracks": [track]})

    def _generate_track(
        self,
        *,
        project: ArrangementProject,
        instrument_id: str,
        melody_range: MelodyRange,
        phrase_length: int,
        motif_shift: int,
    ) -> Track:
        chords_by_bar = _chords_by_bar(project.chord_grid)
        previous_midi = melody_range.midpoint
        bars: list[Bar] = []

        for bar_number in range(1, project.bar_count + 1):
            slots = _slots_for_bar(
                bar_duration=meter_to_quarter_beats(project.meter_at_bar(bar_number)),
                phrase_position=(bar_number - 1) % phrase_length,
                phrase_length=phrase_length,
            )
            events, previous_midi = self._events_for_slots(
                bar_number=bar_number,
                slots=slots,
                chords_by_bar=chords_by_bar,
                max_bar=project.bar_count,
                melody_range=melody_range,
                previous_midi=previous_midi,
                motif_shift=motif_shift,
            )
            bars.append(
                Bar(
                    number=bar_number,
                    events=events,
                    metadata={
                        "phrase_index": (bar_number - 1) // phrase_length + 1,
                        "phrase_position": (bar_number - 1) % phrase_length + 1,
                        "breath_after_bar": any(
                            isinstance(event, RestEvent)
                            and event.annotations.get("breath")
                            for event in events
                        ),
                    },
                )
            )

        return Track(
            id="lead_sheet",
            instrument=instrument_id,
            role="melody",
            name="Lead Sheet",
            bars=bars,
            metadata={
                "generator": "LeadSheetGenerator",
                "phrase_length_bars": phrase_length,
                "range": {"low": melody_range.low, "high": melody_range.high},
            },
        )

    def _events_for_slots(
        self,
        *,
        bar_number: int,
        slots: tuple[MelodySlot, ...],
        chords_by_bar: dict[int, list[ChordSymbol]],
        max_bar: int,
        melody_range: MelodyRange,
        previous_midi: int,
        motif_shift: int,
    ) -> tuple[list[NoteEvent | RestEvent], int]:
        events: list[NoteEvent | RestEvent] = []
        for slot in slots:
            if slot.role == "rest":
                events.append(
                    RestEvent(
                        start=slot.start,
                        duration=slot.duration,
                        annotations={"breath": True},
                    )
                )
                continue

            active_chord = _active_chord(chords_by_bar, bar_number, slot.start)
            next_chord = _next_chord(chords_by_bar, bar_number, slot.start, max_bar)
            palette = self._palette(active_chord.symbol)
            next_palette = self._palette(next_chord.symbol)
            target_pc = _target_pitch_class(
                role=slot.role,
                palette=palette,
                next_palette=next_palette,
                motif_shift=motif_shift,
            )
            pitch, previous_midi = _nearest_note_in_range(
                target_pc,
                melody_range=melody_range,
                anchor_midi=previous_midi,
                prefer_sharps=palette.prefer_sharps,
            )
            events.append(
                NoteEvent(
                    pitch=pitch,
                    start=slot.start,
                    duration=slot.duration,
                    velocity=_velocity_for_slot(slot),
                    articulations=[slot.articulation] if slot.articulation else [],
                    dynamic=slot.dynamic,
                    annotations={
                        "melodic_role": slot.role,
                        "source_chord": active_chord.symbol,
                    },
                )
            )
        return events, previous_midi

    def _palette(self, chord_symbol: str) -> ChordPalette:
        try:
            parsed = self.chord_parser.parse(chord_symbol)
        except ValueError:
            root = _fallback_root(chord_symbol)
            root_pc = pitch_class(root)
            return ChordPalette(
                symbol=chord_symbol,
                root_pc=root_pc,
                chord_tone_pcs=(root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12),
                guide_tone_pcs=((root_pc + 4) % 12, (root_pc + 10) % 12),
                tension_pcs=(),
                prefer_sharps="#" in root and "b" not in root,
            )

        guide_tones = _guide_tones(parsed)
        chord_tones = tuple(parsed.chord_tone_pcs) or (parsed.root_pc,)
        return ChordPalette(
            symbol=chord_symbol,
            root_pc=parsed.root_pc,
            chord_tone_pcs=tuple(chord_tones),
            guide_tone_pcs=guide_tones or chord_tones[:2],
            tension_pcs=tuple(parsed.tension_pcs),
            prefer_sharps="#" in parsed.root and "b" not in parsed.root,
        )


def generate_lead_sheet_project(
    spec: GenerationSpec,
    *,
    project_id: str | None = None,
) -> ArrangementProject:
    return LeadSheetGenerator().generate(spec, project_id=project_id)


def _select_lead_instrument(spec: GenerationSpec, catalog: InstrumentCatalog) -> str:
    configured = spec.constraints.get("lead_instrument")
    if isinstance(configured, str):
        catalog.get(configured)
        return configured

    candidate_ids = list(spec.instruments)
    if not candidate_ids:
        try:
            candidate_ids = [
                instrument.id
                for instrument in catalog.instruments_for_ensemble(spec.ensemble)
            ]
        except KeyError:
            candidate_ids = []

    for instrument_id in candidate_ids:
        try:
            instrument = catalog.get(instrument_id)
        except KeyError:
            continue
        if instrument.breath_required and instrument.family != "percussion":
            return instrument_id

    for instrument_id in candidate_ids:
        if instrument_id not in {"drum_kit", "double_bass"}:
            return instrument_id

    return "piano"


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
        try:
            low, high = catalog.get(instrument_id).comfortable_range
        except KeyError:
            low, high = ("C4", "Bb5")

    low_midi = note_to_midi(low)
    high_midi = note_to_midi(high)
    if low_midi > high_midi:
        raise ValueError(f"Invalid melody range: {low} is above {high}")
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
    raise ValueError("Melody range must be {'low': note, 'high': note}, [low, high], or 'low-high'")


def _phrase_length_bars(bar_count: int) -> int:
    return 4 if bar_count % 4 == 0 else 2


def _slots_for_bar(
    *,
    bar_duration: float,
    phrase_position: int,
    phrase_length: int,
) -> tuple[MelodySlot, ...]:
    if abs(bar_duration - 4.0) > 1e-6:
        note_duration = max(0.5, bar_duration - 1.0)
        return (
            MelodySlot(
                0.0,
                note_duration,
                "cadence" if phrase_position else "third",
                "tenuto",
                "mf",
            ),
            MelodySlot(note_duration, bar_duration - note_duration, "rest"),
        )

    if phrase_position == phrase_length - 1:
        return CADENCE_SLOTS
    if phrase_position == 1:
        return VARIATION_SLOTS
    if phrase_position == 2:
        return SEQUENCE_SLOTS
    return MOTIF_SLOTS


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


def _next_chord(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    start: float,
    max_bar: int,
) -> ChordSymbol:
    for chord in chords_by_bar.get(bar_number, []):
        if chord.beat - 1.0 > start + 1e-6:
            return chord
    for next_bar in range(bar_number + 1, max_bar + 1):
        chords = chords_by_bar.get(next_bar, [])
        if chords:
            return chords[0]
    return _active_chord(chords_by_bar, bar_number, start)


def _guide_tones(parsed: ParsedChord) -> tuple[int, ...]:
    guide_intervals = [
        interval
        for interval in parsed.chord_tone_intervals
        if interval % 12 in {3, 4, 10, 11}
    ]
    return tuple((parsed.root_pc + interval) % 12 for interval in guide_intervals)


def _target_pitch_class(
    *,
    role: str,
    palette: ChordPalette,
    next_palette: ChordPalette,
    motif_shift: int,
) -> int:
    if role == "root":
        return palette.root_pc
    if role == "third":
        return _palette_pick(palette, preferred=(3, 4), fallback=palette.guide_tone_pcs)
    if role == "seventh":
        return _palette_pick(palette, preferred=(10, 11), fallback=palette.guide_tone_pcs)
    if role == "fifth":
        return _palette_pick(palette, preferred=(6, 7, 8), fallback=palette.chord_tone_pcs)
    if role == "color":
        candidates = palette.tension_pcs or palette.chord_tone_pcs
        return candidates[motif_shift % len(candidates)]
    if role == "approach_upper":
        return (_cadence_target(next_palette) + 1) % 12
    if role == "approach_lower":
        return (_cadence_target(next_palette) - 1) % 12
    if role == "cadence":
        return _cadence_target(next_palette)
    return palette.root_pc


def _palette_pick(
    palette: ChordPalette,
    *,
    preferred: tuple[int, ...],
    fallback: tuple[int, ...],
) -> int:
    preferred_pcs = {(palette.root_pc + interval) % 12 for interval in preferred}
    for pc in (*palette.chord_tone_pcs, *palette.tension_pcs):
        if pc in preferred_pcs:
            return pc
    return (fallback or palette.chord_tone_pcs or (palette.root_pc,))[0]


def _cadence_target(palette: ChordPalette) -> int:
    return (palette.guide_tone_pcs or palette.chord_tone_pcs or (palette.root_pc,))[0]


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


def _velocity_for_slot(slot: MelodySlot) -> int:
    if slot.articulation == "accent":
        return 92
    if slot.articulation == "staccato":
        return 76
    return 84


def _fallback_root(chord_symbol: str) -> str:
    for length in (2, 1):
        root = chord_symbol[:length]
        try:
            pitch_class(root)
            return root
        except ValueError:
            continue
    return pitch_class_name(0)
