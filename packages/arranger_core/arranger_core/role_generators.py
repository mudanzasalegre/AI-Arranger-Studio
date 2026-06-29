from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from arranger_core.bass_engine import BassEngine
from arranger_core.catalogs import InstrumentCatalog
from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.drums_engine import DrumsEngine
from arranger_core.harmony_engine import generate_harmony_project
from arranger_core.melody_engine import MelodyAiInfillBackend, MelodyEngine
from arranger_core.music_theory import midi_to_note, note_to_midi, pitch_class, pitch_class_name
from arranger_core.performance import PerformanceMapper
from arranger_core.piano_comping_engine import PianoCompingEngine
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
from arranger_core.song_planner import SongPlan, generate_song_plan

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
DRUM_GRID = tuple(index * 0.5 for index in range(8))


@dataclass(frozen=True)
class ChordInfo:
    symbol: str
    root: str
    root_pc: int
    quality: str
    chord_tone_pcs: tuple[int, ...]
    guide_tone_pcs: tuple[int, ...]
    tension_pcs: tuple[int, ...]
    prefer_sharps: bool


@dataclass
class GenerationContext:
    spec: GenerationSpec
    project: ArrangementProject
    instrument_ids: list[str]
    chord_parser: ChordParser
    instrument_catalog: InstrumentCatalog
    rng: random.Random
    learned_patterns: dict[str, list[dict[str, Any]]]
    song_plan: SongPlan | None = None

    @property
    def chords_by_bar(self) -> dict[int, list[ChordSymbol]]:
        return _chords_by_bar(self.project.chord_grid)


class RoleGenerator(Protocol):
    role: str

    def generate(self, context: GenerationContext) -> Track | list[Track]: ...


class RuleBasedArranger:
    def __init__(
        self,
        *,
        chord_parser: ChordParser | None = None,
        instrument_catalog: InstrumentCatalog | None = None,
        drums_generator: RoleGenerator | None = None,
        bass_generator: RoleGenerator | None = None,
        piano_generator: RoleGenerator | None = None,
        melody_generator: RoleGenerator | None = None,
        horn_response_generator: RoleGenerator | None = None,
    ) -> None:
        self.chord_parser = chord_parser or ChordParser.load_default()
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()
        self.drums = drums_generator or DrumsGenerator()
        self.bass = bass_generator or WalkingBassGenerator()
        self.piano = piano_generator or PianoCompingGenerator()
        self.melody = melody_generator or MelodyGenerator(
            chord_parser=self.chord_parser,
            instrument_catalog=self.instrument_catalog,
        )
        self.horn_response = horn_response_generator or HornResponseGenerator()
        self.shout = ShoutChorusGenerator()
        self.humanizer = Humanizer()

    def generate(
        self,
        spec: GenerationSpec,
        *,
        project_id: str | None = None,
    ) -> ArrangementProject:
        project = generate_harmony_project(
            spec,
            project_id=project_id or f"arrangement-{spec.seed}-{spec.form}",
            include_lead_sheet_track=False,
        )
        song_plan = generate_song_plan(spec, project)
        instrument_ids = _instrument_ids_for_spec(spec, self.instrument_catalog)
        context = GenerationContext(
            spec=spec,
            project=project,
            instrument_ids=instrument_ids,
            chord_parser=self.chord_parser,
            instrument_catalog=self.instrument_catalog,
            rng=random.Random(spec.seed),
            learned_patterns=_load_learned_patterns(spec),
            song_plan=song_plan,
        )

        tracks: list[Track] = []
        if "drum_kit" in instrument_ids:
            tracks.append(self.drums.generate(context))
        if "double_bass" in instrument_ids:
            tracks.append(self.bass.generate(context))
        piano_track = self.piano.generate(context) if "piano" in instrument_ids else None

        lead_instrument = _lead_instrument(instrument_ids, self.instrument_catalog)
        horn_tracks: list[Track] = []
        if lead_instrument is not None:
            melody_track = self.melody.generate_for_instrument(context, lead_instrument)
            if piano_track is not None and lead_instrument == "piano":
                piano_track = _merge_melody_into_piano(piano_track, melody_track)
            else:
                horn_tracks.append(melody_track)

        response_instruments = [
            instrument_id
            for instrument_id in instrument_ids
            if _is_horn(instrument_id, self.instrument_catalog) and instrument_id != lead_instrument
        ]
        for harmony_index, instrument_id in enumerate(response_instruments, start=1):
            horn_tracks.append(
                self.horn_response.generate_for_instrument(
                    context,
                    instrument_id,
                    harmony_index=harmony_index,
                )
            )

        horn_tracks = [
            self.shout.apply_to_track(context, track, harmony_index=index)
            for index, track in enumerate(horn_tracks)
        ]

        if piano_track is not None:
            tracks.append(piano_track)
        tracks.extend(horn_tracks)

        arranged = project.model_copy(
            update={
                "tracks": tracks,
                "metadata": {
                    **project.metadata,
                    "arranger": _arranger_name(self),
                    "role_generators": _role_generator_names(self),
                    "ensemble_instruments": instrument_ids,
                    "pattern_index_used": bool(context.learned_patterns),
                    "song_plan": song_plan.model_dump(mode="json"),
                },
            }
        )
        if spec.constraints.get("humanize", True) is not False:
            arranged = self.humanizer.apply(arranged, seed=spec.seed)
        return arranged


class DrumsGenerator:
    role = "drums"

    def __init__(self, *, engine: DrumsEngine | None = None) -> None:
        self.engine = engine or DrumsEngine()

    def generate(self, context: GenerationContext) -> Track:
        return self.engine.generate(context)

    def _bar_events(
        self,
        bar_number: int,
        *,
        is_fill: bool,
        learned_pattern: dict | None,
        context: GenerationContext,
        bar_duration: float,
    ) -> list[NoteEvent]:
        grid = _grid_for_duration(bar_duration)
        if is_fill:
            fill_cycle = ("snare", "low_tom", "snare", "mid_tom", "snare", "high_tom")
            events: list[NoteEvent] = []
            for index, start in enumerate(grid):
                drum_name = fill_cycle[index % len(fill_cycle)]
                if start == grid[-1]:
                    drum_name = "crash"
                events.append(
                    _drum_note(
                        drum_name,
                        start=start,
                        velocity=96 if drum_name == "crash" else 82,
                        annotations={"fill": True, "bar": bar_number},
                    )
                )
            return events

        learned_events = _drum_events_from_pattern(learned_pattern, bar_duration=bar_duration)
        if learned_events:
            return learned_events

        feel = _style_feel(context.spec)
        if feel == "bossa":
            return _bossa_drum_events(grid)
        if feel == "funk":
            return _funk_drum_events(grid)
        if feel == "waltz":
            return _waltz_drum_events(bar_duration)
        if feel == "ballad":
            return _ballad_drum_events(grid)

        events = []
        for start in grid:
            events.append(_drum_note("ride", start=start, velocity=76))
            if start in {1.0, 3.0}:
                events.append(_drum_note("hihat_pedal", start=start, velocity=72))
            if start in {0.0, 2.0}:
                events.append(_drum_note("kick", start=start, velocity=48))
            if start in {1.5, 3.5}:
                events.append(_drum_note("snare", start=start, velocity=42))
        return events


class WalkingBassGenerator:
    role = "walking_bass"

    def __init__(self, *, engine: BassEngine | None = None) -> None:
        self.engine = engine or BassEngine()

    def generate(self, context: GenerationContext) -> Track:
        return self.engine.generate(context)


class PianoCompingGenerator:
    role = "comping"

    def __init__(self, *, engine: PianoCompingEngine | None = None) -> None:
        self.engine = engine or PianoCompingEngine()

    def generate(self, context: GenerationContext) -> Track:
        return self.engine.generate(context)


class MelodyGenerator:
    role = "melody"

    def __init__(
        self,
        *,
        chord_parser: ChordParser,
        instrument_catalog: InstrumentCatalog,
        ai_backend: MelodyAiInfillBackend | None = None,
    ) -> None:
        self.engine = MelodyEngine(
            chord_parser=chord_parser,
            instrument_catalog=instrument_catalog,
            ai_backend=ai_backend,
        )
        self.instrument_catalog = instrument_catalog

    def generate(self, context: GenerationContext) -> Track:
        instrument_id = _lead_instrument(context.instrument_ids, context.instrument_catalog)
        return self.generate_for_instrument(context, instrument_id or "piano")

    def generate_for_instrument(
        self,
        context: GenerationContext,
        instrument_id: str,
    ) -> Track:
        return self.engine.generate_for_instrument(context, instrument_id)


class HornResponseGenerator:
    role = "horn_response"

    def generate(self, context: GenerationContext) -> list[Track]:
        instruments = [
            instrument_id
            for instrument_id in context.instrument_ids
            if _is_horn(instrument_id, context.instrument_catalog)
        ]
        return [
            self.generate_for_instrument(context, instrument_id, harmony_index=index)
            for index, instrument_id in enumerate(instruments)
        ]

    def generate_for_instrument(
        self,
        context: GenerationContext,
        instrument_id: str,
        *,
        harmony_index: int,
    ) -> Track:
        bars: list[Bar] = []
        low, high = context.instrument_catalog.get(instrument_id).comfortable_range
        for bar_number in range(1, context.project.bar_count + 1):
            chord = _active_chord(context.chords_by_bar, bar_number, 0.0)
            chord_info = _parse_chord(context.chord_parser, chord.symbol)
            bar_duration = _bar_duration(context.project, bar_number)
            note_events = _horn_response_notes(
                chord_info,
                instrument_id=instrument_id,
                low=low,
                high=high,
                harmony_index=harmony_index,
                bar_number=bar_number,
                bar_duration=bar_duration,
            )
            bars.append(
                Bar(
                    number=bar_number,
                    events=_with_rests(note_events, bar_duration, voice=1),
                    metadata={"response": bool(note_events)},
                )
            )
        return Track(
            id=instrument_id,
            instrument=instrument_id,
            role=self.role,
            name=_display_name(instrument_id, context.instrument_catalog),
            bars=bars,
            metadata={
                "generator": "HornResponseGenerator",
                "harmony_index": harmony_index,
            },
        )


class ShoutChorusGenerator:
    role = "shout_chorus"

    def apply_to_track(
        self,
        context: GenerationContext,
        track: Track,
        *,
        harmony_index: int,
    ) -> Track:
        if not _is_horn(track.instrument, context.instrument_catalog):
            return track

        low, high = context.instrument_catalog.get(track.instrument).comfortable_range
        shout_start_bar = max(1, context.project.bar_count - 3)
        bars: list[Bar] = []
        for bar in track.bars:
            if bar.number < shout_start_bar:
                bars.append(bar)
                continue
            chord = _active_chord(context.chords_by_bar, bar.number, 0.0)
            chord_info = _parse_chord(context.chord_parser, chord.symbol)
            bar_duration = _bar_duration(context.project, bar.number)
            shout_events = _shout_notes(
                chord_info,
                low=low,
                high=high,
                harmony_index=harmony_index,
                bar_duration=bar_duration,
            )
            bars.append(
                bar.model_copy(
                    update={
                        "events": [
                            *bar.events,
                            *_with_rests(shout_events, bar_duration, voice=2),
                        ],
                        "metadata": {**bar.metadata, "shout_chorus": True},
                    }
                )
            )
        return track.model_copy(
            update={
                "bars": bars,
                "metadata": {**track.metadata, "shout_chorus": "last_4_bars"},
            }
        )


class Humanizer:
    role = "humanizer"

    def apply(self, project: ArrangementProject, *, seed: int) -> ArrangementProject:
        return PerformanceMapper().apply(project, seed=seed, default_source="rule_based")


def generate_arrangement(
    spec: GenerationSpec,
    *,
    project_id: str | None = None,
) -> ArrangementProject:
    return RuleBasedArranger().generate(spec, project_id=project_id)


def _arranger_name(arranger: RuleBasedArranger) -> str:
    generators = (
        arranger.drums,
        arranger.bass,
        arranger.piano,
        arranger.melody,
        arranger.horn_response,
    )
    return (
        "hybrid_rule_model_v0"
        if any(hasattr(generator, "backend") for generator in generators)
        else "rule_based_v0"
    )


def _role_generator_names(arranger: RuleBasedArranger) -> list[str]:
    return [
        arranger.drums.__class__.__name__,
        arranger.bass.__class__.__name__,
        arranger.piano.__class__.__name__,
        arranger.melody.__class__.__name__,
        arranger.horn_response.__class__.__name__,
        arranger.shout.__class__.__name__,
        arranger.humanizer.__class__.__name__,
    ]


def _instrument_ids_for_spec(
    spec: GenerationSpec,
    catalog: InstrumentCatalog,
) -> list[str]:
    if spec.instruments:
        return list(dict.fromkeys(spec.instruments))
    try:
        return [instrument.id for instrument in catalog.instruments_for_ensemble(spec.ensemble)]
    except KeyError:
        return ["drum_kit", "double_bass", "piano"]


def _lead_instrument(
    instrument_ids: list[str],
    catalog: InstrumentCatalog,
) -> str | None:
    for instrument_id in instrument_ids:
        if _is_horn(instrument_id, catalog):
            return instrument_id
    return "piano" if "piano" in instrument_ids else None


def _is_horn(instrument_id: str, catalog: InstrumentCatalog) -> bool:
    try:
        instrument = catalog.get(instrument_id)
    except KeyError:
        return False
    return instrument.breath_required and instrument.family in {"woodwind", "brass"}


def _display_name(instrument_id: str, catalog: InstrumentCatalog) -> str:
    try:
        return catalog.get(instrument_id).display_name
    except KeyError:
        return instrument_id


def _merge_melody_into_piano(piano_track: Track, melody_track: Track) -> Track:
    melody_by_bar = {bar.number: bar for bar in melody_track.bars}
    bars = []
    for bar in piano_track.bars:
        melody_bar = melody_by_bar.get(bar.number)
        if melody_bar is None:
            bars.append(bar)
            continue
        melody_events = [event.model_copy(update={"voice": 2}) for event in melody_bar.events]
        bars.append(
            bar.model_copy(
                update={
                    "events": sorted(
                        [*bar.events, *melody_events],
                        key=lambda event: (event.voice, event.start, event.duration),
                    ),
                    "metadata": {**bar.metadata, "piano_melody": True},
                }
            )
        )
    return piano_track.model_copy(
        update={
            "role": "piano",
            "bars": bars,
            "metadata": {
                **piano_track.metadata,
                "contains_melody": True,
                "melody_engine": melody_track.metadata.get("melody_engine_version"),
                "melody_engine_mode": melody_track.metadata.get("melody_engine_mode"),
                "motif_ledger": melody_track.metadata.get("motif_ledger"),
                "melody_validation": melody_track.metadata.get("melody_validation"),
            },
        }
    )


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
        velocity=velocity,
        annotations={"drum": drum_name, **(annotations or {})},
    )


def _is_fill_bar(project: ArrangementProject, bar_number: int) -> bool:
    if bar_number == project.bar_count:
        return True
    if bar_number % 4 == 0:
        return True
    return any(section.start_bar == bar_number + 1 for section in project.form)


def _bar_duration(project: ArrangementProject, bar_number: int) -> float:
    return meter_to_quarter_beats(project.meter_at_bar(bar_number))


def _grid_for_duration(bar_duration: float) -> tuple[float, ...]:
    steps = max(1, round(bar_duration / 0.5))
    return tuple(index * 0.5 for index in range(steps))


def _style_feel(spec: GenerationSpec) -> str:
    if spec.meter == "3/4" or spec.style == "jazz_waltz":
        return "waltz"
    if spec.style == "bossa_nova":
        return "bossa"
    if spec.style == "funk_jazz":
        return "funk"
    if spec.style == "jazz_ballad":
        return "ballad"
    return "swing"


def _bossa_drum_events(grid: tuple[float, ...]) -> list[NoteEvent]:
    events = []
    for start in grid:
        events.append(_drum_note("closed_hihat", start=start, velocity=54))
        if start in {0.0, 2.0}:
            events.append(_drum_note("kick", start=start, velocity=58))
        if start in {1.0, 3.0}:
            events.append(_drum_note("snare", start=start, velocity=44))
    return events


def _funk_drum_events(grid: tuple[float, ...]) -> list[NoteEvent]:
    events = []
    for start in grid:
        events.append(_drum_note("closed_hihat", start=start, velocity=72))
        if start in {0.0, 1.5, 2.5}:
            events.append(_drum_note("kick", start=start, velocity=82))
        if start in {1.0, 3.0}:
            events.append(_drum_note("snare", start=start, velocity=88))
    return events


def _waltz_drum_events(bar_duration: float) -> list[NoteEvent]:
    events = []
    for start in _grid_for_duration(bar_duration):
        events.append(_drum_note("ride", start=start, velocity=62))
    events.append(_drum_note("kick", start=0.0, velocity=56))
    if bar_duration >= 3.0:
        events.append(_drum_note("hihat_pedal", start=1.0, velocity=62))
        events.append(_drum_note("snare", start=2.0, velocity=46))
    return events


def _ballad_drum_events(grid: tuple[float, ...]) -> list[NoteEvent]:
    events = []
    for start in grid:
        events.append(_drum_note("ride", start=start, velocity=46 if start % 1.0 else 54))
        if start in {1.0, 3.0}:
            events.append(_drum_note("hihat_pedal", start=start, velocity=48))
        if start == 2.0:
            events.append(_drum_note("kick", start=start, velocity=38))
    return events


def _walking_bass_notes(
    chord_info: ChordInfo,
    next_info: ChordInfo,
    *,
    previous_midi: int,
    beats_per_bar: int = 4,
    style: str = "hard_bop",
) -> tuple[list[str], int]:
    low = note_to_midi("E1")
    high = note_to_midi("C4")
    if style == "bossa_nova":
        beat_pcs = [
            chord_info.root_pc,
            _pc_at_interval(chord_info, (7, 5)),
            chord_info.root_pc,
            _pc_at_interval(chord_info, (7, 10, 11)),
        ]
    elif beats_per_bar == 3:
        beat_pcs = [
            chord_info.root_pc,
            _pc_at_interval(chord_info, (3, 4, 7)),
            (next_info.root_pc - 1) % 12,
        ]
    else:
        beat_pcs = [
            chord_info.root_pc,
            _pc_at_interval(chord_info, (3, 4, 7)),
            _pc_at_interval(chord_info, (7, 10, 11, 3, 4)),
            (next_info.root_pc - 1) % 12,
        ]
    beat_pcs = beat_pcs[:beats_per_bar]
    notes: list[str] = []
    anchor = previous_midi
    for pc in beat_pcs:
        note_name, anchor = _nearest_note_in_range(
            pc,
            low_midi=low,
            high_midi=high,
            anchor_midi=anchor,
            prefer_sharps=False,
        )
        notes.append(note_name)
    return notes, anchor


def _rootless_voicing(chord_info: ChordInfo, *, anchor_midi: int) -> list[str]:
    low = note_to_midi("C3")
    high = note_to_midi("G5")
    candidate_pcs = [
        pc
        for pc in (
            *chord_info.guide_tone_pcs,
            *_default_tension_pcs(chord_info),
            *chord_info.chord_tone_pcs,
        )
        if pc != chord_info.root_pc
    ]
    selected_pcs = list(dict.fromkeys(candidate_pcs))[:4]
    notes: list[str] = []
    anchor = anchor_midi
    for pc in selected_pcs:
        note_name, anchor = _nearest_note_in_range(
            pc,
            low_midi=low,
            high_midi=high,
            anchor_midi=anchor,
            prefer_sharps=chord_info.prefer_sharps,
        )
        notes.append(note_name)
        anchor += 4
    return sorted(notes, key=note_to_midi)


def _default_tension_pcs(chord_info: ChordInfo) -> tuple[int, ...]:
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


def _comping_rhythm(
    bar_number: int,
    density: str,
    *,
    style: str,
    bar_duration: float,
) -> list[float]:
    if bar_duration == 3.0:
        cells = ([0.0, 1.5], [1.0, 2.0], [0.5, 2.0])
        return list(cells[(bar_number - 1) % len(cells)])
    if style == "bossa_nova":
        cells = ([0.0, 1.5, 2.5], [0.5, 2.0, 3.0], [0.0, 1.0, 2.5])
        return list(cells[(bar_number - 1) % len(cells)])
    if style == "funk_jazz":
        cells = ([0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5])
        return list(cells[(bar_number - 1) % len(cells)])
    if style == "jazz_ballad":
        cells = ([0.0, 2.0], [1.5, 3.0])
        return list(cells[(bar_number - 1) % len(cells)])
    if density == "low":
        cells = ([0.0, 2.0], [1.5, 3.0])
    elif density == "high":
        cells = ([0.5, 1.5, 3.0], [0.0, 1.5, 2.5, 3.5])
    else:
        cells = ([0.5, 1.75, 3.0], [1.0, 2.5, 3.5], [0.0, 1.5, 2.75])
    return list(cells[(bar_number - 1) % len(cells)])


def _horn_response_notes(
    chord_info: ChordInfo,
    *,
    instrument_id: str,
    low: str,
    high: str,
    harmony_index: int,
    bar_number: int,
    bar_duration: float,
) -> list[NoteEvent]:
    if bar_number % 4 not in {2, 3}:
        return []
    low_midi = note_to_midi(low)
    high_midi = note_to_midi(high)
    pcs = _horn_pcs(chord_info, harmony_index)
    if bar_duration == 3.0:
        starts = (1.0, 2.0)
    else:
        starts = (2.0, 2.5) if bar_number % 4 == 2 else (1.5, 2.5)
    notes = []
    anchor = (low_midi + high_midi) // 2
    for start, pc in zip(starts, pcs, strict=False):
        pitch, anchor = _nearest_note_in_range(
            pc,
            low_midi=low_midi,
            high_midi=high_midi,
            anchor_midi=anchor,
            prefer_sharps=chord_info.prefer_sharps,
        )
        notes.append(
            NoteEvent(
                pitch=pitch,
                start=start,
                duration=0.5,
                velocity=78,
                articulations=["accent"],
                annotations={
                    "horn_role": "response",
                    "source_chord": chord_info.symbol,
                    "instrument": instrument_id,
                },
            )
        )
    return notes


def _shout_notes(
    chord_info: ChordInfo,
    *,
    low: str,
    high: str,
    harmony_index: int,
    bar_duration: float,
) -> list[NoteEvent]:
    low_midi = note_to_midi(low)
    high_midi = note_to_midi(high)
    pcs = _horn_pcs(chord_info, harmony_index)
    notes = []
    anchor = (low_midi + high_midi) // 2
    starts = (0.0, 1.5) if bar_duration == 3.0 else (0.0, 2.0)
    for start, pc in zip(starts, pcs, strict=False):
        pitch, anchor = _nearest_note_in_range(
            pc,
            low_midi=low_midi,
            high_midi=high_midi,
            anchor_midi=anchor,
            prefer_sharps=chord_info.prefer_sharps,
        )
        notes.append(
            NoteEvent(
                pitch=pitch,
                start=start,
                duration=0.75,
                velocity=92,
                articulations=["accent"],
                dynamic="f",
                annotations={"horn_role": "shout_chorus", "source_chord": chord_info.symbol},
                voice=2,
            )
        )
    return notes


def _horn_pcs(chord_info: ChordInfo, harmony_index: int) -> tuple[int, ...]:
    pool = tuple(
        dict.fromkeys(
            [
                *chord_info.guide_tone_pcs,
                *_default_tension_pcs(chord_info),
                *chord_info.chord_tone_pcs,
            ]
        )
    )
    if not pool:
        return (chord_info.root_pc,)
    start = harmony_index % len(pool)
    return tuple(pool[(start + offset) % len(pool)] for offset in range(2))


def _with_rests(
    note_events: list[NoteEvent],
    bar_duration: float,
    *,
    voice: int,
) -> list[NoteEvent | RestEvent]:
    events = [event.model_copy(update={"voice": voice}) for event in note_events]
    intervals = sorted(
        {(event.start, event.start + event.duration) for event in events},
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
    return sorted([*events, *rests], key=lambda event: (event.start, event.duration))


def _parse_chord(parser: ChordParser, symbol: str) -> ChordInfo:
    try:
        parsed = parser.parse(symbol)
    except ValueError:
        root = _fallback_root(symbol)
        root_pc = pitch_class(root)
        return ChordInfo(
            symbol=symbol,
            root=root,
            root_pc=root_pc,
            quality="major_triad",
            chord_tone_pcs=(root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12),
            guide_tone_pcs=((root_pc + 4) % 12, (root_pc + 10) % 12),
            tension_pcs=(),
            prefer_sharps="#" in root and "b" not in root,
        )
    return ChordInfo(
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
        interval for interval in parsed.chord_tone_intervals if interval % 12 in {3, 4, 10, 11}
    ]
    return tuple((parsed.root_pc + interval) % 12 for interval in intervals)


def _pc_at_interval(chord_info: ChordInfo, intervals: tuple[int, ...]) -> int:
    wanted = {(chord_info.root_pc + interval) % 12 for interval in intervals}
    for pc in chord_info.chord_tone_pcs:
        if pc in wanted:
            return pc
    return chord_info.chord_tone_pcs[0]


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


def _fallback_root(chord_symbol: str) -> str:
    for length in (2, 1):
        root = chord_symbol[:length]
        try:
            pitch_class(root)
            return root
        except ValueError:
            continue
    return pitch_class_name(0)


def _load_learned_patterns(spec: GenerationSpec) -> dict[str, list[dict[str, Any]]]:
    inline_index = spec.constraints.get("pattern_index")
    if isinstance(inline_index, dict):
        raw_patterns = inline_index.get("patterns", [])
    else:
        pattern_index_path = spec.constraints.get("pattern_index_path")
        if not pattern_index_path:
            return {}
        path = Path(str(pattern_index_path))
        if not path.exists():
            raise FileNotFoundError(f"Pattern index not found: {path}")
        try:
            raw_patterns = json.loads(path.read_text(encoding="utf-8")).get("patterns", [])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Pattern index is not valid JSON: {path}") from exc

    min_quality = int(spec.constraints.get("pattern_min_quality", 3))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pattern in raw_patterns:
        if not isinstance(pattern, dict):
            continue
        if not pattern.get("usable_for_pattern_extraction", False):
            continue
        if int(pattern.get("quality", 0) or 0) < min_quality:
            continue
        category = str(pattern.get("category", ""))
        if not category:
            continue
        grouped.setdefault(category, []).append(pattern)

    for patterns in grouped.values():
        patterns.sort(
            key=lambda item: (
                -int(item.get("quality", 0) or 0),
                -float(item.get("weight", 0.0) or 0.0),
                str(item.get("id", "")),
            )
        )
    return grouped


def _select_learned_pattern(
    context: GenerationContext,
    category: str,
    *,
    role: str | None = None,
) -> dict[str, Any] | None:
    candidates = context.learned_patterns.get(category, [])
    if role is not None:
        candidates = [pattern for pattern in candidates if pattern.get("role") == role]
    if not candidates:
        return None

    style_matches = [
        pattern for pattern in candidates if pattern.get("style") in {context.spec.style, "unknown"}
    ]
    selected_pool = style_matches or candidates
    return selected_pool[context.spec.seed % len(selected_pool)]


def _drum_events_from_pattern(
    pattern: dict[str, Any] | None,
    *,
    bar_duration: float,
) -> list[NoteEvent]:
    if bar_duration != 4.0:
        return []
    payload = _learned_payload(pattern)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []

    allowed_pitches = {note_to_midi(pitch) for pitch in DRUM_PITCHES.values()}
    note_events: list[NoteEvent] = []
    covered_starts: set[float] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        try:
            start = _quantized_half_beat(float(raw_event["beat"]))
            midi_pitch = int(raw_event["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        if start not in DRUM_GRID or midi_pitch not in allowed_pitches:
            continue
        covered_starts.add(start)
        note_events.append(
            NoteEvent(
                pitch=midi_to_note(midi_pitch, prefer_sharps=True),
                start=start,
                duration=0.5,
                velocity=int(raw_event.get("velocity", 74) or 74),
                annotations={
                    "drum": "learned",
                    "learned_pattern_id": pattern.get("id") if pattern else None,
                },
            )
        )

    if set(DRUM_GRID) - covered_starts:
        return []
    return sorted(note_events, key=lambda event: (event.start, event.pitch))


def _walking_bass_notes_from_pattern(
    chord_info: ChordInfo,
    pattern: dict[str, Any],
    *,
    previous_midi: int,
) -> tuple[list[str], int]:
    payload = _learned_payload(pattern)
    raw_intervals = payload.get("pitch_intervals")
    if not isinstance(raw_intervals, list):
        return [], previous_midi

    intervals: list[int] = []
    for value in raw_intervals[:4]:
        try:
            intervals.append(int(value))
        except (TypeError, ValueError):
            continue
    if not intervals:
        return [], previous_midi

    fallback_intervals = [0, 3 if chord_info.quality.startswith("minor") else 4, 7, 10]
    intervals = [0, *intervals[1:4]]
    intervals.extend(fallback_intervals[len(intervals) :])

    low = note_to_midi("E1")
    high = note_to_midi("C4")
    notes: list[str] = []
    anchor = previous_midi
    for interval in intervals[:4]:
        target_pc = (chord_info.root_pc + interval) % 12
        note_name, anchor = _nearest_note_in_range(
            target_pc,
            low_midi=low,
            high_midi=high,
            anchor_midi=anchor,
            prefer_sharps=chord_info.prefer_sharps,
        )
        notes.append(note_name)
    return notes, anchor


def _rootless_voicing_from_pattern(
    chord_info: ChordInfo,
    pattern: dict[str, Any],
    *,
    anchor_midi: int,
) -> list[str]:
    payload = _learned_payload(pattern)
    raw_relative_notes = payload.get("relative_notes")
    if not isinstance(raw_relative_notes, list):
        return []

    relative_notes: list[int] = []
    for value in raw_relative_notes[:5]:
        try:
            relative_notes.append(int(value))
        except (TypeError, ValueError):
            continue
    if not relative_notes:
        return []

    base_pcs = (
        chord_info.guide_tone_pcs
        or tuple(pc for pc in chord_info.chord_tone_pcs if pc != chord_info.root_pc)
        or _default_tension_pcs(chord_info)
    )
    low = note_to_midi("C3")
    high = note_to_midi("G5")
    _, base_midi = _nearest_note_in_range(
        base_pcs[0],
        low_midi=low,
        high_midi=high,
        anchor_midi=anchor_midi,
        prefer_sharps=chord_info.prefer_sharps,
    )

    midi_notes: list[int] = []
    for interval in relative_notes:
        midi_note = base_midi + interval
        while midi_note < low:
            midi_note += 12
        while midi_note > high:
            midi_note -= 12
        if midi_note % 12 == chord_info.root_pc or midi_note in midi_notes:
            continue
        midi_notes.append(midi_note)

    if len(midi_notes) < 2:
        return []
    return [
        midi_to_note(midi_note, prefer_sharps=chord_info.prefer_sharps)
        for midi_note in sorted(midi_notes)
    ]


def _learned_payload(pattern: dict[str, Any] | None) -> dict[str, Any]:
    if not pattern:
        return {}
    payload = pattern.get("payload")
    return payload if isinstance(payload, dict) else {}


def _quantized_half_beat(value: float) -> float:
    return round(round(value * 2) / 2, 3)
