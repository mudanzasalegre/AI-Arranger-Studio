from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

from arranger_core.catalogs import ProgressionLibrary
from arranger_core.music_theory import pitch_class, pitch_class_name
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    KeyMark,
    MeterMark,
    RestEvent,
    Section,
    TempoMark,
    Track,
    meter_to_quarter_beats,
)

FORM_ALIASES = {
    "minor_blues": "minor_blues_12",
    "minor_blues_12": "minor_blues_12",
    "jazz_blues": "jazz_blues_12",
    "jazz_blues_12": "jazz_blues_12",
    "blues_12": "jazz_blues_12",
    "aaba": "aaba_32",
    "aaba_32": "aaba_32",
    "ballad": "ballad_aaba_32",
    "ballad_aaba": "ballad_aaba_32",
    "ballad_aaba_32": "ballad_aaba_32",
    "rhythm_changes_like": "rhythm_changes_like_32",
    "rhythm_changes_like_32": "rhythm_changes_like_32",
    "modal_vamp": "modal_vamp",
    "modal_vamp_8": "modal_vamp",
    "modal_vamp_16": "modal_vamp_16",
    "bossa": "bossa_32",
    "bossa_32": "bossa_32",
    "latin_bossa_32": "bossa_32",
    "jazz_waltz": "jazz_waltz_32",
    "jazz_waltz_32": "jazz_waltz_32",
    "waltz": "jazz_waltz_32",
    "sixteen_bar": "sixteen_bar",
    "sixteen_bar_tune": "sixteen_bar",
    "tune_16": "sixteen_bar",
}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
ROMAN_TO_DEGREE = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
}
DEGREE_RE = re.compile(
    r"^(?P<accidentals>[b#]*)"
    r"(?P<roman>VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)"
    r"(?P<suffix>.*)$"
)


@dataclass(frozen=True)
class KeyContext:
    root: str
    mode: Literal["major", "minor"]

    @property
    def prefer_sharps(self) -> bool:
        return "#" in self.root and "b" not in self.root


@dataclass
class HarmonyPlan:
    form_id: str
    key: str
    meter: str
    tempo: int
    sections: list[Section]
    chord_grid: list[ChordSymbol]
    applied_variations: list[str] = field(default_factory=list)

    @property
    def bar_count(self) -> int:
        return max((section.end_bar for section in self.sections), default=0)


class HarmonyFormEngine:
    def __init__(self, progression_library: ProgressionLibrary | None = None) -> None:
        self.progressions = progression_library or ProgressionLibrary.load_default()

    def generate(self, spec: GenerationSpec) -> HarmonyPlan:
        rng = random.Random(spec.seed)
        form_id = _canonical_form(spec)
        key_context = parse_key(spec.key)
        degree_bars, sections = self._degree_form(form_id, spec)
        chord_grid = self._materialize_chord_grid(
            degree_bars=degree_bars,
            key_context=key_context,
            meter=spec.meter,
        )
        variations = self._apply_variations(
            chord_grid=chord_grid,
            key_context=key_context,
            complexity=spec.complexity,
            rng=rng,
            form_id=form_id,
            meter=spec.meter,
        )

        return HarmonyPlan(
            form_id=form_id,
            key=spec.key,
            meter=spec.meter,
            tempo=spec.tempo,
            sections=sections,
            chord_grid=chord_grid,
            applied_variations=variations,
        )

    def create_project(
        self,
        spec: GenerationSpec,
        *,
        project_id: str | None = None,
        include_lead_sheet_track: bool = True,
    ) -> ArrangementProject:
        plan = self.generate(spec)
        tracks = [_lead_sheet_track(plan)] if include_lead_sheet_track else []
        return ArrangementProject(
            project_id=project_id or f"harmony-{spec.seed}-{plan.form_id}",
            metadata={
                "title": f"{spec.style} {plan.form_id}",
                "style": spec.style,
                "seed": spec.seed,
                "harmony_engine": "rule_based_v0",
                "applied_variations": plan.applied_variations,
            },
            generation_spec=spec,
            tempo_map=[TempoMark(bar=1, bpm=spec.tempo)],
            key_map=[KeyMark(bar=1, key=spec.key)],
            meter_map=[MeterMark(bar=1, meter=spec.meter)],
            form=plan.sections,
            chord_grid=plan.chord_grid,
            tracks=tracks,
        )

    def _degree_form(
        self,
        form_id: str,
        spec: GenerationSpec,
    ) -> tuple[list[list[str]], list[Section]]:
        if form_id == "minor_blues_12":
            return _degrees_from_template(self.progressions.get("minor_blues_12").degrees), [
                Section(name="Minor Blues", start_bar=1, end_bar=12, label="Blues")
            ]
        if form_id == "jazz_blues_12":
            return _degrees_from_template(self.progressions.get("jazz_blues_12").degrees), [
                Section(name="Jazz Blues", start_bar=1, end_bar=12, label="Blues")
            ]
        if form_id == "rhythm_changes_like_32":
            return self._rhythm_changes_like_form()
        if form_id == "aaba_32":
            return self._aaba_form(ballad=False)
        if form_id == "ballad_aaba_32":
            return self._aaba_form(ballad=True)
        if form_id in {"modal_vamp", "modal_vamp_16"}:
            target_bars = 16 if form_id == "modal_vamp_16" else (spec.duration_bars or 8)
            return self._modal_vamp_form(target_bars=target_bars)
        if form_id == "bossa_32":
            return self._bossa_form()
        if form_id == "jazz_waltz_32":
            return self._jazz_waltz_form()
        if form_id == "sixteen_bar":
            return self._sixteen_bar_form()
        raise ValueError(f"Unsupported form: {form_id}")

    def _rhythm_changes_like_form(self) -> tuple[list[list[str]], list[Section]]:
        a = _degrees_from_template(self.progressions.get("rhythm_changes_like_a").degrees)
        bridge = _degrees_from_template(self.progressions.get("rhythm_changes_like_bridge").degrees)
        degrees = [*a, *a, *bridge, *a]
        sections = [
            Section(name="A1", start_bar=1, end_bar=8),
            Section(name="A2", start_bar=9, end_bar=16),
            Section(name="Bridge", start_bar=17, end_bar=24, label="B"),
            Section(name="A3", start_bar=25, end_bar=32),
        ]
        return degrees, sections

    def _aaba_form(self, *, ballad: bool) -> tuple[list[list[str]], list[Section]]:
        if ballad:
            a = [
                ["Imaj7"],
                ["vi-7", "II7"],
                ["ii-7"],
                ["V7b9"],
                ["iii-7", "VI7alt"],
                ["ii-7", "V7"],
                ["Imaj7", "IVmaj7"],
                ["iii-7", "VI7alt"],
            ]
            bridge = [
                ["iv-7"],
                ["bVII7"],
                ["Imaj7"],
                ["VI7alt"],
                ["ii-7"],
                ["V7sus4"],
                ["Imaj7"],
                ["V7alt"],
            ]
        else:
            a = [
                ["Imaj9", "VI7alt"],
                ["ii-9", "V13"],
                ["iii-9", "VI7alt"],
                ["ii-9", "V13"],
                ["Imaj9"],
                ["IV13", "#IVdim7"],
                ["Imaj9", "VI7alt"],
                ["ii-9", "V13"],
            ]
            bridge = [
                ["III13"],
                ["III7alt"],
                ["VI13"],
                ["VI7alt"],
                ["II13"],
                ["II7alt"],
                ["V13"],
                ["V7alt"],
            ]
        degrees = [*a, *a, *bridge, *a]
        sections = [
            Section(name="A1", start_bar=1, end_bar=8),
            Section(name="A2", start_bar=9, end_bar=16),
            Section(name="Bridge", start_bar=17, end_bar=24, label="B"),
            Section(name="A3", start_bar=25, end_bar=32),
        ]
        return degrees, sections

    def _modal_vamp_form(self, *, target_bars: int) -> tuple[list[list[str]], list[Section]]:
        base = _degrees_from_template(self.progressions.get("modal_dorian_vamp").degrees)
        degrees = [base[index % len(base)] for index in range(target_bars)]
        sections = [Section(name="Modal Vamp", start_bar=1, end_bar=target_bars, label="Vamp")]
        return degrees, sections

    def _bossa_form(self) -> tuple[list[list[str]], list[Section]]:
        a = [
            ["Imaj9"],
            ["vi-7", "II7"],
            ["ii-9"],
            ["V13"],
            ["iii-7", "VI7alt"],
            ["ii-9", "V13"],
            ["Imaj9"],
            ["ii-7", "V13"],
        ]
        b = [
            ["IVmaj9"],
            ["iv-7", "bVII13"],
            ["iii-7", "VI7alt"],
            ["ii-9", "V13"],
            ["Imaj9"],
            ["VI7alt"],
            ["ii-9"],
            ["V13"],
        ]
        degrees = [*a, *a, *b, *a]
        sections = [
            Section(name="A1", start_bar=1, end_bar=8),
            Section(name="A2", start_bar=9, end_bar=16),
            Section(name="Bridge", start_bar=17, end_bar=24, label="B"),
            Section(name="A3", start_bar=25, end_bar=32),
        ]
        return degrees, sections

    def _jazz_waltz_form(self) -> tuple[list[list[str]], list[Section]]:
        a = [
            ["i-9"],
            ["iv-9"],
            ["bVII13"],
            ["IIImaj9"],
            ["VImaj9"],
            ["iim7b5", "V7alt"],
            ["i-9"],
            ["V7alt"],
        ]
        b = [
            ["IVm9"],
            ["bVII13"],
            ["IIImaj9"],
            ["VI7alt"],
            ["iim7b5"],
            ["V7alt"],
            ["i-9", "VI7alt"],
            ["iim7b5", "V7alt"],
        ]
        degrees = [*a, *a, *b, *a]
        sections = [
            Section(name="A1", start_bar=1, end_bar=8),
            Section(name="A2", start_bar=9, end_bar=16),
            Section(name="Bridge", start_bar=17, end_bar=24, label="B"),
            Section(name="A3", start_bar=25, end_bar=32),
        ]
        return degrees, sections

    def _sixteen_bar_form(self) -> tuple[list[list[str]], list[Section]]:
        degrees = [
            ["i-9"],
            ["iv-9"],
            ["bVII13"],
            ["IIImaj9"],
            ["VImaj7"],
            ["iim7b5"],
            ["V7alt"],
            ["i-9"],
            ["iv-9"],
            ["bVII13"],
            ["IIImaj9"],
            ["VI7alt"],
            ["iim7b5"],
            ["V7alt"],
            ["i-9", "VI7alt"],
            ["iim7b5", "V7alt"],
        ]
        sections = [
            Section(name="A", start_bar=1, end_bar=8),
            Section(name="A Prime", start_bar=9, end_bar=16, label="A'"),
        ]
        return degrees, sections

    def _materialize_chord_grid(
        self,
        *,
        degree_bars: list[list[str]],
        key_context: KeyContext,
        meter: str,
    ) -> list[ChordSymbol]:
        bar_duration = meter_to_quarter_beats(meter)
        chord_grid: list[ChordSymbol] = []
        for bar_index, degree_tokens in enumerate(degree_bars, start=1):
            duration = bar_duration / len(degree_tokens)
            for token_index, degree_token in enumerate(degree_tokens):
                symbol = degree_to_chord_symbol(degree_token, key_context)
                chord_grid.append(
                    ChordSymbol(
                        symbol=symbol,
                        bar=bar_index,
                        beat=1 + token_index * duration,
                        duration=duration,
                        metadata={"degree": degree_token},
                    )
                )
        return chord_grid

    def _apply_variations(
        self,
        *,
        chord_grid: list[ChordSymbol],
        key_context: KeyContext,
        complexity: float,
        rng: random.Random,
        form_id: str,
        meter: str,
    ) -> list[str]:
        variations: list[str] = []
        if complexity < 0.35:
            return variations

        if complexity >= 0.45:
            variations.extend(_apply_turnaround(chord_grid, key_context, meter))
        if complexity >= 0.55:
            variations.extend(
                _apply_passing_diminished(chord_grid, key_context, rng, form_id, meter)
            )
        if complexity >= 0.65:
            variations.extend(_apply_secondary_dominants(chord_grid, rng, meter))
        if complexity >= 0.72:
            variations.extend(_apply_tritone_substitutions(chord_grid, rng))
        if complexity >= 0.80:
            variations.extend(_apply_backdoor_cadence(chord_grid, key_context, meter))
        return variations


def generate_harmony_plan(spec: GenerationSpec) -> HarmonyPlan:
    return HarmonyFormEngine().generate(spec)


def generate_harmony_project(
    spec: GenerationSpec,
    *,
    project_id: str | None = None,
    include_lead_sheet_track: bool = True,
) -> ArrangementProject:
    return HarmonyFormEngine().create_project(
        spec,
        project_id=project_id,
        include_lead_sheet_track=include_lead_sheet_track,
    )


def degree_to_chord_symbol(degree_token: str, key_context: KeyContext) -> str:
    token = _normalize_degree_token(degree_token)
    match = DEGREE_RE.match(token)
    if not match:
        if re.match(r"^[A-G](?:#|b)?", token):
            return token
        raise ValueError(f"Invalid degree token: {degree_token!r}")

    accidentals = match.group("accidentals")
    roman = match.group("roman")
    suffix = match.group("suffix")
    degree = ROMAN_TO_DEGREE[roman.upper()]
    root_pc = _degree_pitch_class(
        key_context,
        degree,
        use_major_reference=bool(accidentals),
    )
    for accidental in accidentals:
        root_pc += 1 if accidental == "#" else -1
    root = pitch_class_name(root_pc, prefer_sharps=key_context.prefer_sharps)
    return f"{root}{_degree_suffix_to_chord_suffix(suffix, roman)}"


def parse_key(key: str) -> KeyContext:
    parts = key.strip().split()
    if not parts:
        raise ValueError("Key cannot be empty")
    root = parts[0]
    mode = "minor" if any(part.lower() == "minor" for part in parts[1:]) else "major"
    pitch_class(root)
    return KeyContext(root=root, mode=mode)


def _canonical_form(spec: GenerationSpec) -> str:
    requested = FORM_ALIASES.get(spec.form, spec.form)
    if spec.duration_bars == 16 and requested in {"modal_vamp", "minor_blues_12"}:
        return "sixteen_bar" if requested != "modal_vamp" else "modal_vamp_16"
    return requested


def _degrees_from_template(template_degrees: list[str]) -> list[list[str]]:
    return [_split_bar_degrees(bar) for bar in template_degrees]


def _split_bar_degrees(bar: str) -> list[str]:
    return [token for token in bar.split() if token]


def _normalize_degree_token(token: str) -> str:
    return (
        token.replace("\u00f87", "m7b5")
        .replace("\u00f8", "m7b5")
        .replace("\u00c3\u00b87", "m7b5")
        .replace("\u00c3\u00b8", "m7b5")
        .strip()
    )


def _degree_pitch_class(
    key_context: KeyContext,
    degree: int,
    *,
    use_major_reference: bool = False,
) -> int:
    scale = MINOR_SCALE if key_context.mode == "minor" and not use_major_reference else MAJOR_SCALE
    return pitch_class(key_context.root) + scale[degree - 1]


def _degree_suffix_to_chord_suffix(suffix: str, roman: str) -> str:
    if suffix.startswith("m7b5"):
        return suffix
    if suffix.startswith("-"):
        return "m" + suffix[1:]
    if suffix == "":
        return "m" if roman.islower() else ""
    return suffix


def _lead_sheet_track(plan: HarmonyPlan) -> Track:
    bar_duration = meter_to_quarter_beats(plan.meter)
    return Track(
        id="lead_sheet",
        instrument="piano",
        role="lead_sheet",
        name="Lead Sheet",
        bars=[
            Bar(number=bar_number, events=[RestEvent(start=0, duration=bar_duration)])
            for bar_number in range(1, plan.bar_count + 1)
        ],
    )


def _chords_in_bar(chord_grid: list[ChordSymbol], bar_number: int) -> list[ChordSymbol]:
    return [chord for chord in chord_grid if chord.bar == bar_number]


def _replace_bar_chords(
    chord_grid: list[ChordSymbol],
    *,
    bar_number: int,
    symbols: list[str],
    meter: str,
    variation: str,
) -> None:
    existing = _chords_in_bar(chord_grid, bar_number)
    if not existing:
        return
    bar_duration = meter_to_quarter_beats(meter)
    first_index = chord_grid.index(existing[0])
    for chord in existing:
        chord_grid.remove(chord)
    duration = bar_duration / len(symbols)
    replacements = [
        ChordSymbol(
            symbol=symbol,
            bar=bar_number,
            beat=1 + index * duration,
            duration=duration,
            metadata={"variation": variation},
        )
        for index, symbol in enumerate(symbols)
    ]
    for offset, replacement in enumerate(replacements):
        chord_grid.insert(first_index + offset, replacement)


def _apply_turnaround(
    chord_grid: list[ChordSymbol],
    key_context: KeyContext,
    meter: str,
) -> list[str]:
    last_bar = max(chord.bar or 0 for chord in chord_grid)
    if last_bar == 0:
        return []
    symbols = (
        ["i-7", "VI7alt", "iim7b5", "V7alt"]
        if key_context.mode == "minor"
        else ["Imaj7", "VI7alt", "ii-7", "V13"]
    )
    _replace_bar_chords(
        chord_grid,
        bar_number=last_bar,
        symbols=[degree_to_chord_symbol(symbol, key_context) for symbol in symbols],
        meter=meter,
        variation="turnaround",
    )
    return ["turnaround"]


def _apply_passing_diminished(
    chord_grid: list[ChordSymbol],
    key_context: KeyContext,
    rng: random.Random,
    form_id: str,
    meter: str,
) -> list[str]:
    candidate_bars = [6, 14, 30] if form_id.endswith("32") else [6]
    applied = []
    for bar_number in candidate_bars:
        if not _chords_in_bar(chord_grid, bar_number) or rng.random() > 0.85:
            continue
        symbol = degree_to_chord_symbol("#IVdim7", key_context)
        _replace_bar_chords(
            chord_grid,
            bar_number=bar_number,
            symbols=[symbol],
            meter=meter,
            variation="passing_diminished",
        )
        applied.append("passing_diminished")
        break
    return applied


def _apply_secondary_dominants(
    chord_grid: list[ChordSymbol],
    rng: random.Random,
    meter: str,
) -> list[str]:
    applied = []
    bars = sorted({chord.bar for chord in chord_grid if chord.bar is not None})
    for bar_number in bars[1:-1]:
        if rng.random() > 0.16:
            continue
        current = _chords_in_bar(chord_grid, bar_number)
        following = _chords_in_bar(chord_grid, bar_number + 1)
        if not current or not following:
            continue
        dominant = _dominant_of(following[0].symbol)
        if dominant:
            _replace_bar_chords(
                chord_grid,
                bar_number=bar_number,
                symbols=[current[0].symbol, dominant],
                meter=meter,
                variation="secondary_dominant",
            )
            applied.append("secondary_dominant")
            if len(applied) >= 2:
                break
    return applied


def _apply_tritone_substitutions(
    chord_grid: list[ChordSymbol],
    rng: random.Random,
) -> list[str]:
    applied = []
    for chord in chord_grid:
        if not _is_dominant_chord(chord.symbol):
            continue
        if rng.random() > 0.22:
            continue
        root = _chord_root(chord.symbol)
        if root is None:
            continue
        sub_root = pitch_class_name(pitch_class(root) + 6, prefer_sharps="#" in root)
        chord.symbol = f"{sub_root}7#11"
        chord.metadata["variation"] = "tritone_substitution"
        applied.append("tritone_substitution")
        if len(applied) >= 2:
            break
    return applied


def _apply_backdoor_cadence(
    chord_grid: list[ChordSymbol],
    key_context: KeyContext,
    meter: str,
) -> list[str]:
    last_bar = max(chord.bar or 0 for chord in chord_grid)
    target_bar = max(1, last_bar - 1)
    symbols = ["iv-7", "bVII7"]
    _replace_bar_chords(
        chord_grid,
        bar_number=target_bar,
        symbols=[degree_to_chord_symbol(symbol, key_context) for symbol in symbols],
        meter=meter,
        variation="backdoor_cadence",
    )
    return ["backdoor_cadence"]


def _dominant_of(chord_symbol: str) -> str | None:
    root = _chord_root(chord_symbol)
    if root is None:
        return None
    dominant_root = pitch_class_name(pitch_class(root) + 7, prefer_sharps="#" in root)
    return f"{dominant_root}7alt"


def _chord_root(chord_symbol: str) -> str | None:
    match = re.match(r"^(?P<root>[A-G](?:#|b)?)", chord_symbol)
    return match.group("root") if match else None


def _is_dominant_chord(chord_symbol: str) -> bool:
    match = re.match(r"^[A-G](?:#|b)?(?P<suffix>.*)$", chord_symbol)
    if not match:
        return False
    suffix = match.group("suffix")
    return "7" in suffix and not suffix.startswith(("m", "maj", "dim"))
