from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.config_loader import MusicConfigLoader
from arranger_core.music_theory import (
    interval_to_pitch_class,
    normalize_note_name,
    pitch_class,
    pitch_class_name,
)

ROOT_RE = re.compile(r"^(?P<root>[A-G](?:#|b)?)(?P<body>.*)$")
LETTER_ORDER = ["C", "D", "E", "F", "G", "A", "B"]
NATURAL_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class ParsedChord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    root: str
    root_pc: int
    quality: str
    bass: str | None = None
    bass_pc: int | None = None
    extensions: list[str] = Field(default_factory=list)
    alterations: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)
    chord_tone_intervals: list[int] = Field(default_factory=list)
    tension_intervals: list[int] = Field(default_factory=list)
    alteration_intervals: list[int] = Field(default_factory=list)
    chord_tone_pcs: list[int] = Field(default_factory=list)
    tension_pcs: list[int] = Field(default_factory=list)
    alteration_pcs: list[int] = Field(default_factory=list)
    chord_tones: list[str] = Field(default_factory=list)
    tension_notes: list[str] = Field(default_factory=list)
    alteration_notes: list[str] = Field(default_factory=list)
    default_scale: str | None = None

    @property
    def all_pitch_classes(self) -> list[int]:
        return _dedupe_preserve_order(
            [*self.chord_tone_pcs, *self.tension_pcs, *self.alteration_pcs]
        )


@dataclass(frozen=True)
class QualityMatch:
    quality: str
    remaining: str
    seeded_extensions: tuple[str, ...] = ()
    seeded_alterations: tuple[str, ...] = ()


@dataclass
class TokenAccumulator:
    extensions: list[str] = field(default_factory=list)
    alterations: list[str] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    omissions: list[str] = field(default_factory=list)


class ChordParser:
    def __init__(self, chord_dictionary: dict[str, Any]) -> None:
        self.chord_dictionary = chord_dictionary
        self.qualities = chord_dictionary.get("qualities", {})
        self.extensions = chord_dictionary.get("extensions", {})
        self.alterations = chord_dictionary.get("alterations", {})

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> ChordParser:
        loader = MusicConfigLoader(config_root)
        return cls(loader.load_yaml("chord_dictionary.yaml"))

    def parse(self, symbol: str) -> ParsedChord:
        original_symbol = symbol.strip()
        if not original_symbol:
            raise ValueError("Chord symbol cannot be empty")

        chord_part, bass = _split_slash_chord(original_symbol)
        match = ROOT_RE.match(chord_part)
        if not match:
            raise ValueError(f"Invalid chord symbol: {symbol!r}")

        root = normalize_note_name(match.group("root"))
        body = _normalize_half_diminished(match.group("body"))
        bass_name = normalize_note_name(bass) if bass else None
        quality_match = _match_quality(body)
        tokens = _scan_tokens(
            quality_match.remaining,
            seeded_extensions=quality_match.seeded_extensions,
            seeded_alterations=quality_match.seeded_alterations,
        )

        quality_data = self.qualities[quality_match.quality]
        chord_tone_intervals = list(quality_data.get("intervals", []))
        tension_intervals: list[int] = []
        alteration_intervals: list[int] = []

        for extension in tokens.extensions:
            self._apply_extension(
                extension,
                quality_match.quality,
                chord_tone_intervals,
                tension_intervals,
            )

        for addition in tokens.additions:
            interval = self._addition_interval(addition)
            tension_intervals.append(interval)

        for alteration in tokens.alterations:
            interval = self.alterations[alteration]
            alteration_intervals.append(interval)
            tension_intervals.append(interval)

        chord_tone_intervals = _apply_omissions(
            _dedupe_preserve_order(chord_tone_intervals),
            tokens.omissions,
        )
        tension_intervals = _dedupe_preserve_order(tension_intervals)
        alteration_intervals = _dedupe_preserve_order(alteration_intervals)

        return ParsedChord(
            symbol=original_symbol,
            root=root,
            root_pc=pitch_class(root),
            quality=quality_match.quality,
            bass=bass_name,
            bass_pc=pitch_class(bass_name) if bass_name else None,
            extensions=tokens.extensions,
            alterations=tokens.alterations,
            additions=tokens.additions,
            omissions=tokens.omissions,
            chord_tone_intervals=chord_tone_intervals,
            tension_intervals=tension_intervals,
            alteration_intervals=alteration_intervals,
            chord_tone_pcs=[
                interval_to_pitch_class(root, interval) for interval in chord_tone_intervals
            ],
            tension_pcs=[interval_to_pitch_class(root, interval) for interval in tension_intervals],
            alteration_pcs=[
                interval_to_pitch_class(root, interval) for interval in alteration_intervals
            ],
            chord_tones=[
                _spell_interval(root, interval, spelling_kind="chord")
                for interval in chord_tone_intervals
            ],
            tension_notes=[
                _spell_interval(
                    root,
                    interval,
                    spelling_kind="alteration"
                    if interval in alteration_intervals
                    else "tension",
                )
                for interval in tension_intervals
            ],
            alteration_notes=[
                _spell_interval(root, interval, spelling_kind="alteration")
                for interval in alteration_intervals
            ],
            default_scale=quality_data.get("default_scale"),
        )

    def _apply_extension(
        self,
        extension: str,
        quality: str,
        chord_tone_intervals: list[int],
        tension_intervals: list[int],
    ) -> None:
        normalized = "maj7" if extension in {"M7", "Maj7", "major7"} else extension
        if normalized not in self.extensions:
            raise ValueError(f"Unsupported chord extension: {extension}")

        interval = self.extensions[normalized]
        if normalized in {"6", "7", "maj7"}:
            chord_tone_intervals.append(interval)
            return

        if quality in {"major_triad", "minor_triad"} and normalized in {"9", "11", "13"}:
            _ensure_seventh_for_extended_chord(quality, chord_tone_intervals)

        tension_intervals.append(interval)

    def _addition_interval(self, addition: str) -> int:
        token = addition.removeprefix("add")
        if token in self.extensions:
            return self.extensions[token]
        if token in self.alterations:
            return self.alterations[token]
        raise ValueError(f"Unsupported add token: {addition}")


def _split_slash_chord(symbol: str) -> tuple[str, str | None]:
    if "/" not in symbol:
        return symbol, None

    chord_part, bass = symbol.split("/", maxsplit=1)
    if not bass:
        raise ValueError(f"Slash chord {symbol!r} is missing bass note")
    return chord_part, bass


def _normalize_half_diminished(text: str) -> str:
    return text.replace("ø", "hdim").replace("Ã¸", "hdim")


def _match_quality(body: str) -> QualityMatch:
    ordered_matches = [
        ("half_diminished", ("m7b5", "min7b5", "-7b5", "hdim7", "hdim"), (), ()),
        ("diminished_seventh", ("dim7", "o7"), (), ()),
        ("diminished", ("dim", "o"), (), ()),
        ("minor_triad", ("mMaj13", "mM13", "mMaj9", "mM9"), ("maj7", "9"), ()),
        ("minor_triad", ("mMaj7", "mM7"), ("maj7",), ()),
        ("major_triad", ("major13", "maj13", "M13"), ("maj7", "13"), ()),
        ("major_triad", ("major9", "maj9", "M9"), ("maj7", "9"), ()),
        ("major_triad", ("major7", "maj7", "M7"), ("maj7",), ()),
        ("minor_triad", ("minor", "min", "m", "-"), (), ()),
        ("augmented", ("aug", "+"), (), ()),
        ("suspended", ("13sus4", "13sus", "9sus4", "9sus", "7sus4", "7sus"), (), ()),
        ("suspended", ("sus4", "sus2", "sus"), (), ()),
    ]

    for quality, prefixes, seeded_extensions, seeded_alterations in ordered_matches:
        for prefix in prefixes:
            if body.startswith(prefix):
                return QualityMatch(
                    quality=quality,
                    remaining=body[len(prefix) :],
                    seeded_extensions=tuple(seeded_extensions),
                    seeded_alterations=tuple(seeded_alterations),
                )

    if body.startswith(("13", "11", "9", "7")):
        return QualityMatch(quality="dominant", remaining=body)

    return QualityMatch(quality="major_triad", remaining=body)


def _scan_tokens(
    remaining: str,
    *,
    seeded_extensions: tuple[str, ...] = (),
    seeded_alterations: tuple[str, ...] = (),
) -> TokenAccumulator:
    tokens = TokenAccumulator(
        extensions=list(seeded_extensions),
        alterations=list(seeded_alterations),
    )
    cursor = 0
    patterns = (
        ("alterations", ("b13", "#11", "#9", "b9", "#5", "b5", "alt")),
        ("additions", ("add#11", "addb13", "add13", "add11", "add9", "add6")),
        ("omissions", ("no3", "no5")),
        ("extensions", ("maj13", "maj9", "maj7", "13", "11", "9", "7", "6")),
        ("suspensions", ("sus4", "sus2", "sus")),
    )

    while cursor < len(remaining):
        matched = False
        for target, options in patterns:
            for option in options:
                if remaining.startswith(option, cursor):
                    _append_token(tokens, target, option)
                    cursor += len(option)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            raise ValueError(f"Could not parse chord suffix near {remaining[cursor:]!r}")

    if "alt" in tokens.alterations:
        tokens.alterations.remove("alt")
        tokens.alterations.extend(["b9", "#9", "b5", "#5", "b13"])

    tokens.extensions = _dedupe_preserve_order(tokens.extensions)
    tokens.alterations = _dedupe_preserve_order(tokens.alterations)
    tokens.additions = _dedupe_preserve_order(tokens.additions)
    tokens.omissions = _dedupe_preserve_order(tokens.omissions)
    return tokens


def _append_token(tokens: TokenAccumulator, target: str, token: str) -> None:
    if target == "extensions":
        tokens.extensions.append("maj7" if token in {"maj13", "maj9", "maj7"} else token)
        if token == "maj13":
            tokens.extensions.append("13")
        if token == "maj9":
            tokens.extensions.append("9")
        return

    if target == "alterations":
        tokens.alterations.append(token)
        return

    if target == "additions":
        tokens.additions.append(token)
        return

    if target == "omissions":
        tokens.omissions.append(token)
        return


def _ensure_seventh_for_extended_chord(quality: str, chord_tone_intervals: list[int]) -> None:
    if 10 in chord_tone_intervals or 11 in chord_tone_intervals:
        return

    seventh = 11 if quality == "major_triad" else 10
    chord_tone_intervals.append(seventh)


def _apply_omissions(intervals: list[int], omissions: list[str]) -> list[int]:
    omitted = set()
    if "no3" in omissions:
        omitted.update({3, 4})
    if "no5" in omissions:
        omitted.update({6, 7, 8})
    return [interval for interval in intervals if interval % 12 not in omitted]


def _dedupe_preserve_order(values: list[int] | list[str]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        key = value % 12 if isinstance(value, int) else value
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _spell_interval(root: str, interval: int, *, spelling_kind: str) -> str:
    degree_steps = _degree_steps_for_interval(interval, spelling_kind=spelling_kind)
    root_letter = root[0]
    root_index = LETTER_ORDER.index(root_letter)
    target_letter = LETTER_ORDER[(root_index + degree_steps) % len(LETTER_ORDER)]
    target_pc = interval_to_pitch_class(root, interval)
    natural_pc = NATURAL_PITCH_CLASSES[target_letter]
    accidental_delta = _nearest_accidental_delta(target_pc - natural_pc)
    if abs(accidental_delta) == 2:
        return pitch_class_name(target_pc, prefer_sharps="#" in root and "b" not in root)

    accidentals = {
        -2: "bb",
        -1: "b",
        0: "",
        1: "#",
        2: "##",
    }
    accidental = accidentals.get(accidental_delta)
    if accidental is None:
        return target_letter
    return f"{target_letter}{accidental}"


def _degree_steps_for_interval(interval: int, *, spelling_kind: str) -> int:
    interval_mod = interval % 12
    if spelling_kind == "chord":
        if interval_mod == 0:
            return 0
        if interval_mod in {3, 4}:
            return 2
        if interval_mod in {6, 7, 8}:
            return 4
        if interval_mod == 9:
            return 5
        if interval_mod in {10, 11}:
            return 6

    if interval in {13, 14, 15}:
        return 1
    if interval in {17, 18}:
        return 3
    if interval in {20, 21}:
        return 5
    if spelling_kind == "alteration" and interval in {6, 8}:
        return 4

    return _fallback_degree_steps(interval_mod)


def _fallback_degree_steps(interval_mod: int) -> int:
    if interval_mod in {1, 2}:
        return 1
    if interval_mod in {3, 4}:
        return 2
    if interval_mod in {5, 6}:
        return 3
    if interval_mod in {7, 8}:
        return 4
    if interval_mod == 9:
        return 5
    if interval_mod in {10, 11}:
        return 6
    return 0


def _nearest_accidental_delta(delta: int) -> int:
    while delta > 6:
        delta -= 12
    while delta < -6:
        delta += 12
    if delta > 2:
        delta -= 12
    if delta < -2:
        delta += 12
    return delta
