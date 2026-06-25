from __future__ import annotations

import re

PITCH_CLASS_BY_NAME = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NATURAL_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
NOTE_RE = re.compile(r"^(?P<name>[A-G](?:#|b)?)(?P<octave>-?\d+)?$")


def normalize_note_name(note_name: str) -> str:
    match = NOTE_RE.match(note_name.strip())
    if not match:
        raise ValueError(f"Invalid note name: {note_name!r}")

    name = match.group("name")
    if name not in PITCH_CLASS_BY_NAME:
        raise ValueError(f"Unsupported note name: {note_name!r}")
    return name


def pitch_class(note_name: str) -> int:
    return PITCH_CLASS_BY_NAME[normalize_note_name(note_name)]


def pitch_class_name(pitch_class_value: int, *, prefer_sharps: bool = False) -> str:
    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES
    return names[pitch_class_value % 12]


def parse_note(note: str) -> tuple[str, int | None]:
    match = NOTE_RE.match(note.strip())
    if not match:
        raise ValueError(f"Invalid note: {note!r}")
    return normalize_note_name(match.group("name")), _parse_optional_int(match.group("octave"))


def note_to_midi(note: str) -> int:
    name, octave = parse_note(note)
    if octave is None:
        raise ValueError(f"Note {note!r} has no octave")
    return (octave + 1) * 12 + pitch_class(name)


def midi_to_note(midi_note: int, *, prefer_sharps: bool = False) -> str:
    if midi_note < 0 or midi_note > 127:
        raise ValueError(f"MIDI note must be in 0..127, got {midi_note}")

    octave = (midi_note // 12) - 1
    name = pitch_class_name(midi_note % 12, prefer_sharps=prefer_sharps)
    return f"{name}{octave}"


def transpose_note(note: str, semitones: int, *, prefer_sharps: bool = False) -> str:
    return midi_to_note(note_to_midi(note) + semitones, prefer_sharps=prefer_sharps)


def transpose_pitch_class_name(
    root: str,
    semitones: int,
    *,
    prefer_sharps: bool | None = None,
) -> str:
    if prefer_sharps is None:
        prefer_sharps = "#" in root and "b" not in root
    return pitch_class_name(pitch_class(root) + semitones, prefer_sharps=prefer_sharps)


def interval_to_pitch_class(root: str, interval: int) -> int:
    return (pitch_class(root) + interval) % 12


def interval_to_note_name(
    root: str,
    interval: int,
    *,
    prefer_sharps: bool | None = None,
) -> str:
    if prefer_sharps is None:
        prefer_sharps = "#" in root and "b" not in root
    return pitch_class_name(interval_to_pitch_class(root, interval), prefer_sharps=prefer_sharps)


def _parse_optional_int(value: str | None) -> int | None:
    return int(value) if value is not None else None
