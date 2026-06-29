from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import mido
from music21 import converter, harmony

from dataset_tools.models import (
    CommercialTrainingUse,
    DatasetFileProfile,
    DatasetManifestEntry,
    DatasetProfileReport,
    DatasetRole,
    LicenseConfidence,
    RoleClassification,
    TrackProfile,
)

PERMISSIVE_LICENSES = {
    "cc0",
    "cc0-1.0",
    "public domain",
    "public-domain",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
}
REVIEW_LICENSES = {
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "creative commons",
}
BLOCKED_LICENSES = {"", "unknown", "proprietary", "all rights reserved", "private"}
ROLE_ORDER: tuple[DatasetRole, ...] = (
    "melody",
    "bass",
    "drums",
    "comping",
    "horns",
    "pad",
    "solo",
    "harmony",
    "unknown",
)
GM_PROGRAM_HINTS = {
    0: "piano",
    1: "piano",
    4: "electric_piano",
    5: "electric_piano",
    16: "organ",
    24: "guitar",
    25: "guitar",
    32: "bass",
    33: "bass",
    34: "bass",
    35: "bass",
    40: "strings",
    41: "strings",
    48: "strings",
    52: "choir",
    56: "trumpet",
    57: "trombone",
    60: "horn",
    64: "sax",
    65: "sax",
    66: "sax",
    67: "sax",
    68: "oboe",
    71: "clarinet",
    88: "pad",
    89: "pad",
    90: "pad",
}


def profile_dataset_file(
    path: str | Path,
    *,
    file_id: str | None = None,
    normalized_path: str | Path | None = None,
    metadata: DatasetManifestEntry | dict[str, Any] | None = None,
    file_hash: str | None = None,
    duplicate_of: str | None = None,
) -> DatasetFileProfile:
    source_path = Path(path)
    metadata_map = _metadata_map(metadata)
    resolved_hash = file_hash or _sha256_file(source_path)
    resolved_file_id = file_id or f"file_{resolved_hash[:12]}"
    suffix = source_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        return _profile_midi_file(
            source_path,
            file_id=resolved_file_id,
            normalized_path=normalized_path,
            metadata=metadata_map,
            file_hash=resolved_hash,
            duplicate_of=duplicate_of,
        )
    if suffix in {".musicxml", ".xml"}:
        return _profile_musicxml_file(
            source_path,
            file_id=resolved_file_id,
            normalized_path=normalized_path,
            metadata=metadata_map,
            file_hash=resolved_hash,
            duplicate_of=duplicate_of,
        )
    return _base_profile(
        source_path,
        file_id=resolved_file_id,
        normalized_path=normalized_path,
        metadata=metadata_map,
        file_hash=resolved_hash,
        duplicate_of=duplicate_of,
        file_format=suffix.removeprefix(".") or "unknown",
        track_profiles=[],
        file_features={"format": suffix.removeprefix(".") or "unknown"},
    )


def build_dataset_profile_report(
    profiles: list[DatasetFileProfile],
) -> DatasetProfileReport:
    role_counts = Counter(
        track.classification.role
        for profile in profiles
        for track in profile.track_profiles
    )
    note_count = sum(int(profile.file_features.get("note_count", 0) or 0) for profile in profiles)
    track_count = sum(len(profile.track_profiles) for profile in profiles)
    return DatasetProfileReport(
        files=profiles,
        role_counts=dict(sorted(role_counts.items())),
        file_count=len(profiles),
        track_count=track_count,
        note_count=note_count,
    )


def write_role_manifest(
    report: DatasetProfileReport,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": report.schema_version,
        "file_count": report.file_count,
        "track_count": report.track_count,
        "note_count": report.note_count,
        "role_counts": report.role_counts,
        "files": [
            {
                "file_id": profile.file_id,
                "path": profile.original_path,
                "hash": profile.hash,
                "format": profile.format,
                "roles": profile.role_coverage,
                "contains_melody": profile.contains_melody,
                "contains_chords": profile.contains_chords,
                "contains_arrangement": profile.contains_arrangement,
                "commercial_training": profile.commercial_training,
                "local_learning_only": profile.local_learning_only,
                "pattern_sensitivity": profile.pattern_sensitivity,
                "tracks": [
                    {
                        "track_index": track.track_index,
                        "name": track.name,
                        "instrument_guess": track.instrument_guess,
                        "role": track.classification.role,
                        "confidence": track.classification.confidence,
                        "reasons": track.classification.reasons,
                    }
                    for track in profile.track_profiles
                ],
            }
            for profile in report.files
        ],
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def classify_midi_track_role(
    name: str,
    channels: list[int],
    notes: list[dict[str, Any]],
    *,
    programs: list[int] | None = None,
) -> RoleClassification:
    features = midi_note_features(notes, channels)
    instrument = instrument_guess(name, channels, programs or [])
    return classify_role(
        name=name,
        channels=channels,
        programs=programs or [],
        instrument_guess=instrument,
        features=features,
    )


def classify_role(
    *,
    name: str,
    channels: list[int],
    programs: list[int],
    instrument_guess: str,
    features: dict[str, Any],
) -> RoleClassification:
    normalized_name = name.lower()
    note_count = int(features.get("note_count", 0) or 0)
    if note_count == 0:
        return RoleClassification(
            role="unknown",
            confidence=0.72,
            reasons=["no pitched notes were detected"],
            alternatives={"unknown": 0.72},
        )

    scores: dict[DatasetRole, float] = defaultdict(float)
    reasons: dict[DatasetRole, list[str]] = defaultdict(list)

    def add(role: DatasetRole, value: float, reason: str) -> None:
        scores[role] = max(scores[role], value)
        reasons[role].append(reason)

    drum_ratio = float(features.get("drum_channel_ratio", 0.0) or 0.0)
    mean_pitch = float(features.get("mean_pitch", 0.0) or 0.0)
    notes_per_bar = float(features.get("notes_per_bar", 0.0) or 0.0)
    chord_onset_ratio = float(features.get("chord_onset_ratio", 0.0) or 0.0)
    polyphonic_ratio = float(features.get("polyphonic_note_ratio", 0.0) or 0.0)
    long_note_ratio = float(features.get("long_note_ratio", 0.0) or 0.0)
    pitch_span = float(features.get("pitch_span", 0.0) or 0.0)

    if drum_ratio >= 0.5 or _has_any(normalized_name, ("drum", "perc", "kit")):
        add("drums", 0.98, "drum channel or percussion name")
    if _has_any(normalized_name, ("bass", "contrabass")) or instrument_guess == "bass":
        add("bass", 0.94, "bass instrument hint")
    if mean_pitch < 48 and polyphonic_ratio < 0.25 and notes_per_bar >= 1.5:
        add("bass", 0.84, "low monophonic register")
    if _has_any(normalized_name, ("piano", "comp", "keys", "guitar", "organ")):
        add("comping", 0.91, "comping-capable instrument name")
    if instrument_guess in {"piano", "electric_piano", "guitar", "organ"}:
        add("comping", 0.84, "comping-capable program")
    if chord_onset_ratio >= 0.3 or polyphonic_ratio >= 0.45:
        add("comping", 0.86, "polyphonic chord onsets")
    if _has_any(normalized_name, ("pad", "strings", "sustain", "synth")):
        add("pad", 0.9, "pad or sustained instrument name")
    if instrument_guess in {"pad", "strings", "choir"} and long_note_ratio >= 0.35:
        add("pad", 0.82, "long sustained notes")
    if _has_any(normalized_name, ("solo", "improv")):
        add("solo", 0.9, "solo/improvisation name hint")
    if _has_any(normalized_name, ("trumpet", "trombone", "horn", "brass")):
        add("horns", 0.9, "horn section instrument name")
    if instrument_guess in {"trumpet", "trombone", "horn"}:
        add("horns", 0.84, "brass program")
    if _has_any(normalized_name, ("lead", "melody", "sax", "flute", "clarinet", "vocal")):
        add("melody", 0.9, "lead or melody instrument name")
    if instrument_guess in {"sax", "oboe", "clarinet"}:
        add("melody", 0.82, "single-line woodwind program")
    if mean_pitch >= 56 and polyphonic_ratio < 0.25 and pitch_span >= 5:
        add("melody", 0.74, "mid/high monophonic contour")
    if not scores:
        add("unknown", 0.45, "no role heuristic crossed threshold")

    if scores.get("drums", 0.0) >= 0.95:
        best_role: DatasetRole = "drums"
    else:
        best_role = max(ROLE_ORDER, key=lambda role: scores.get(role, 0.0))
    confidence = round(min(0.99, max(0.0, scores.get(best_role, 0.0))), 3)
    alternatives = {
        role: round(score, 3)
        for role, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    }
    return RoleClassification(
        role=best_role,
        confidence=confidence,
        reasons=reasons[best_role][:4],
        alternatives=alternatives,
    )


def instrument_guess(
    name: str,
    channels: list[int],
    programs: list[int],
) -> str:
    normalized = name.lower()
    if 9 in channels or _has_any(normalized, ("drum", "perc", "kit")):
        return "drums"
    name_hints = (
        ("double_bass", ("double bass", "upright bass", "contrabass")),
        ("bass", ("bass",)),
        ("piano", ("piano",)),
        ("electric_piano", ("rhodes", "wurli", "electric piano")),
        ("guitar", ("guitar",)),
        ("organ", ("organ",)),
        ("trumpet", ("trumpet",)),
        ("trombone", ("trombone",)),
        ("horn", ("horn", "brass")),
        ("sax", ("sax", "saxophone")),
        ("clarinet", ("clarinet",)),
        ("flute", ("flute",)),
        ("strings", ("strings", "violin", "viola", "cello")),
        ("pad", ("pad", "synth")),
    )
    for instrument, tokens in name_hints:
        if _has_any(normalized, tokens):
            return instrument
    for program in programs:
        if program in GM_PROGRAM_HINTS:
            return GM_PROGRAM_HINTS[program]
    return "unknown"


def midi_note_features(
    notes: list[dict[str, Any]],
    channels: list[int],
) -> dict[str, Any]:
    if not notes:
        return {
            "note_count": 0,
            "duration_beats": 0.0,
            "active_bars": 0,
            "channels": sorted(set(channels)),
        }
    pitches = [int(note["note"]) for note in notes]
    starts = [round(float(note["start"]), 3) for note in notes]
    durations = [float(note.get("duration", 0.0) or 0.0) for note in notes]
    velocities = [int(note.get("velocity", 0) or 0) for note in notes]
    end_beat = max(start + duration for start, duration in zip(starts, durations, strict=False))
    active_bars = max(1, math.ceil(end_beat / 4.0))
    onset_counts = Counter(starts)
    polyphonic_notes = sum(count for count in onset_counts.values() if count > 1)
    chord_onsets = sum(1 for count in onset_counts.values() if count > 1)
    drum_notes = sum(1 for note in notes if int(note.get("channel", -1)) == 9)
    beat_positions = [round(start % 4.0, 3) for start in starts]
    beat_position_counts = Counter(beat_positions)
    repeated_pitch_count = sum(count - 1 for count in Counter(pitches).values() if count > 1)
    intervals = [right - left for left, right in zip(pitches, pitches[1:], strict=False)]

    return {
        "note_count": len(notes),
        "duration_beats": round(end_beat, 3),
        "active_bars": active_bars,
        "channels": sorted(set(channels)),
        "min_pitch": min(pitches),
        "max_pitch": max(pitches),
        "mean_pitch": round(mean(pitches), 3),
        "pitch_span": max(pitches) - min(pitches),
        "unique_pitches": len(set(pitches)),
        "mean_velocity": round(mean(velocities), 3) if velocities else 0.0,
        "mean_duration": round(mean(durations), 3),
        "notes_per_bar": round(len(notes) / active_bars, 3),
        "onset_count": len(onset_counts),
        "mean_notes_per_onset": round(len(notes) / max(1, len(onset_counts)), 3),
        "chord_onset_ratio": round(chord_onsets / max(1, len(onset_counts)), 3),
        "polyphonic_note_ratio": round(polyphonic_notes / len(notes), 3),
        "drum_channel_ratio": round(drum_notes / len(notes), 3),
        "long_note_ratio": round(sum(duration >= 2.0 for duration in durations) / len(notes), 3),
        "short_note_ratio": round(sum(duration <= 0.5 for duration in durations) / len(notes), 3),
        "low_pitch_ratio": round(sum(pitch < 48 for pitch in pitches) / len(notes), 3),
        "high_pitch_ratio": round(sum(pitch >= 60 for pitch in pitches) / len(notes), 3),
        "syncopation_ratio": round(
            sum((start * 2) % 2 != 0 for start in starts) / len(notes),
            3,
        ),
        "repeated_pitch_ratio": round(repeated_pitch_count / len(notes), 3),
        "dominant_beat_ratio": round(max(beat_position_counts.values()) / len(notes), 3),
        "mean_abs_interval": (
            round(mean(abs(interval) for interval in intervals), 3) if intervals else 0.0
        ),
    }


def _profile_midi_file(
    path: Path,
    *,
    file_id: str,
    normalized_path: str | Path | None,
    metadata: dict[str, Any],
    file_hash: str,
    duplicate_of: str | None,
) -> DatasetFileProfile:
    midi = mido.MidiFile(path)
    track_profiles: list[TrackProfile] = []
    total_notes = 0
    duration_beats = 0.0
    for track_index, track in enumerate(midi.tracks):
        parsed = _parse_midi_track(track, ticks_per_beat=midi.ticks_per_beat)
        features = midi_note_features(parsed["notes"], parsed["channels"])
        classification = classify_role(
            name=parsed["name"],
            channels=parsed["channels"],
            programs=parsed["programs"],
            instrument_guess=parsed["instrument_guess"],
            features=features,
        )
        total_notes += int(features.get("note_count", 0) or 0)
        duration_beats = max(duration_beats, float(features.get("duration_beats", 0.0) or 0.0))
        if int(features.get("note_count", 0) or 0) == 0 and track_index > 0:
            continue
        track_profiles.append(
            TrackProfile(
                track_index=track_index,
                name=parsed["name"],
                source_kind="midi_track",
                channels=parsed["channels"],
                programs=parsed["programs"],
                instrument_guess=parsed["instrument_guess"],
                classification=classification,
                features=features,
                no_memorization_fingerprint=_fingerprint(
                    {
                        "file_hash": file_hash,
                        "track_index": track_index,
                        "role": classification.role,
                        "features": features,
                    }
                ),
            )
        )

    file_features = {
        "format": "midi",
        "ticks_per_beat": midi.ticks_per_beat,
        "tracks": len(track_profiles),
        "notes": total_notes,
        "note_count": total_notes,
        "duration_beats": round(duration_beats, 3),
        "duration_bars_4_4": max(1, math.ceil(duration_beats / 4.0)) if total_notes else 0,
    }
    return _base_profile(
        path,
        file_id=file_id,
        normalized_path=normalized_path,
        metadata=metadata,
        file_hash=file_hash,
        duplicate_of=duplicate_of,
        file_format="midi",
        track_profiles=track_profiles,
        file_features=file_features,
    )


def _profile_musicxml_file(
    path: Path,
    *,
    file_id: str,
    normalized_path: str | Path | None,
    metadata: dict[str, Any],
    file_hash: str,
    duplicate_of: str | None,
) -> DatasetFileProfile:
    score = converter.parse(path)
    chord_count = len(score.recurse().getElementsByClass(harmony.ChordSymbol))
    parts = list(score.parts) or [score]
    track_profiles: list[TrackProfile] = []
    total_notes = 0
    duration_beats = 0.0
    for part_index, part in enumerate(parts):
        name = str(
            getattr(part, "partName", None)
            or getattr(part, "id", None)
            or f"part_{part_index}"
        )
        notes = _musicxml_part_notes(part)
        features = midi_note_features(notes, [])
        instrument = instrument_guess(name, [], [])
        classification = classify_role(
            name=name,
            channels=[],
            programs=[],
            instrument_guess=instrument,
            features=features,
        )
        total_notes += int(features.get("note_count", 0) or 0)
        duration_beats = max(duration_beats, float(features.get("duration_beats", 0.0) or 0.0))
        if int(features.get("note_count", 0) or 0) == 0 and chord_count == 0:
            continue
        track_profiles.append(
            TrackProfile(
                track_index=part_index,
                name=name,
                source_kind="musicxml_part",
                instrument_guess=instrument,
                classification=classification,
                features=features,
                no_memorization_fingerprint=_fingerprint(
                    {
                        "file_hash": file_hash,
                        "part_index": part_index,
                        "role": classification.role,
                        "features": features,
                    }
                ),
            )
        )

    if chord_count and not track_profiles:
        track_profiles.append(
            TrackProfile(
                track_index=0,
                name="harmony",
                source_kind="musicxml_part",
                instrument_guess="chord_symbols",
                classification=RoleClassification(
                    role="harmony",
                    confidence=0.95,
                    reasons=["MusicXML contains chord symbols without pitched parts"],
                    alternatives={"harmony": 0.95},
                ),
                features={"note_count": 0, "chord_symbols": chord_count},
                no_memorization_fingerprint=_fingerprint(
                    {"file_hash": file_hash, "role": "harmony", "chord_symbols": chord_count}
                ),
            )
        )
    elif chord_count:
        track_profiles.append(
            TrackProfile(
                track_index=len(track_profiles),
                name="harmony",
                source_kind="musicxml_part",
                instrument_guess="chord_symbols",
                classification=RoleClassification(
                    role="harmony",
                    confidence=0.95,
                    reasons=["MusicXML chord symbols detected"],
                    alternatives={"harmony": 0.95},
                ),
                features={"note_count": 0, "chord_symbols": chord_count},
                no_memorization_fingerprint=_fingerprint(
                    {"file_hash": file_hash, "role": "harmony", "chord_symbols": chord_count}
                ),
            )
        )

    file_features = {
        "format": "musicxml",
        "parts": len(parts),
        "tracks": len(track_profiles),
        "notes": total_notes,
        "note_count": total_notes,
        "chord_symbols": chord_count,
        "duration_beats": round(duration_beats, 3),
        "duration_bars_4_4": max(1, math.ceil(duration_beats / 4.0)) if total_notes else 0,
    }
    return _base_profile(
        path,
        file_id=file_id,
        normalized_path=normalized_path,
        metadata=metadata,
        file_hash=file_hash,
        duplicate_of=duplicate_of,
        file_format="musicxml",
        track_profiles=track_profiles,
        file_features=file_features,
    )


def _base_profile(
    path: Path,
    *,
    file_id: str,
    normalized_path: str | Path | None,
    metadata: dict[str, Any],
    file_hash: str,
    duplicate_of: str | None,
    file_format: str,
    track_profiles: list[TrackProfile],
    file_features: dict[str, Any],
) -> DatasetFileProfile:
    roles = _role_coverage(track_profiles, metadata)
    commercial_training = _commercial_training(metadata)
    contains_melody = bool(
        metadata.get("contains_melody", False)
        or any(role in {"melody", "solo"} for role in roles)
    )
    contains_chords = bool(
        metadata.get("contains_chords", False)
        or any(role in {"comping", "harmony", "pad"} for role in roles)
        or int(file_features.get("chord_symbols", 0) or 0) > 0
    )
    contains_arrangement = bool(
        metadata.get("contains_arrangement", False)
        or len({role for role in roles if role != "unknown"}) >= 3
    )
    sensitivity = _pattern_sensitivity(metadata, commercial_training)
    return DatasetFileProfile(
        file_id=file_id,
        original_path=str(path),
        normalized_path=str(normalized_path or ""),
        source=str(metadata.get("source", "local")),
        license=str(metadata.get("license", "unknown")),
        license_confidence=_license_confidence(metadata),
        commercial_training=commercial_training,
        local_learning_only=bool(metadata.get("local_learning_only", False)),
        style=str(metadata.get("style", "unknown")),
        quality=int(metadata.get("quality", 3) or 3),
        tags=list(metadata.get("tags", [])),
        usable_for_training=bool(metadata.get("usable_for_training", False)),
        usable_for_pattern_extraction=bool(metadata.get("usable_for_pattern_extraction", False)),
        duplicate_of=duplicate_of,
        hash=file_hash,
        format=file_format,
        track_profiles=track_profiles,
        file_features={
            **file_features,
            "role_coverage": roles,
            "contains_melody": contains_melody,
            "contains_chords": contains_chords,
            "contains_arrangement": contains_arrangement,
        },
        role_coverage=roles,
        contains_melody=contains_melody,
        contains_chords=contains_chords,
        contains_arrangement=contains_arrangement,
        pattern_sensitivity=sensitivity,
        no_memorization_fingerprint=_fingerprint(
            {
                "file_hash": file_hash,
                "roles": roles,
                "features": file_features,
                "commercial_training": commercial_training,
            }
        ),
    )


def _parse_midi_track(track: mido.MidiTrack, *, ticks_per_beat: int) -> dict[str, Any]:
    absolute = 0
    name = "unnamed_track"
    active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    notes: list[dict[str, Any]] = []
    channels: list[int] = []
    programs: list[int] = []
    for message in track:
        absolute += message.time
        if message.type == "track_name":
            name = message.name
        if not hasattr(message, "channel"):
            continue
        channel = int(message.channel)
        channels.append(channel)
        if message.type == "program_change":
            programs.append(int(message.program))
            continue
        key = (channel, getattr(message, "note", -1))
        if message.type == "note_on" and message.velocity > 0:
            active[key].append((absolute, message.velocity))
        elif message.type in {"note_off", "note_on"}:
            starts = active.get(key)
            if not starts:
                continue
            start_tick, velocity = starts.pop(0)
            duration_ticks = max(1, absolute - start_tick)
            notes.append(
                {
                    "note": int(message.note),
                    "channel": channel,
                    "velocity": int(velocity),
                    "start": start_tick / ticks_per_beat,
                    "duration": duration_ticks / ticks_per_beat,
                }
            )
    unique_channels = sorted(set(channels))
    unique_programs = sorted(set(programs))
    return {
        "name": name,
        "channels": unique_channels,
        "programs": unique_programs,
        "instrument_guess": instrument_guess(name, unique_channels, unique_programs),
        "notes": sorted(notes, key=lambda item: (item["start"], item["note"])),
    }


def _musicxml_part_notes(part: Any) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for element in part.recurse().notes:
        start = float(element.offset)
        duration = float(getattr(element.duration, "quarterLength", 0.0) or 0.0)
        velocity = 80
        pitches = getattr(element, "pitches", None)
        if pitches:
            for pitch in pitches:
                notes.append(
                    {
                        "note": int(pitch.midi),
                        "channel": 0,
                        "velocity": velocity,
                        "start": start,
                        "duration": max(0.25, duration),
                    }
                )
            continue
        pitch = getattr(element, "pitch", None)
        if pitch is not None:
            notes.append(
                {
                    "note": int(pitch.midi),
                    "channel": 0,
                    "velocity": velocity,
                    "start": start,
                    "duration": max(0.25, duration),
                }
            )
    return sorted(notes, key=lambda item: (item["start"], item["note"]))


def _metadata_map(metadata: DatasetManifestEntry | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, DatasetManifestEntry):
        return metadata.model_dump(mode="json")
    return dict(metadata)


def _role_coverage(
    track_profiles: list[TrackProfile],
    metadata: dict[str, Any],
) -> list[DatasetRole]:
    roles = {
        track.classification.role
        for track in track_profiles
        if track.classification.role != "unknown"
    }
    for role in metadata.get("roles", []):
        if role in ROLE_ORDER:
            roles.add(role)
    if not roles and track_profiles:
        roles.add("unknown")
    return [role for role in ROLE_ORDER if role in roles]


def _commercial_training(metadata: dict[str, Any]) -> CommercialTrainingUse:
    raw = metadata.get("commercial_training")
    if raw in {"allowed", "forbidden"}:
        return raw
    license_name = str(metadata.get("license", "unknown")).strip().lower()
    if license_name in PERMISSIVE_LICENSES and bool(metadata.get("usable_for_training", False)):
        return "allowed"
    if license_name in BLOCKED_LICENSES or bool(metadata.get("local_learning_only", False)):
        return "forbidden"
    if license_name in REVIEW_LICENSES:
        return "review_required"
    return "review_required"


def _license_confidence(metadata: dict[str, Any]) -> LicenseConfidence:
    raw = metadata.get("license_confidence")
    if raw in {"high", "medium", "low"}:
        return raw
    license_name = str(metadata.get("license", "unknown")).strip().lower()
    if license_name in PERMISSIVE_LICENSES:
        return "high"
    if license_name in REVIEW_LICENSES:
        return "medium"
    return "low"


def _pattern_sensitivity(
    metadata: dict[str, Any],
    commercial_training: CommercialTrainingUse,
) -> dict[str, Any]:
    local_learning_only = bool(metadata.get("local_learning_only", False))
    if commercial_training == "allowed" and not local_learning_only:
        level = "low"
    elif commercial_training == "forbidden" or local_learning_only:
        level = "high"
    else:
        level = "review"
    return {
        "level": level,
        "commercial_training": commercial_training,
        "local_learning_only": local_learning_only,
        "license_confidence": _license_confidence(metadata),
    }


def _has_any(value: str, tokens: tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def _fingerprint(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
