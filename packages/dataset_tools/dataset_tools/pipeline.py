from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mido
from music21 import converter, harmony

from dataset_tools.models import (
    SUPPORTED_EXTENSIONS,
    DatasetManifest,
    DatasetManifestEntry,
    ExtractedPattern,
    ImportSummary,
    NormalizedFile,
    PatternIndex,
)
from dataset_tools.profiler import (
    build_dataset_profile_report,
    classify_midi_track_role,
    instrument_guess,
    profile_dataset_file,
    write_role_manifest,
)


def create_manifest(
    source_dir: str | Path,
    manifest_path: str | Path,
    *,
    default_metadata: dict[str, Any] | None = None,
    metadata_by_name: dict[str, dict[str, Any]] | None = None,
) -> DatasetManifest:
    source_root = Path(source_dir)
    defaults = default_metadata or {}
    per_file = metadata_by_name or {}
    entries: list[DatasetManifestEntry] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        metadata = {**defaults, **per_file.get(path.name, {})}
        roles = _metadata_roles(metadata)
        license_name = str(metadata.get("license", "unknown"))
        commercial_training = str(
            metadata.get(
                "commercial_training",
                _commercial_training_for_metadata(metadata),
            )
        )
        entries.append(
            DatasetManifestEntry(
                path=str(path.relative_to(source_root)),
                source=str(metadata.get("source", "local")),
                license=license_name,
                license_confidence=str(
                    metadata.get(
                        "license_confidence",
                        _license_confidence_for_license(license_name),
                    )
                ),
                commercial_training=commercial_training,
                local_learning_only=bool(metadata.get("local_learning_only", False)),
                copyright_notes=str(metadata.get("copyright_notes", "")),
                usable_for_training=bool(metadata.get("usable_for_training", False)),
                usable_for_pattern_extraction=bool(
                    metadata.get("usable_for_pattern_extraction", False)
                ),
                style=str(metadata.get("style", "unknown")),
                quality=int(metadata.get("quality", 3)),
                tags=list(metadata.get("tags", [])),
                roles=roles,
                contains_melody=bool(
                    metadata.get("contains_melody", any(role == "melody" for role in roles))
                ),
                contains_chords=bool(
                    metadata.get(
                        "contains_chords",
                        any(role in {"comping", "harmony"} for role in roles),
                    )
                ),
                contains_arrangement=bool(
                    metadata.get("contains_arrangement", len(set(roles)) >= 3)
                ),
                imported_at=str(
                    metadata.get("imported_at", datetime.now(UTC).isoformat())
                ),
                hash=sha256_file(path),
            )
        )
    manifest = DatasetManifest(entries=entries)
    manifest.save_json(manifest_path)
    return manifest


def import_dataset(
    source_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> ImportSummary:
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    normalized_root = output_root / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest.load_json(manifest_path)
    normalized_files: list[NormalizedFile] = []
    file_profiles = []
    pattern_index = PatternIndex()
    seen_hashes: dict[str, str] = {}
    counters: Counter[str] = Counter()

    for entry_index, entry in enumerate(manifest.entries, start=1):
        source_path = _resolve_source_path(source_root, entry.path)
        file_hash = sha256_file(source_path)
        if entry.hash and entry.hash != file_hash:
            raise ValueError(f"Hash mismatch for {source_path}")
        if not entry.hash:
            entry.hash = file_hash

        file_id = f"file_{entry_index:04d}_{file_hash[:12]}"
        duplicate_of = seen_hashes.get(file_hash)
        canonical_file_id = duplicate_of or file_id
        normalized_path = normalized_root / f"{canonical_file_id}{source_path.suffix.lower()}"
        if duplicate_of is None:
            seen_hashes[file_hash] = file_id
            shutil.copy2(source_path, normalized_path)
            counters["imported_files"] += 1
        else:
            counters["duplicate_files"] += 1

        profile = profile_dataset_file(
            source_path,
            file_id=file_id,
            normalized_path=normalized_path,
            metadata=entry,
            file_hash=file_hash,
            duplicate_of=duplicate_of,
        )
        file_profiles.append(profile)
        counters["profiled_files"] += 1
        entry.roles = profile.role_coverage
        entry.contains_melody = profile.contains_melody
        entry.contains_chords = profile.contains_chords
        entry.contains_arrangement = profile.contains_arrangement
        entry.license_confidence = profile.license_confidence
        entry.commercial_training = profile.commercial_training
        entry.local_learning_only = profile.local_learning_only

        normalized = NormalizedFile(
            file_id=file_id,
            original_path=str(source_path),
            normalized_path=str(normalized_path),
            source=entry.source,
            license=entry.license,
            license_confidence=profile.license_confidence,
            commercial_training=profile.commercial_training,
            local_learning_only=profile.local_learning_only,
            hash=file_hash,
            style=entry.style,
            quality=entry.quality,
            tags=entry.tags,
            usable_for_training=entry.usable_for_training,
            usable_for_pattern_extraction=entry.usable_for_pattern_extraction,
            duplicate_of=duplicate_of,
            role_hints=sorted({*_role_hints(entry), *profile.role_coverage}),
            roles=profile.role_coverage,
            contains_melody=profile.contains_melody,
            contains_chords=profile.contains_chords,
            contains_arrangement=profile.contains_arrangement,
            stats={
                **profile.file_features,
                "pattern_sensitivity": profile.pattern_sensitivity,
                "no_memorization_fingerprint": profile.no_memorization_fingerprint,
            },
        )
        normalized_files.append(normalized)

        if duplicate_of is not None:
            continue
        if not entry.usable_for_pattern_extraction:
            counters["skipped_for_license"] += 1
            continue
        if entry.quality < 3:
            counters["skipped_for_quality"] += 1
            continue

        for pattern in extract_patterns(source_path, normalized):
            pattern_index.add(pattern)

    counters["extracted_patterns"] = len(pattern_index.patterns)
    pattern_counts = Counter(pattern.category for pattern in pattern_index.patterns)
    profile_report = build_dataset_profile_report(file_profiles)

    manifest_output_path = output_root / "dataset_manifest.json"
    normalized_output_path = output_root / "normalized_files.json"
    pattern_index_path = output_root / "pattern_index.json"
    profile_report_path = output_root / "dataset_profile.json"
    role_manifest_path = output_root / "role_manifest.json"
    summary_path = output_root / "import_summary.json"

    manifest.save_json(manifest_output_path)
    normalized_output_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in normalized_files], indent=2)
        + "\n",
        encoding="utf-8",
    )
    pattern_index.save_json(pattern_index_path)
    profile_report.save_json(profile_report_path)
    write_role_manifest(profile_report, role_manifest_path)

    summary = ImportSummary(
        imported_files=counters["imported_files"],
        duplicate_files=counters["duplicate_files"],
        skipped_for_license=counters["skipped_for_license"],
        skipped_for_quality=counters["skipped_for_quality"],
        extracted_patterns=counters["extracted_patterns"],
        pattern_counts=dict(sorted(pattern_counts.items())),
        profiled_files=counters["profiled_files"],
        role_counts=dict(sorted(profile_report.role_counts.items())),
        manifest_path=str(manifest_output_path),
        normalized_files_path=str(normalized_output_path),
        pattern_index_path=str(pattern_index_path),
        profile_report_path=str(profile_report_path),
        role_manifest_path=str(role_manifest_path),
        summary_path=str(summary_path),
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def extract_patterns(path: str | Path, normalized: NormalizedFile) -> list[ExtractedPattern]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        return _extract_midi_patterns(source_path, normalized)
    if suffix in {".musicxml", ".xml"}:
        return _extract_musicxml_patterns(source_path, normalized)
    return []


def load_pattern_index(path: str | Path) -> PatternIndex:
    return PatternIndex.load_json(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_midi_patterns(
    path: Path,
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    midi = mido.MidiFile(path)
    tracks = _midi_tracks(midi)
    patterns: list[ExtractedPattern] = []
    for track in tracks:
        role = track["role"]
        notes = track["notes"]
        if not notes:
            continue
        if role == "drums":
            patterns.extend(_drum_grooves(track, normalized))
        elif role == "walking_bass":
            patterns.extend(_walking_bass_cells(track, normalized))
        elif role == "piano":
            patterns.extend(_piano_voicings(track, normalized))
        elif role == "horn_response":
            patterns.extend(_horn_responses(track, normalized))
        elif role in {"melody", "solo"}:
            patterns.extend(_melodic_motifs(track, normalized))
    return patterns


def _extract_musicxml_patterns(
    path: Path,
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    score = converter.parse(path)
    chord_symbols = [
        getattr(chord_symbol, "figure", str(chord_symbol))
        for chord_symbol in score.recurse().getElementsByClass(harmony.ChordSymbol)
    ]
    chord_symbols = [symbol for symbol in chord_symbols if symbol]
    patterns: list[ExtractedPattern] = []
    for window in (2, 4, 8, 12, 16, 32):
        if len(chord_symbols) < window:
            continue
        for start in range(0, len(chord_symbols) - window + 1, window):
            progression = chord_symbols[start : start + window]
            patterns.append(
                _pattern(
                    "progressions",
                    "harmony",
                    normalized,
                    payload={
                        "chords": progression,
                        "length": window,
                        "start_index": start,
                    },
                    context={"meter": "4/4"},
                )
            )
    return patterns


def _midi_tracks(midi: mido.MidiFile) -> list[dict[str, Any]]:
    output = []
    ticks_per_beat = midi.ticks_per_beat
    for track_number, track in enumerate(midi.tracks):
        absolute = 0
        name = f"track_{track_number}"
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
            channels.append(message.channel)
            if message.type == "program_change":
                programs.append(int(message.program))
                continue
            key = (message.channel, getattr(message, "note", -1))
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
                        "note": message.note,
                        "channel": message.channel,
                        "velocity": velocity,
                        "start": start_tick / ticks_per_beat,
                        "duration": duration_ticks / ticks_per_beat,
                    }
                )
        unique_channels = sorted(set(channels))
        unique_programs = sorted(set(programs))
        classification = classify_midi_track_role(
            name,
            unique_channels,
            notes,
            programs=unique_programs,
        )
        role = _extraction_role(classification.role)
        output.append(
            {
                "name": name,
                "role": role,
                "classified_role": classification.role,
                "role_confidence": classification.confidence,
                "role_reasons": classification.reasons,
                "instrument_guess": instrument_guess(name, unique_channels, unique_programs),
                "channels": unique_channels,
                "programs": unique_programs,
                "notes": sorted(notes, key=lambda item: (item["start"], item["note"])),
            }
        )
    return output


def _drum_grooves(
    track: dict[str, Any],
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    patterns = []
    for bar_index, notes in _notes_by_bar(track["notes"]).items():
        if not notes:
            continue
        events = [
            {
                "beat": round(note["start"] % 4, 3),
                "pitch": note["note"],
                "duration": round(note["duration"], 3),
            }
            for note in notes
        ]
        patterns.append(
            _pattern(
                "drum_grooves",
                "drums",
                normalized,
                payload={"bar": bar_index + 1, "events": events, "meter": "4/4"},
                context=_track_context(track),
            )
        )
        if len(patterns) >= 4:
            break
    return patterns


def _walking_bass_cells(
    track: dict[str, Any],
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    patterns = []
    for bar_index, notes in _notes_by_bar(track["notes"]).items():
        ordered = sorted(notes, key=lambda item: item["start"])[:4]
        if len(ordered) < 3:
            continue
        pitches = [note["note"] for note in ordered]
        starts = [round(note["start"] % 4, 3) for note in ordered]
        patterns.append(
            _pattern(
                "walking_bass_cells",
                "walking_bass",
                normalized,
                payload={
                    "bar": bar_index + 1,
                    "pitch_intervals": [pitch - pitches[0] for pitch in pitches],
                    "rhythm": starts,
                    "contour": _contour(pitches),
                    "start_degree": 0,
                    "end_degree": pitches[-1] - pitches[0],
                },
                context=_track_context(track),
            )
        )
        if len(patterns) >= 8:
            break
    return patterns


def _piano_voicings(
    track: dict[str, Any],
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for note in track["notes"]:
        grouped[round(note["start"], 3)].append(note)
    patterns = []
    for start, notes in sorted(grouped.items()):
        if len(notes) < 2:
            continue
        pitches = sorted(note["note"] for note in notes)
        patterns.append(
            _pattern(
                "piano_voicings",
                "comping",
                normalized,
                payload={
                    "relative_notes": [pitch - pitches[0] for pitch in pitches],
                    "register": pitches[0],
                    "density": len(pitches),
                    "inversion": _inversion_label(pitches),
                    "onset": start,
                },
                context=_track_context(track),
            )
        )
        if len(patterns) >= 8:
            break
    return patterns


def _melodic_motifs(
    track: dict[str, Any],
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    ordered = sorted(track["notes"], key=lambda item: item["start"])
    patterns = []
    for start in range(0, max(0, len(ordered) - 3), 4):
        window = ordered[start : start + 4]
        if len(window) < 3:
            continue
        pitches = [note["note"] for note in window]
        patterns.append(
            _pattern(
                "melodic_motifs",
                "melody",
                normalized,
                payload={
                    "relative_degrees": [pitch - pitches[0] for pitch in pitches],
                    "rhythm": [round(note["duration"], 3) for note in window],
                    "contour": _contour(pitches),
                    "phrase_position": start,
                },
                context=_track_context(track),
            )
        )
        if len(patterns) >= 8:
            break
    return patterns


def _horn_responses(
    track: dict[str, Any],
    normalized: NormalizedFile,
) -> list[ExtractedPattern]:
    ordered = sorted(track["notes"], key=lambda item: item["start"])
    patterns = []
    for start in range(0, max(0, len(ordered) - 1), 2):
        window = ordered[start : start + 2]
        if len(window) < 2:
            continue
        pitches = [note["note"] for note in window]
        patterns.append(
            _pattern(
                "horn_responses",
                "horn_response",
                normalized,
                payload={
                    "voices": 1,
                    "relative_notes": [pitch - pitches[0] for pitch in pitches],
                    "rhythm": [round(note["duration"], 3) for note in window],
                    "spacing": 0,
                    "range": [min(pitches), max(pitches)],
                },
                context=_track_context(track),
            )
        )
        if len(patterns) >= 8:
            break
    return patterns


def _pattern(
    category: str,
    role: str,
    normalized: NormalizedFile,
    *,
    payload: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> ExtractedPattern:
    base_context = dict(context or {})
    fingerprint = _fingerprint(
        {
            "category": category,
            "role": role,
            "style": normalized.style,
            "payload": payload,
        }
    )
    source_pattern_fingerprint = _fingerprint(
        {
            "source_hash": normalized.hash,
            "pattern_fingerprint": fingerprint,
            "source_file_id": normalized.file_id,
        }
    )
    base_context.setdefault("pattern_sensitivity", _pattern_sensitivity(normalized))
    base_context.setdefault("no_memorization_fingerprint", source_pattern_fingerprint)
    role_confidence = float(base_context.get("role_confidence", 1.0) or 1.0)
    pattern_id = f"{category}_{fingerprint[:12]}"
    return ExtractedPattern(
        id=pattern_id,
        category=category,
        role=role,
        style=normalized.style,
        quality=normalized.quality,
        source_file_id=normalized.file_id,
        source_path=normalized.original_path,
        source_hash=normalized.hash,
        license=normalized.license,
        usable_for_training=normalized.usable_for_training,
        usable_for_pattern_extraction=normalized.usable_for_pattern_extraction,
        tags=normalized.tags,
        weight=round(float(normalized.quality) * max(0.2, min(1.0, role_confidence)), 3),
        context=base_context,
        payload=payload,
        fingerprint=fingerprint,
    )


def _file_stats(path: Path) -> dict[str, Any]:
    profile = profile_dataset_file(path)
    return profile.file_features


def _notes_by_bar(notes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        grouped[int(note["start"] // 4)].append(note)
    return dict(grouped)


def _track_role(
    name: str,
    channels: list[int],
    notes: list[dict[str, Any]],
) -> str:
    classification = classify_midi_track_role(name, channels, notes)
    return _extraction_role(classification.role)


def _role_hints(entry: DatasetManifestEntry) -> list[str]:
    hints = {
        tag
        for tag in entry.tags
        if tag
        in {
            "drums",
            "walking_bass",
            "bass",
            "piano",
            "comping",
            "melody",
            "horn_response",
            "horns",
            "pad",
            "solo",
            "harmony",
        }
    }
    hints.update(entry.roles)
    return sorted(hints)


def _metadata_roles(metadata: dict[str, Any]) -> list[str]:
    raw_roles = metadata.get("roles")
    if isinstance(raw_roles, list):
        candidates = [str(role) for role in raw_roles]
    else:
        candidates = [str(tag) for tag in metadata.get("tags", [])]
    role_aliases = {
        "walking_bass": "bass",
        "double_bass": "bass",
        "piano": "comping",
        "horn_response": "horns",
        "horn": "horns",
        "progressions": "harmony",
        "chords": "harmony",
    }
    allowed = {
        "melody",
        "bass",
        "drums",
        "comping",
        "horns",
        "pad",
        "solo",
        "harmony",
        "unknown",
    }
    normalized = {role_aliases.get(role, role) for role in candidates}
    return sorted(role for role in normalized if role in allowed)


def _commercial_training_for_metadata(metadata: dict[str, Any]) -> str:
    raw = metadata.get("commercial_training")
    if raw in {"allowed", "forbidden", "review_required"}:
        return str(raw)
    license_name = str(metadata.get("license", "unknown")).strip().lower()
    usable_for_training = bool(metadata.get("usable_for_training", False))
    local_learning_only = bool(metadata.get("local_learning_only", False))
    permissive = {
        "cc0",
        "cc0-1.0",
        "public domain",
        "public-domain",
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
    }
    blocked = {"", "unknown", "proprietary", "all rights reserved", "private"}
    if license_name in permissive and usable_for_training and not local_learning_only:
        return "allowed"
    if license_name in blocked or local_learning_only:
        return "forbidden"
    return "review_required"


def _license_confidence_for_license(license_name: str) -> str:
    normalized = license_name.strip().lower()
    if normalized in {
        "cc0",
        "cc0-1.0",
        "public domain",
        "public-domain",
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
    }:
        return "high"
    if normalized in {"cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0", "creative commons"}:
        return "medium"
    return "low"


def _extraction_role(classified_role: str) -> str:
    return {
        "drums": "drums",
        "bass": "walking_bass",
        "comping": "piano",
        "horns": "horn_response",
        "melody": "melody",
        "solo": "solo",
    }.get(classified_role, classified_role)


def _track_context(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_name": track["name"],
        "classified_role": track.get("classified_role", "unknown"),
        "extraction_role": track.get("role", "unknown"),
        "role_confidence": track.get("role_confidence", 0.0),
        "role_reasons": track.get("role_reasons", []),
        "instrument_guess": track.get("instrument_guess", "unknown"),
        "channels": track.get("channels", []),
        "programs": track.get("programs", []),
    }


def _pattern_sensitivity(normalized: NormalizedFile) -> dict[str, Any]:
    if normalized.commercial_training == "allowed" and not normalized.local_learning_only:
        level = "low"
    elif normalized.commercial_training == "forbidden" or normalized.local_learning_only:
        level = "high"
    else:
        level = "review"
    return {
        "level": level,
        "commercial_training": normalized.commercial_training,
        "local_learning_only": normalized.local_learning_only,
        "license_confidence": normalized.license_confidence,
    }


def _contour(pitches: list[int]) -> list[int]:
    contour = []
    for left, right in zip(pitches, pitches[1:], strict=False):
        if right > left:
            contour.append(1)
        elif right < left:
            contour.append(-1)
        else:
            contour.append(0)
    return contour


def _inversion_label(pitches: list[int]) -> str:
    if not pitches:
        return "unknown"
    return "root_position" if pitches[0] % 12 in {0, 5, 7} else "inverted"


def _fingerprint(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_source_path(source_root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return source_root / path
