from __future__ import annotations

import html
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import mido
from music21 import converter

from arranger_core.catalogs import InstrumentCatalog
from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.music_theory import note_to_midi
from arranger_core.schema import (
    ArrangementProject,
    ChordSymbol,
    NoteEvent,
    RestEvent,
    Track,
)

Severity = Literal["error", "warning"]
Issue = dict[str, Any]
Report = dict[str, Any]

DRUM_ALLOWED_MIDI = {
    36,  # kick
    38,  # snare
    42,  # closed hihat
    44,  # hihat pedal
    45,  # low tom
    47,  # mid tom
    49,  # crash
    50,  # high tom
    51,  # ride
}


class MusicValidationError(ValueError):
    def __init__(self, report: Report) -> None:
        self.report = report
        errors = report.get("errors", [])
        message = f"Music validation failed with {len(errors)} error(s)"
        if errors:
            message = f"{message}: {errors[0].get('message', 'validation error')}"
        super().__init__(message)


def validate_project(
    project: ArrangementProject,
    *,
    instrument_catalog: InstrumentCatalog | None = None,
    chord_parser: ChordParser | None = None,
) -> Report:
    catalog = instrument_catalog or InstrumentCatalog.load_default()
    parser = chord_parser or ChordParser.load_default()
    issues: list[Issue] = []
    metrics: dict[str, Any] = {
        "bars": project.bar_count,
        "tracks": len(project.tracks),
        "note_events": _note_event_count(project),
    }

    issues.extend(_validate_bar_durations(project))
    issues.extend(_validate_instrument_ranges(project, catalog))
    issues.extend(_validate_transposition(project, catalog))
    issues.extend(_validate_harmony(project, parser, metrics))
    issues.extend(_validate_voice_leading(project, catalog))
    issues.extend(_validate_breathing(project, catalog))
    issues.extend(_validate_piano_voicings(project))
    issues.extend(_validate_drums(project))

    metrics["avg_note_density"] = _avg_note_density(project)
    return _build_report(project, issues, metrics=metrics)


def validate_export_package(
    project: ArrangementProject,
    manifest: dict[str, Any],
    output_dir: str | Path,
) -> Report:
    output_root = Path(output_dir)
    files = manifest.get("files", [])
    issues: list[Issue] = []
    metrics: dict[str, Any] = {
        "exported_files": len(files),
        "midi_track_files": sum(
            1 for file_record in files if file_record.get("kind") == "midi_track"
        ),
    }

    required_kinds = {
        "project_json",
        "generation_spec_json",
        "validation_report_json",
        "validation_report_html",
        "midi_full",
        "musicxml_full",
        "model_trace_json",
        "session_readme",
        "takes_manifest_json",
        "export_manifest",
    }
    available_kinds = {str(file_record.get("kind")) for file_record in files}
    for kind in sorted(required_kinds - available_kinds):
        issues.append(
            _issue(
                "error",
                "ExportValidator",
                "missing_manifest_kind",
                f"Export manifest is missing required file kind {kind!r}",
                details={"kind": kind},
            )
        )

    for file_record in files:
        status = file_record.get("status", "created")
        if status == "skipped":
            continue
        path = _resolve_export_path(output_root, file_record)
        if not path.exists():
            issues.append(
                _issue(
                    "error",
                    "ExportValidator",
                    "missing_file",
                    f"Exported file does not exist: {path}",
                    details={"kind": file_record.get("kind"), "path": str(path)},
                )
            )

    musicxml_path = _manifest_path(files, "musicxml_full")
    if musicxml_path is not None:
        try:
            converter.parse(musicxml_path)
        except Exception as exc:
            issues.append(
                _issue(
                    "error",
                    "ExportValidator",
                    "musicxml_not_parseable",
                    f"MusicXML is not parseable: {exc}",
                    details={"path": str(musicxml_path)},
                )
            )

    midi_path = _manifest_path(files, "midi_full")
    if midi_path is not None:
        try:
            midi = mido.MidiFile(midi_path)
            metrics["midi_tracks"] = len(midi.tracks)
        except Exception as exc:
            issues.append(
                _issue(
                    "error",
                    "ExportValidator",
                    "midi_not_parseable",
                    f"MIDI is not parseable: {exc}",
                    details={"path": str(midi_path)},
                )
            )

    midi_track_files = metrics["midi_track_files"]
    if midi_track_files != len(project.tracks):
        issues.append(
            _issue(
                "error",
                "ExportValidator",
                "midi_track_count_mismatch",
                (
                    f"Expected {len(project.tracks)} MIDI track files, "
                    f"found {midi_track_files}"
                ),
                details={"expected": len(project.tracks), "actual": midi_track_files},
            )
        )

    if manifest.get("pdf_status") == "created" and "pdf_full" not in available_kinds:
        issues.append(
            _issue(
                "error",
                "ExportValidator",
                "missing_full_pdf",
                "PDF export status is created but no full-score PDF is listed",
            )
        )

    return _build_report(project, issues, metrics=metrics)


def merge_validation_reports(*reports: Report) -> Report:
    reports = tuple(report for report in reports if report)
    if not reports:
        return _build_report(None, [], metrics={})

    issues: list[Issue] = []
    metrics: dict[str, Any] = {}
    project_id = reports[0].get("project_id")
    for report in reports:
        issues.extend(report.get("errors", []))
        issues.extend(report.get("warnings", []))
        metrics.update(report.get("metrics", {}))

    merged = _build_report(None, issues, metrics=metrics)
    merged["project_id"] = project_id
    return merged


def write_validation_json(report: Report, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_validation_html(report: Report, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_validation_html(report), encoding="utf-8")
    return output_path


def _validate_bar_durations(project: ArrangementProject) -> list[Issue]:
    return [
        _issue(
            "error",
            "BarDurationValidator",
            "bar_duration",
            duration_issue.message,
            track_id=duration_issue.track_id,
            bar_number=duration_issue.bar_number,
            voice=duration_issue.voice,
            details=duration_issue.model_dump(mode="json"),
        )
        for duration_issue in project.validate_bar_durations()
    ]


def _validate_instrument_ranges(
    project: ArrangementProject,
    catalog: InstrumentCatalog,
) -> list[Issue]:
    issues: list[Issue] = []
    for track, bar, event in _iter_bar_note_events(project):
        try:
            instrument = catalog.get(track.instrument)
        except KeyError:
            issues.append(
                _issue(
                    "error",
                    "InstrumentRangeValidator",
                    "unknown_instrument",
                    f"Unknown instrument {track.instrument!r}",
                    track_id=track.id,
                    bar_number=bar.number,
                )
            )
            continue

        try:
            midi_note = note_to_midi(event.pitch)
        except ValueError as exc:
            issues.append(
                _issue(
                    "error",
                    "InstrumentRangeValidator",
                    "invalid_pitch",
                    str(exc),
                    track_id=track.id,
                    bar_number=bar.number,
                    beat=event.start + 1,
                )
            )
            continue

        low_abs, high_abs = _range_midi(instrument.sounding_range)
        low_comfort, high_comfort = _range_midi(instrument.comfortable_range)
        if midi_note < low_abs or midi_note > high_abs:
            issues.append(
                _issue(
                    "error",
                    "InstrumentRangeValidator",
                    "outside_absolute_range",
                    (
                        f"{event.pitch} is outside {track.instrument} absolute range "
                        f"{instrument.sounding_range[0]}-{instrument.sounding_range[1]}"
                    ),
                    track_id=track.id,
                    bar_number=bar.number,
                    beat=event.start + 1,
                    details={"pitch": event.pitch, "instrument": track.instrument},
                )
            )
        elif midi_note < low_comfort or midi_note > high_comfort:
            issues.append(
                _issue(
                    "warning",
                    "InstrumentRangeValidator",
                    "outside_comfortable_range",
                    (
                        f"{event.pitch} is outside {track.instrument} comfortable range "
                        f"{instrument.comfortable_range[0]}-{instrument.comfortable_range[1]}"
                    ),
                    track_id=track.id,
                    bar_number=bar.number,
                    beat=event.start + 1,
                    details={"pitch": event.pitch, "instrument": track.instrument},
                )
            )
    return issues


def _validate_transposition(
    project: ArrangementProject,
    catalog: InstrumentCatalog,
) -> list[Issue]:
    issues: list[Issue] = []
    for track, bar, event in _iter_bar_note_events(project):
        try:
            instrument = catalog.get(track.instrument)
        except KeyError:
            continue
        if instrument.transposition_semitones == 0:
            continue
        sounding_midi = note_to_midi(event.pitch) + instrument.transposition_semitones
        if sounding_midi < 0 or sounding_midi > 127:
            issues.append(
                _issue(
                    "error",
                    "TranspositionValidator",
                    "sounding_pitch_out_of_midi_range",
                    (
                        f"{track.instrument} written pitch {event.pitch} transposes "
                        "outside MIDI range"
                    ),
                    track_id=track.id,
                    bar_number=bar.number,
                    beat=event.start + 1,
                    details={
                        "written_pitch": event.pitch,
                        "transposition_semitones": instrument.transposition_semitones,
                    },
                )
            )
    return issues


def _validate_harmony(
    project: ArrangementProject,
    parser: ChordParser,
    metrics: dict[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []
    chords_by_bar = _chords_by_bar(project.chord_grid)
    track_scores: dict[str, float] = {}
    for track in project.tracks:
        if track.role == "drums" or track.instrument == "drum_kit":
            continue
        harmonic_notes = 0
        checked_notes = 0
        for bar in track.bars:
            for event in bar.events:
                if not isinstance(event, NoteEvent):
                    continue
                chord = _active_chord(chords_by_bar, bar.number, event.start)
                if chord is None:
                    continue
                parsed = _safe_parse(parser, chord.symbol)
                if parsed is None:
                    continue
                checked_notes += 1
                if _is_harmonically_supported(event, parsed):
                    harmonic_notes += 1
        if checked_notes == 0:
            continue
        score = harmonic_notes / checked_notes
        track_scores[track.id] = round(score, 3)
        threshold = 0.5 if track.role == "walking_bass" else 0.55
        if score < threshold:
            issues.append(
                _issue(
                    "warning",
                    "HarmonyValidator",
                    "low_harmony_score",
                    (
                        f"Track {track.id!r} has low chord-tone/tension support "
                        f"({score:.2f})"
                    ),
                    track_id=track.id,
                    details={"score": round(score, 3), "threshold": threshold},
                )
            )
    metrics["harmony_score_by_track"] = track_scores
    metrics["harmony_score"] = (
        round(sum(track_scores.values()) / len(track_scores), 3)
        if track_scores
        else None
    )
    return issues


def _validate_voice_leading(
    project: ArrangementProject,
    catalog: InstrumentCatalog,
) -> list[Issue]:
    issues: list[Issue] = []
    for track in project.tracks:
        if track.role == "drums" or track.instrument == "drum_kit":
            continue
        if track.instrument == "piano" and track.role in {"comping", "piano"}:
            continue
        notes = [
            (bar.number, event)
            for bar in track.bars
            for event in bar.events
            if isinstance(event, NoteEvent) and event.voice == 1
        ]
        previous: tuple[int, NoteEvent] | None = None
        for current in notes:
            if previous is None:
                previous = current
                continue
            _, previous_event = previous
            bar_number, event = current
            leap = abs(note_to_midi(event.pitch) - note_to_midi(previous_event.pitch))
            limit = 12 if _instrument_requires_breath(track, catalog) else 16
            if leap > limit:
                issues.append(
                    _issue(
                        "warning",
                        "VoiceLeadingValidator",
                        "large_melodic_leap",
                        f"Large leap of {leap} semitones in track {track.id!r}",
                        track_id=track.id,
                        bar_number=bar_number,
                        beat=event.start + 1,
                        details={"semitones": leap, "limit": limit},
                    )
                )
            previous = current
    return issues


def _validate_breathing(
    project: ArrangementProject,
    catalog: InstrumentCatalog,
) -> list[Issue]:
    issues: list[Issue] = []
    for track in project.tracks:
        if not _instrument_requires_breath(track, catalog):
            continue
        longest_phrase = 0.0
        current_phrase = 0.0
        rests = 0
        for bar in track.bars:
            for event in sorted(bar.events, key=lambda item: item.start):
                if isinstance(event, RestEvent) and event.duration >= 0.5:
                    rests += 1
                    longest_phrase = max(longest_phrase, current_phrase)
                    current_phrase = 0.0
                elif isinstance(event, NoteEvent):
                    current_phrase += event.duration
        longest_phrase = max(longest_phrase, current_phrase)
        if rests == 0 and _note_count(track) > 0:
            issues.append(
                _issue(
                    "warning",
                    "BreathValidator",
                    "no_breaths",
                    f"Breath instrument track {track.id!r} has no notated rests",
                    track_id=track.id,
                )
            )
        if longest_phrase > 16:
            issues.append(
                _issue(
                    "warning",
                    "BreathValidator",
                    "phrase_too_long",
                    (
                        f"Breath instrument track {track.id!r} has a phrase of "
                        f"{longest_phrase:g} beats"
                    ),
                    track_id=track.id,
                    details={"longest_phrase_beats": longest_phrase},
                )
            )
    return issues


def _validate_piano_voicings(project: ArrangementProject) -> list[Issue]:
    issues: list[Issue] = []
    for track in project.tracks:
        if track.instrument != "piano":
            continue
        for bar in track.bars:
            slots = _notes_by_slot(bar.events)
            for (start, _duration, voice), notes in slots.items():
                if len(notes) <= 1:
                    continue
                midi_notes = sorted(note_to_midi(event.pitch) for event in notes)
                span = midi_notes[-1] - midi_notes[0]
                if len(notes) > 6:
                    issues.append(
                        _issue(
                            "warning",
                            "PianoPlayabilityValidator",
                            "too_many_notes",
                            f"Piano voicing has {len(notes)} notes",
                            track_id=track.id,
                            bar_number=bar.number,
                            beat=start + 1,
                            voice=voice,
                        )
                    )
                if span > 28:
                    issues.append(
                        _issue(
                            "warning",
                            "PianoPlayabilityValidator",
                            "wide_voicing",
                            f"Piano voicing spans {span} semitones",
                            track_id=track.id,
                            bar_number=bar.number,
                            beat=start + 1,
                            voice=voice,
                            details={"span_semitones": span},
                        )
                    )
                root_pc = notes[0].annotations.get("root_pc")
                if root_pc is not None and any(midi % 12 == root_pc for midi in midi_notes):
                    issues.append(
                        _issue(
                            "warning",
                            "PianoPlayabilityValidator",
                            "rootless_voicing_has_root",
                            "Rootless piano voicing doubles the root",
                            track_id=track.id,
                            bar_number=bar.number,
                            beat=start + 1,
                            voice=voice,
                        )
                    )
    return issues


def _validate_drums(project: ArrangementProject) -> list[Issue]:
    issues: list[Issue] = []
    for track in project.tracks:
        if track.role != "drums" and track.instrument != "drum_kit":
            continue
        if track.channel is not None and track.channel != 10:
            issues.append(
                _issue(
                    "error",
                    "DrumValidator",
                    "wrong_midi_channel",
                    f"Drum track {track.id!r} is assigned to MIDI channel {track.channel}",
                    track_id=track.id,
                    details={"expected_channel": 10, "actual_channel": track.channel},
                )
            )
        has_fill = any(
            bar.metadata.get("fill")
            or any(
                isinstance(event, NoteEvent) and event.annotations.get("fill")
                for event in bar.events
            )
            for bar in track.bars
        )
        if not has_fill and track.bars:
            issues.append(
                _issue(
                    "warning",
                    "DrumValidator",
                    "missing_fills",
                    f"Drum track {track.id!r} has no fills",
                    track_id=track.id,
                )
            )
        for bar in track.bars:
            for event in bar.events:
                if not isinstance(event, NoteEvent):
                    continue
                midi_note = note_to_midi(event.pitch)
                if midi_note not in DRUM_ALLOWED_MIDI:
                    issues.append(
                        _issue(
                            "error",
                            "DrumValidator",
                            "unsupported_drum_pitch",
                            f"Unsupported drum pitch {event.pitch}",
                            track_id=track.id,
                            bar_number=bar.number,
                            beat=event.start + 1,
                            details={"pitch": event.pitch},
                        )
                    )
    return issues


def _build_report(
    project: ArrangementProject | None,
    issues: list[Issue],
    *,
    metrics: dict[str, Any],
) -> Report:
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    status = "fail" if errors else "pass_with_warnings" if warnings else "pass"
    metrics = {
        **metrics,
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {
        "status": status,
        "project_id": project.project_id if project else None,
        "errors": errors,
        "warnings": warnings,
        "by_track": _issues_by_track(issues),
        "by_bar": _issues_by_bar(issues),
        "metrics": metrics,
    }


def _issue(
    severity: Severity,
    validator: str,
    code: str,
    message: str,
    *,
    track_id: str | None = None,
    bar_number: int | None = None,
    beat: float | None = None,
    voice: int | None = None,
    details: dict[str, Any] | None = None,
) -> Issue:
    return {
        "severity": severity,
        "validator": validator,
        "code": code,
        "message": message,
        "track_id": track_id,
        "bar_number": bar_number,
        "beat": beat,
        "voice": voice,
        "details": details or {},
    }


def _issues_by_track(issues: list[Issue]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for issue in issues:
        track_id = issue.get("track_id") or "<project>"
        entry = grouped.setdefault(track_id, {"errors": 0, "warnings": 0, "issues": []})
        entry["issues"].append(issue)
        if issue["severity"] == "error":
            entry["errors"] += 1
        else:
            entry["warnings"] += 1
    return grouped


def _issues_by_bar(issues: list[Issue]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for issue in issues:
        bar_number = issue.get("bar_number")
        if bar_number is None:
            continue
        key = f"{issue.get('track_id') or '<project>'}:{bar_number}"
        entry = grouped.setdefault(key, {"errors": 0, "warnings": 0, "issues": []})
        entry["issues"].append(issue)
        if issue["severity"] == "error":
            entry["errors"] += 1
        else:
            entry["warnings"] += 1
    return grouped


def _validation_html(report: Report) -> str:
    rows = []
    for issue in [*report.get("errors", []), *report.get("warnings", [])]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(issue['severity'])}</td>"
            f"<td>{html.escape(issue['validator'])}</td>"
            f"<td>{html.escape(str(issue.get('track_id') or ''))}</td>"
            f"<td>{html.escape(str(issue.get('bar_number') or ''))}</td>"
            f"<td>{html.escape(issue['message'])}</td>"
            "</tr>"
        )
    issue_rows = "\n".join(rows) if rows else "<tr><td colspan='5'>No issues</td></tr>"
    status = html.escape(str(report.get("status", "unknown")))
    metrics = html.escape(json.dumps(report.get("metrics", {}), indent=2))
    return (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>Validation Report</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:6px;text-align:left}"
        "pre{background:#f6f6f6;padding:12px}</style></head><body>"
        f"<h1>Validation Report: {status}</h1>"
        "<h2>Issues</h2><table><thead><tr>"
        "<th>Severity</th><th>Validator</th><th>Track</th><th>Bar</th><th>Message</th>"
        f"</tr></thead><tbody>{issue_rows}</tbody></table>"
        f"<h2>Metrics</h2><pre>{metrics}</pre>"
        "</body></html>\n"
    )


def _iter_bar_note_events(
    project: ArrangementProject,
) -> Iterable[tuple[Track, Any, NoteEvent]]:
    for track in project.tracks:
        for bar in track.bars:
            for event in bar.events:
                if isinstance(event, NoteEvent):
                    yield track, bar, event


def _note_event_count(project: ArrangementProject) -> int:
    return sum(1 for _ in _iter_bar_note_events(project))


def _note_count(track: Track) -> int:
    return sum(
        1
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    )


def _avg_note_density(project: ArrangementProject) -> dict[str, float]:
    densities: dict[str, float] = {}
    for track in project.tracks:
        bars = max(1, len(track.bars))
        densities[track.id] = round(_note_count(track) / bars, 3)
    return densities


def _range_midi(note_range: tuple[str, str]) -> tuple[int, int]:
    return note_to_midi(note_range[0]), note_to_midi(note_range[1])


def _instrument_requires_breath(track: Track, catalog: InstrumentCatalog) -> bool:
    try:
        return catalog.get(track.instrument).breath_required
    except KeyError:
        return False


def _chords_by_bar(chord_grid: list[ChordSymbol]) -> dict[int, list[ChordSymbol]]:
    grouped: dict[int, list[ChordSymbol]] = defaultdict(list)
    for chord in chord_grid:
        if chord.bar is not None:
            grouped[chord.bar].append(chord)
    for chords in grouped.values():
        chords.sort(key=lambda chord: chord.beat)
    return dict(grouped)


def _active_chord(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    start: float,
) -> ChordSymbol | None:
    chords = chords_by_bar.get(bar_number, [])
    if not chords:
        return None
    active = chords[0]
    for chord in chords:
        if chord.beat - 1.0 <= start + 1e-6:
            active = chord
        else:
            break
    return active


def _safe_parse(parser: ChordParser, chord_symbol: str) -> ParsedChord | None:
    try:
        return parser.parse(chord_symbol)
    except ValueError:
        return None


def _is_harmonically_supported(event: NoteEvent, parsed: ParsedChord) -> bool:
    pitch_class = note_to_midi(event.pitch) % 12
    allowed = {
        *parsed.chord_tone_pcs,
        *parsed.tension_pcs,
        *parsed.alteration_pcs,
    }
    if pitch_class in allowed:
        return True
    role = str(
        event.annotations.get("melodic_role")
        or event.annotations.get("bass_role")
        or event.annotations.get("horn_role")
        or ""
    )
    return role.startswith("approach") or role in {"walking", "response"}


def _notes_by_slot(events: list[Any]) -> dict[tuple[float, float, int], list[NoteEvent]]:
    grouped: dict[tuple[float, float, int], list[NoteEvent]] = defaultdict(list)
    for event in events:
        if isinstance(event, NoteEvent):
            grouped[(event.start, event.duration, event.voice)].append(event)
    return dict(grouped)


def _manifest_path(files: list[dict[str, Any]], kind: str) -> Path | None:
    for file_record in files:
        if file_record.get("kind") == kind and file_record.get("status") != "skipped":
            return Path(str(file_record.get("path")))
    return None


def _resolve_export_path(output_root: Path, file_record: dict[str, Any]) -> Path:
    path = Path(str(file_record.get("path", "")))
    if path.is_absolute() or path.exists():
        return path
    return output_root / path
