from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from arranger_core import (
    ChordParser,
    NoteEvent,
    PresetLibrary,
    RestEvent,
    Track,
    export_project,
    generate_arrangement,
    note_to_midi,
    validate_project,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "golden"


def build_golden_baseline(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    preset_ids: list[str] | None = None,
    include_pdf: bool = False,
    clean: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if clean:
        _clean_output_dir(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    library = PresetLibrary.load_default()
    selected_presets = (
        [library.get(preset_id) for preset_id in preset_ids]
        if preset_ids
        else library.list_presets()
    )

    preset_summaries: list[dict[str, Any]] = []
    for preset in selected_presets:
        project = generate_arrangement(preset.spec, project_id=preset.id)
        validation_report = validate_project(project)
        if validation_report["errors"]:
            raise RuntimeError(f"{preset.id} validation errors: {validation_report['errors']}")

        preset_output = output_path / preset.id
        export_manifest = export_project(project, preset_output, include_pdf=include_pdf)
        metrics = compute_music_metrics(
            project,
            validation_report=validation_report,
            export_manifest=export_manifest,
            preset_metadata={
                "preset_id": preset.id,
                "preset_name": preset.name,
                "description": preset.description,
            },
        )
        (preset_output / "music_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        preset_summaries.append(_preset_summary(metrics, preset_output))

    summary = {
        "status": "pass",
        "output_dir": str(output_path),
        "preset_count": len(preset_summaries),
        "presets": preset_summaries,
        "aggregate": _aggregate_summary(preset_summaries),
    }
    (output_path / "golden_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "golden_summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def compute_music_metrics(
    project: Any,
    *,
    validation_report: dict[str, Any],
    export_manifest: dict[str, Any] | None = None,
    preset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parser = ChordParser.load_default()
    track_metrics = [
        _track_metrics(project, track, parser)
        for track in project.tracks
    ]
    section_metrics = _section_metrics(project)
    project_note_count = sum(track["note_count"] for track in track_metrics)
    project_bar_count = max(1, project.bar_count)
    quality_flags = _quality_flags(track_metrics, section_metrics, validation_report)

    return {
        "schema_version": "0.1.0",
        "preset": preset_metadata or {},
        "project": {
            "project_id": project.project_id,
            "style": project.generation_spec.style if project.generation_spec else None,
            "form": project.generation_spec.form if project.generation_spec else None,
            "meter": project.generation_spec.meter if project.generation_spec else None,
            "tempo": project.generation_spec.tempo if project.generation_spec else None,
            "bars": project.bar_count,
            "tracks": len(project.tracks),
            "note_events": project_note_count,
            "notes_per_bar": round(project_note_count / project_bar_count, 3),
            "roles": [track.role for track in project.tracks],
        },
        "validation": {
            "status": validation_report["status"],
            "errors": len(validation_report["errors"]),
            "warnings": len(validation_report["warnings"]),
            "harmony_score": validation_report.get("metrics", {}).get("harmony_score"),
        },
        "export": {
            "status": export_manifest.get("status") if export_manifest else None,
            "files": len(export_manifest.get("files", [])) if export_manifest else 0,
            "pdf_status": export_manifest.get("pdf_status") if export_manifest else None,
        },
        "sections": section_metrics,
        "tracks": track_metrics,
        "quality_flags": quality_flags,
        "baseline_rating": _baseline_rating(quality_flags, validation_report),
    }


def _track_metrics(project: Any, track: Track, parser: ChordParser) -> dict[str, Any]:
    notes = _track_notes(track)
    rests = _track_rests(track)
    midi_notes = [_safe_note_to_midi(event.pitch) for event in notes]
    midi_notes = [value for value in midi_notes if value is not None]
    velocities = [event.velocity for event in notes]
    starts = [round(event.start, 3) for event in notes]
    durations = [event.duration for event in notes]
    leaps = [
        abs(right - left)
        for left, right in zip(midi_notes, midi_notes[1:], strict=False)
    ]
    by_bar = Counter(
        bar.number
        for bar in track.bars
        if any(isinstance(event, NoteEvent) for event in bar.events)
    )

    metrics: dict[str, Any] = {
        "track_id": track.id,
        "instrument": track.instrument,
        "role": track.role,
        "bars": len(track.bars),
        "active_bars": len(by_bar),
        "active_bar_ratio": round(len(by_bar) / max(1, project.bar_count), 3),
        "note_count": len(notes),
        "rest_count": len(rests),
        "notes_per_bar": round(len(notes) / max(1, project.bar_count), 3),
        "avg_velocity": round(mean(velocities), 3) if velocities else 0,
        "velocity_stddev": round(pstdev(velocities), 3) if len(velocities) > 1 else 0,
        "avg_duration": round(mean(durations), 3) if durations else 0,
        "unique_onsets": len(set(starts)),
        "pitch_min": min(midi_notes) if midi_notes else None,
        "pitch_max": max(midi_notes) if midi_notes else None,
        "pitch_span": max(midi_notes) - min(midi_notes) if midi_notes else 0,
        "unique_pitches": len(set(midi_notes)),
        "max_leap": max(leaps) if leaps else 0,
        "large_leaps": sum(1 for leap in leaps if leap > 12),
    }

    if track.role == "walking_bass":
        metrics.update(_bass_metrics(project, track, parser))
    if track.instrument == "piano" or track.role in {"comping", "piano"}:
        metrics.update(_piano_metrics(track))
    if track.role == "drums" or track.instrument == "drum_kit":
        metrics.update(_drum_metrics(track))
    if track.role in {"melody", "horn_response"}:
        metrics.update(_phrase_metrics(track))
    return metrics


def _bass_metrics(project: Any, track: Track, parser: ChordParser) -> dict[str, Any]:
    chords_by_bar = _chords_by_bar(project.chord_grid)
    checked_roots = 0
    root_hits = 0
    approach_hits = 0
    checked_approaches = 0
    for bar in track.bars:
        notes = [event for event in bar.events if isinstance(event, NoteEvent)]
        if not notes:
            continue
        chord = _active_chord(chords_by_bar, bar.number, 0.0)
        if chord is not None:
            parsed = _safe_parse(parser, chord.symbol)
            if parsed is not None:
                checked_roots += 1
                if note_to_midi(notes[0].pitch) % 12 == parsed.root_pc:
                    root_hits += 1

        next_chord = _first_chord_after(chords_by_bar, bar.number, project.bar_count)
        if next_chord is not None and notes:
            parsed_next = _safe_parse(parser, next_chord.symbol)
            if parsed_next is not None:
                checked_approaches += 1
                distance = _pc_distance(note_to_midi(notes[-1].pitch), parsed_next.root_pc)
                if distance <= 2:
                    approach_hits += 1

    return {
        "beat1_root_score": round(root_hits / checked_roots, 3) if checked_roots else None,
        "approach_to_next_root_score": (
            round(approach_hits / checked_approaches, 3) if checked_approaches else None
        ),
    }


def _piano_metrics(track: Track) -> dict[str, Any]:
    voicing_sizes: list[int] = []
    rootless_violations = 0
    for bar in track.bars:
        grouped: dict[tuple[float, int], list[NoteEvent]] = defaultdict(list)
        for event in bar.events:
            if isinstance(event, NoteEvent):
                grouped[(event.start, event.voice)].append(event)
        for events in grouped.values():
            if len(events) > 1:
                voicing_sizes.append(len(events))
            for event in events:
                root_pc = event.annotations.get("root_pc")
                if root_pc is not None and note_to_midi(event.pitch) % 12 == root_pc:
                    rootless_violations += 1
    return {
        "avg_voicing_size": round(mean(voicing_sizes), 3) if voicing_sizes else 0,
        "max_voicing_size": max(voicing_sizes) if voicing_sizes else 0,
        "rootless_violations": rootless_violations,
    }


def _drum_metrics(track: Track) -> dict[str, Any]:
    fill_bars = [bar.number for bar in track.bars if bar.metadata.get("fill")]
    drum_pitches = Counter(
        event.pitch
        for event in _track_notes(track)
    )
    return {
        "fill_bars": fill_bars,
        "fill_bar_count": len(fill_bars),
        "drum_pitch_count": len(drum_pitches),
        "most_common_drum_pitch": drum_pitches.most_common(1)[0][0] if drum_pitches else None,
    }


def _phrase_metrics(track: Track) -> dict[str, Any]:
    rests = _track_rests(track)
    breath_rests = [rest for rest in rests if rest.duration >= 0.5]
    active_bars = [
        bar.number
        for bar in track.bars
        if any(isinstance(event, NoteEvent) for event in bar.events)
    ]
    return {
        "breath_rest_count": len(breath_rests),
        "silent_bar_count": len(track.bars) - len(set(active_bars)),
    }


def _section_metrics(project: Any) -> list[dict[str, Any]]:
    metrics = []
    for section in project.form:
        bars = set(range(section.start_bar, section.end_bar + 1))
        note_count = sum(
            1
            for track in project.tracks
            for bar in track.bars
            if bar.number in bars
            for event in bar.events
            if isinstance(event, NoteEvent)
        )
        metrics.append(
            {
                "name": section.name,
                "label": section.label,
                "start_bar": section.start_bar,
                "end_bar": section.end_bar,
                "bars": section.duration_bars,
                "note_count": note_count,
                "notes_per_bar": round(note_count / max(1, section.duration_bars), 3),
            }
        )
    return metrics


def _quality_flags(
    track_metrics: list[dict[str, Any]],
    section_metrics: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if validation_report["warnings"]:
        flags.append(
            {
                "severity": "medium",
                "code": "validation_warnings",
                "message": "El proyecto tiene warnings musicales o de exportacion.",
            }
        )

    section_densities = [section["notes_per_bar"] for section in section_metrics]
    if len(section_densities) > 1 and pstdev(section_densities) < 2.0:
        flags.append(
            {
                "severity": "high",
                "code": "flat_section_energy",
                "message": "La densidad por seccion es muy plana; falta narrativa.",
            }
        )

    for metrics in track_metrics:
        if metrics["role"] == "drums" and metrics.get("fill_bar_count", 0) <= 1:
            flags.append(
                {
                    "severity": "medium",
                    "code": "drums_low_fill_language",
                    "message": f"{metrics['track_id']} tiene pocos fills o setups.",
                }
            )
        if metrics["role"] == "walking_bass":
            approach_score = metrics.get("approach_to_next_root_score")
            if approach_score is not None and approach_score < 0.45:
                flags.append(
                    {
                        "severity": "high",
                        "code": "bass_weak_approaches",
                        "message": "El bajo conecta poco con la raiz del siguiente acorde.",
                    }
                )
        if metrics["role"] in {"comping", "piano"} and metrics["notes_per_bar"] > 10:
            flags.append(
                {
                    "severity": "medium",
                    "code": "comping_dense",
                    "message": f"{metrics['track_id']} puede saturar el arreglo.",
                }
            )
        if metrics["role"] == "melody" and metrics.get("breath_rest_count", 0) < 4:
            flags.append(
                {
                    "severity": "medium",
                    "code": "melody_low_breathing",
                    "message": f"{metrics['track_id']} tiene poca respiracion.",
                }
            )
    return flags


def _baseline_rating(
    quality_flags: list[dict[str, str]],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    score = 5.0
    score -= len(validation_report["errors"]) * 2.0
    score -= len([flag for flag in quality_flags if flag["severity"] == "high"]) * 0.75
    score -= len([flag for flag in quality_flags if flag["severity"] == "medium"]) * 0.35
    score = max(1.0, min(5.0, score))
    return {
        "estimated_score_1_to_5": round(score, 2),
        "note": "Heuristic baseline score for prioritization, not a listening verdict.",
    }


def _preset_summary(metrics: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "preset_id": metrics["preset"].get("preset_id"),
        "style": metrics["project"]["style"],
        "form": metrics["project"]["form"],
        "bars": metrics["project"]["bars"],
        "tracks": metrics["project"]["tracks"],
        "note_events": metrics["project"]["note_events"],
        "validation_status": metrics["validation"]["status"],
        "quality_flags": len(metrics["quality_flags"]),
        "estimated_score_1_to_5": metrics["baseline_rating"]["estimated_score_1_to_5"],
        "output_dir": str(output_dir),
    }


def _aggregate_summary(presets: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [preset["estimated_score_1_to_5"] for preset in presets]
    return {
        "avg_estimated_score": round(mean(scores), 3) if scores else 0,
        "min_estimated_score": min(scores) if scores else 0,
        "max_estimated_score": max(scores) if scores else 0,
        "total_quality_flags": sum(preset["quality_flags"] for preset in presets),
        "total_note_events": sum(preset["note_events"] for preset in presets),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Golden baseline summary",
        "",
        f"- Presets: {summary['preset_count']}",
        f"- Average heuristic score: {summary['aggregate']['avg_estimated_score']}",
        f"- Quality flags: {summary['aggregate']['total_quality_flags']}",
        "",
        "| Preset | Score | Flags | Validation | Output |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for preset in summary["presets"]:
        lines.append(
            "| "
            f"{preset['preset_id']} | "
            f"{preset['estimated_score_1_to_5']} | "
            f"{preset['quality_flags']} | "
            f"{preset['validation_status']} | "
            f"{preset['output_dir']} |"
        )
    return "\n".join(lines) + "\n"


def _track_notes(track: Track) -> list[NoteEvent]:
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]


def _track_rests(track: Track) -> list[RestEvent]:
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, RestEvent)
    ]


def _chords_by_bar(chords: list[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for chord in chords:
        if chord.bar is not None:
            grouped.setdefault(chord.bar, []).append(chord)
    for bar_chords in grouped.values():
        bar_chords.sort(key=lambda chord: chord.beat)
    return grouped


def _active_chord(chords_by_bar: dict[int, list[Any]], bar_number: int, start: float) -> Any | None:
    chords = chords_by_bar.get(bar_number, [])
    if not chords:
        return None
    active = chords[0]
    for chord in chords:
        if chord.beat - 1 <= start + 1e-6:
            active = chord
    return active


def _first_chord_after(
    chords_by_bar: dict[int, list[Any]],
    bar_number: int,
    max_bar: int,
) -> Any | None:
    for next_bar in range(bar_number + 1, max_bar + 1):
        chords = chords_by_bar.get(next_bar, [])
        if chords:
            return chords[0]
    return _active_chord(chords_by_bar, bar_number, 0.0)


def _safe_parse(parser: ChordParser, symbol: str) -> Any | None:
    try:
        return parser.parse(symbol)
    except ValueError:
        return None


def _safe_note_to_midi(pitch: str) -> int | None:
    try:
        return note_to_midi(pitch)
    except ValueError:
        return None


def _pc_distance(midi_note: int, target_pc: int) -> int:
    current_pc = midi_note % 12
    direct = abs(current_pc - target_pc)
    return min(direct, 12 - direct)


def _clean_output_dir(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_outputs = OUTPUTS_ROOT.resolve()
    if resolved_path == resolved_outputs or not resolved_path.is_relative_to(resolved_outputs):
        raise RuntimeError(f"Refusing to clean path outside outputs/: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden demos and baseline metrics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset-id", action="append", default=None)
    parser.add_argument(
        "--include-pdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate PDFs when MuseScore CLI is available.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the output directory first. Cleaning is limited to outputs/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_golden_baseline(
        output_dir=args.output_dir,
        preset_ids=args.preset_id,
        include_pdf=args.include_pdf,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
