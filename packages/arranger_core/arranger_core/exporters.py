from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import mido
from music21 import (
    articulations,
    converter,
    dynamics,
    expressions,
    harmony,
    instrument,
    metadata,
    meter,
    note,
    stream,
    tempo,
)
from music21 import chord as m21_chord
from music21 import key as m21_key

from arranger_core.catalogs import Instrument as CatalogInstrument
from arranger_core.catalogs import InstrumentCatalog
from arranger_core.music_theory import note_to_midi
from arranger_core.release_gate import ReleaseExportMode, validate_release_quality
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    NoteEvent,
    RestEvent,
    Section,
    Track,
    meter_to_quarter_beats,
)
from arranger_core.validators import (
    MusicValidationError,
    merge_validation_reports,
    validate_export_package,
    validate_project,
    write_validation_html,
    write_validation_json,
)

TICKS_PER_BEAT = 480
DEFAULT_TEMPO_BPM = 120
DRUM_CHANNEL_ZERO_BASED = 9


def export_project(
    project: ArrangementProject,
    output_dir: str | Path,
    *,
    include_pdf: bool = True,
    instrument_catalog: InstrumentCatalog | None = None,
    validation_policy: Literal["strict", "report_only"] = "strict",
    export_mode: ReleaseExportMode | None = None,
) -> dict[str, Any]:
    """Export an ArrangementProject to MIDI, MusicXML and optional PDFs."""

    catalog = instrument_catalog or InstrumentCatalog.load_default()
    export_root = Path(output_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    (export_root / "midi_tracks").mkdir(exist_ok=True)
    (export_root / "parts_pdf").mkdir(exist_ok=True)

    validation_report = validate_project(project, instrument_catalog=catalog)
    project.validation_report = validation_report
    validation_json_path = export_root / "validation_report.json"
    validation_html_path = export_root / "validation_report.html"
    write_validation_json(validation_report, validation_json_path)
    write_validation_html(validation_report, validation_html_path)
    if validation_policy == "strict" and validation_report["status"] == "fail":
        raise MusicValidationError(validation_report)

    files: list[dict[str, Any]] = []
    files.append(
        _write_json_file(project, export_root / "arrangement_project.json", "project_json")
    )
    files.append(
        _write_generation_spec(project.generation_spec, export_root / "generation_spec.json")
    )
    song_plan_record = _write_song_plan(project, export_root / "song_plan.json")
    if song_plan_record is not None:
        files.append(song_plan_record)
    files.append(_file_record(validation_json_path, "validation_report_json"))
    files.append(_file_record(validation_html_path, "validation_report_html"))
    takes_manifest = _build_export_takes_manifest(project, export_root)
    files.append(
        _write_json_mapping(
            takes_manifest,
            export_root / "takes_manifest.json",
            "takes_manifest_json",
        )
    )
    model_trace = _build_model_trace(project, takes_manifest)
    files.append(
        _write_json_mapping(
            model_trace,
            export_root / "model_trace.json",
            "model_trace_json",
        )
    )
    files.append(
        _write_session_readme(
            project,
            export_root / "session_readme.md",
            takes_manifest=takes_manifest,
            model_trace=model_trace,
        )
    )

    midi_full_path = export_root / "full_arrangement.mid"
    write_full_midi(project, midi_full_path, catalog)
    files.append(_file_record(midi_full_path, "midi_full"))

    for track in project.tracks:
        track_midi_path = export_root / "midi_tracks" / f"{_track_file_stem(track)}.mid"
        write_track_midi(project, track, track_midi_path, catalog)
        files.append(_file_record(track_midi_path, "midi_track", track_id=track.id))

    musicxml_path = export_root / "full_score.musicxml"
    write_musicxml_score(project, musicxml_path, catalog)
    files.append(_file_record(musicxml_path, "musicxml_full"))

    pdf_status = "skipped"
    musescore_path = find_musescore_cli() if include_pdf else None
    if include_pdf and musescore_path is not None:
        full_pdf = export_root / "full_score.pdf"
        _run_musescore_export(musescore_path, musicxml_path, full_pdf)
        files.append(_file_record(full_pdf, "pdf_full"))
        pdf_status = "created"

        for track in project.tracks:
            part_pdf = export_root / "parts_pdf" / f"{_track_file_stem(track)}.pdf"
            part_musicxml = export_root / "parts_pdf" / f"{_track_file_stem(track)}.musicxml"
            write_musicxml_score(
                project.model_copy(update={"tracks": [track]}),
                part_musicxml,
                catalog,
            )
            _run_musescore_export(musescore_path, part_musicxml, part_pdf)
            part_musicxml.unlink(missing_ok=True)
            files.append(_file_record(part_pdf, "pdf_part", track_id=track.id))
    elif include_pdf:
        files.append(
            {
                "kind": "pdf_full",
                "path": str(export_root / "full_score.pdf"),
                "status": "skipped",
                "reason": "MuseScore CLI not found",
            }
        )

    manifest = {
        "project_id": project.project_id,
        "schema_version": project.schema_version,
        "exporter_version": "0.1.0",
        "output_dir": str(export_root),
        "status": "exported",
        "pdf_status": pdf_status,
        "musescore_cli": str(musescore_path) if musescore_path else None,
        "files": files,
    }
    project.export_manifest = manifest
    _write_json_mapping(manifest, export_root / "export_manifest.json", "export_manifest")
    files.append(_file_record(export_root / "export_manifest.json", "export_manifest"))
    _write_json_mapping(manifest, export_root / "export_manifest.json", "export_manifest")

    export_validation_report = validate_export_package(project, manifest, export_root)
    validation_report = merge_validation_reports(validation_report, export_validation_report)
    project.validation_report = validation_report
    write_validation_json(validation_report, validation_json_path)
    write_validation_html(validation_report, validation_html_path)
    if validation_policy == "strict" and validation_report["status"] == "fail":
        raise MusicValidationError(validation_report)

    release_quality_report = validate_release_quality(
        project,
        manifest,
        export_root,
        export_mode=export_mode,
    )
    validation_report = merge_validation_reports(validation_report, release_quality_report)
    project.validation_report = validation_report
    write_validation_json(validation_report, validation_json_path)
    write_validation_html(validation_report, validation_html_path)
    if validation_policy == "strict" and release_quality_report["status"] == "fail":
        raise MusicValidationError(validation_report)

    project.save_json(export_root / "arrangement_project.json")
    return manifest


def write_full_midi(
    project: ArrangementProject,
    path: str | Path,
    instrument_catalog: InstrumentCatalog | None = None,
) -> Path:
    catalog = instrument_catalog or InstrumentCatalog.load_default()
    midi_file = _new_midi_file(project)
    channel_allocator = ChannelAllocator()

    for track in project.tracks:
        midi_file.tracks.append(_build_midi_track(project, track, catalog, channel_allocator))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(output_path)
    return output_path


def write_track_midi(
    project: ArrangementProject,
    track: Track,
    path: str | Path,
    instrument_catalog: InstrumentCatalog | None = None,
) -> Path:
    catalog = instrument_catalog or InstrumentCatalog.load_default()
    midi_file = _new_midi_file(project)
    midi_file.tracks.append(_build_midi_track(project, track, catalog, ChannelAllocator()))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(output_path)
    return output_path


def write_musicxml_score(
    project: ArrangementProject,
    path: str | Path,
    instrument_catalog: InstrumentCatalog | None = None,
) -> Path:
    catalog = instrument_catalog or InstrumentCatalog.load_default()
    score = _build_music21_score(project, catalog)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_path = Path(score.write("musicxml", fp=str(output_path)))
    if written_path != output_path and written_path.exists():
        output_path.write_bytes(written_path.read_bytes())
    converter.parse(output_path)
    return output_path


def find_musescore_cli() -> Path | None:
    command_names = ["musescore", "mscore", "mscore3", "mscore4", "MuseScore4", "MuseScore3"]
    for command_name in command_names:
        found = shutil.which(command_name)
        if found:
            return Path(found)

    windows_candidates = [
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"),
        Path(r"C:\Program Files\MuseScore 4\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\MuseScore3.exe"),
    ]
    for candidate in windows_candidates:
        if candidate.exists():
            return candidate
    return None


def _new_midi_file(project: ArrangementProject) -> mido.MidiFile:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))

    tempo_map = sorted(project.tempo_map, key=lambda item: item.bar)
    initial_tempo = tempo_map[0].bpm if tempo_map else _project_tempo(project)
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(initial_tempo), time=0))

    for meter_marker in sorted(project.meter_map, key=lambda item: item.bar):
        numerator, denominator = _split_meter(meter_marker.meter)
        tick = _bar_start_ticks(project, meter_marker.bar)
        conductor.append(
            mido.MetaMessage(
                "time_signature",
                numerator=numerator,
                denominator=denominator,
                time=tick,
            )
        )

    for key_marker in sorted(project.key_map, key=lambda item: item.bar):
        tick = _bar_start_ticks(project, key_marker.bar)
        conductor.append(
            mido.MetaMessage("key_signature", key=_midi_key(key_marker.key), time=tick)
        )

    for section in project.form:
        conductor.append(
            mido.MetaMessage(
                "marker",
                text=section.label or section.name,
                time=_bar_start_ticks(project, section.start_bar),
            )
        )

    _absolutize_midi_track(conductor)
    midi_file.tracks.append(conductor)
    return midi_file


def _build_midi_track(
    project: ArrangementProject,
    track: Track,
    catalog: InstrumentCatalog,
    channel_allocator: ChannelAllocator,
) -> mido.MidiTrack:
    midi_track = mido.MidiTrack()
    midi_track.append(
        mido.MetaMessage("track_name", name=_track_display_name(track, catalog), time=0)
    )

    instrument_info = _instrument_info(track, catalog)
    channel = _track_midi_channel(track, instrument_info, channel_allocator)
    if instrument_info.midi_program is not None and channel != DRUM_CHANNEL_ZERO_BASED:
        midi_track.append(
            mido.Message(
                "program_change",
                channel=channel,
                program=instrument_info.midi_program,
                time=0,
            )
        )

    events: list[tuple[int, int, mido.Message]] = []
    for absolute_start, note_event, window_start, window_end in _iter_absolute_note_events(
        project,
        track,
    ):
        start_tick = round(absolute_start * TICKS_PER_BEAT)
        start_tick += _event_microtiming_ticks(project, note_event)
        start_tick = _clamp_start_tick(start_tick, window_start, window_end)
        duration_tick = max(1, round(note_event.duration * TICKS_PER_BEAT))
        end_tick = min(start_tick + duration_tick, round(window_end * TICKS_PER_BEAT))
        end_tick = max(start_tick + 1, end_tick)
        midi_note = note_to_midi(note_event.pitch)
        events.append(
            (
                start_tick,
                2,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=midi_note,
                    velocity=note_event.velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_tick,
                1,
                mido.Message("note_off", channel=channel, note=midi_note, velocity=0, time=0),
            )
        )

    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = tick
        midi_track.append(message)

    _absolutize_midi_track(midi_track)
    return midi_track


def _build_music21_score(project: ArrangementProject, catalog: InstrumentCatalog) -> stream.Score:
    score = stream.Score(id=project.project_id)
    title = str(project.metadata.get("title", project.project_id))
    score.insert(0, metadata.Metadata(title=title))

    for track in project.tracks:
        score.append(_build_music21_part(project, track, catalog))
    return score


def _build_music21_part(
    project: ArrangementProject,
    track: Track,
    catalog: InstrumentCatalog,
) -> stream.Part:
    part = stream.Part(id=track.id)
    instrument_info = _instrument_info(track, catalog)
    part.partName = _track_display_name(track, catalog)
    part.insert(0, _music21_instrument(track, instrument_info))

    chord_symbols_by_bar = _chord_symbols_by_bar(project.chord_grid)
    for bar_number in range(1, max(1, project.bar_count) + 1):
        bar = _bar_for_number(track, bar_number)
        measure = stream.Measure(number=bar_number)
        meter_text = bar.meter if bar else project.meter_at_bar(bar_number)

        if bar_number == 1 or _meter_changes_at_bar(project, bar_number):
            measure.insert(0, meter.TimeSignature(meter_text))
        if bar_number == 1:
            measure.insert(0, tempo.MetronomeMark(number=_project_tempo(project)))
            key_text = _project_key(project)
            if key_text:
                measure.insert(0, _music21_key(key_text))

        section = _section_starting_at(project.form, bar_number)
        if section:
            measure.insert(0, expressions.RehearsalMark(section.label or section.name))

        measure_chords = [
            *chord_symbols_by_bar.get(bar_number, []),
            *(bar.chords if bar else []),
        ]
        for chord_symbol in measure_chords:
            _insert_chord_symbol(measure, chord_symbol)

        if bar is None or not bar.events:
            full_rest = note.Rest(quarterLength=meter_to_quarter_beats(meter_text))
            measure.insert(0, full_rest)
        else:
            _insert_bar_events(measure, bar.events)

        part.append(measure)
    return part


def _insert_bar_events(measure: stream.Measure, events: list[NoteEvent | RestEvent]) -> None:
    notes_by_slot: dict[tuple[float, float, int], list[NoteEvent]] = defaultdict(list)
    rests: list[RestEvent] = []

    for event in events:
        if isinstance(event, NoteEvent):
            notes_by_slot[(event.start, event.duration, event.voice)].append(event)
        elif isinstance(event, RestEvent):
            rests.append(event)

    for rest_event in rests:
        rest = note.Rest(quarterLength=rest_event.duration)
        measure.insert(rest_event.start, rest)

    for (start, event_duration, _voice), note_events in notes_by_slot.items():
        if len(note_events) == 1:
            source_event = note_events[0]
            m21_note = note.Note(source_event.pitch, quarterLength=event_duration)
            m21_note.volume.velocity = source_event.velocity
            _apply_note_markup(m21_note, source_event)
            _insert_dynamic(measure, start, source_event.dynamic)
            measure.insert(start, m21_note)
        else:
            m21_ch = m21_chord.Chord(
                [event.pitch for event in note_events],
                quarterLength=event_duration,
            )
            if note_events:
                m21_ch.volume.velocity = max(event.velocity for event in note_events)
                _apply_note_markup(m21_ch, note_events[0])
                _insert_dynamic(measure, start, note_events[0].dynamic)
            measure.insert(start, m21_ch)


def _apply_note_markup(music21_note: Any, event: NoteEvent) -> None:
    articulation_factories = {
        "accent": articulations.Accent,
        "staccato": articulations.Staccato,
        "tenuto": articulations.Tenuto,
        "strong_accent": articulations.StrongAccent,
    }
    for articulation_name in event.articulations:
        factory = articulation_factories.get(articulation_name)
        if factory is not None:
            music21_note.articulations.append(factory())


def _insert_dynamic(
    measure: stream.Measure,
    start: float,
    dynamic_text: str | None,
) -> None:
    if dynamic_text:
        measure.insert(start, dynamics.Dynamic(dynamic_text))


def _insert_chord_symbol(measure: stream.Measure, chord_symbol: ChordSymbol) -> None:
    offset = max(0.0, chord_symbol.beat - 1.0)
    try:
        symbol = harmony.ChordSymbol(chord_symbol.symbol)
    except Exception:
        symbol = expressions.TextExpression(chord_symbol.symbol)
    measure.insert(offset, symbol)


def _write_json_file(project: ArrangementProject, path: Path, kind: str) -> dict[str, Any]:
    project.save_json(path)
    return _file_record(path, kind)


def _write_generation_spec(spec: GenerationSpec | None, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = spec.model_dump(mode="json") if spec else {}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return _file_record(path, "generation_spec_json")


def _write_song_plan(project: ArrangementProject, path: Path) -> dict[str, Any] | None:
    song_plan = project.metadata.get("song_plan")
    if not isinstance(song_plan, dict):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(song_plan, indent=2) + "\n", encoding="utf-8")
    return _file_record(path, "song_plan_json")


def _build_export_takes_manifest(
    project: ArrangementProject,
    export_root: Path,
) -> dict[str, Any]:
    source_path = export_root / "takes" / "takes_manifest.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        active_take_id = str(
            project.metadata.get("active_take_id") or source.get("active_take_id") or ""
        )
        source_takes = source.get("takes", [])
        takes = [
            _export_take_record(take)
            for take in source_takes
            if isinstance(take, dict) and take.get("status") == "accepted"
        ]
        if active_take_id and not any(take.get("take_id") == active_take_id for take in takes):
            active = next(
                (
                    _export_take_record(take)
                    for take in source_takes
                    if isinstance(take, dict)
                    and take.get("take_id") == active_take_id
                    and take.get("status") == "accepted"
                ),
                None,
            )
            if active is not None:
                takes.append(active)
        return {
            "schema_version": source.get("schema_version", "0.1.0"),
            "project_id": project.project_id,
            "active_take_id": active_take_id or None,
            "count": len(takes),
            "takes": takes,
            "export_policy": "accepted_only",
            "excluded_statuses": ["pending", "rejected"],
        }

    active_take_id = str(
        project.metadata.get("active_take_id")
        or project.metadata.get("take_id")
        or "take_base"
    )
    source = "model" if project.metadata.get("take_id") else "rule_based"
    return {
        "schema_version": "0.1.0",
        "project_id": project.project_id,
        "active_take_id": active_take_id,
        "count": 1,
        "takes": [
            {
                "take_id": active_take_id,
                "project_id": project.project_id,
                "source": source,
                "status": "accepted",
                "bars": [],
                "artifact_ids": [],
                "metadata": {
                    "validation_status": project.validation_report.get("status"),
                    "take_status": project.metadata.get("take_status", "accepted"),
                },
            }
        ],
        "export_policy": "accepted_only",
        "excluded_statuses": ["pending", "rejected"],
    }


def _export_take_record(take: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "take_id",
        "project_id",
        "parent_take_id",
        "source",
        "backend_id",
        "task",
        "track_id",
        "bars",
        "instruction",
        "seed",
        "status",
        "validation_report_id",
        "artifact_ids",
        "created_at",
        "updated_at",
    )
    record = {key: take[key] for key in allowed_keys if key in take}
    metadata = take.get("metadata", {})
    record["metadata"] = _safe_take_metadata(metadata if isinstance(metadata, dict) else {})
    return record


def _safe_take_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in ("label", "validation_status", "accepted_at"):
        if key in metadata:
            safe[key] = metadata[key]
    trace = metadata.get("model_trace")
    if isinstance(trace, dict):
        safe["model_trace"] = _safe_model_trace_fields(trace)
    return safe


def _safe_model_trace_fields(trace: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "backend",
        "backend_id",
        "task",
        "prompt",
        "instruction",
        "track_id",
        "bars",
        "density",
        "temperature",
        "seed",
        "validation_status",
        "commercial_use",
    )
    return {key: trace[key] for key in allowed_keys if key in trace}


def _build_model_trace(
    project: ArrangementProject,
    takes_manifest: dict[str, Any],
) -> dict[str, Any]:
    takes = [
        take
        for take in takes_manifest.get("takes", [])
        if isinstance(take, dict) and take.get("status") == "accepted"
    ]
    active_take_id = takes_manifest.get("active_take_id")
    active_take = next(
        (take for take in takes if take.get("take_id") == active_take_id),
        None,
    )
    model_artifacts: list[dict[str, Any]] = []
    if active_take is not None and active_take.get("source") == "model":
        model_artifacts.append(_model_trace_record(project, active_take))
    return {
        "schema_version": "0.1.0",
        "project_id": project.project_id,
        "active_take_id": active_take_id,
        "trace_scope": "active_accepted_take",
        "status": "traced" if model_artifacts else "no_model_artifacts",
        "accepted_take_count": len(takes),
        "model_artifacts": model_artifacts,
    }


def _model_trace_record(
    project: ArrangementProject,
    take: dict[str, Any],
) -> dict[str, Any]:
    metadata = take.get("metadata", {})
    trace = metadata.get("model_trace", {}) if isinstance(metadata, dict) else {}
    if not isinstance(trace, dict):
        trace = {}
    backend_id = trace.get("backend_id") or trace.get("backend") or take.get("backend_id")
    project_prompt = project.generation_spec.prompt if project.generation_spec else None
    return {
        "take_id": take.get("take_id"),
        "status": take.get("status"),
        "backend_id": backend_id,
        "task": trace.get("task") or take.get("task"),
        "prompt": trace.get("prompt") or project_prompt,
        "instruction": trace.get("instruction") or take.get("instruction"),
        "track_id": trace.get("track_id") or take.get("track_id"),
        "bars": _int_list(trace.get("bars") or take.get("bars") or []),
        "seed": trace.get("seed") if trace.get("seed") is not None else take.get("seed"),
        "validation_result": (
            trace.get("validation_status")
            or metadata.get("validation_status")
            or project.validation_report.get("status")
        ),
        "commercial_use": trace.get("commercial_use", "unknown"),
    }


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    items: list[int] = []
    for item in value:
        try:
            items.append(int(item))
        except (TypeError, ValueError):
            continue
    return items


def _write_json_mapping(data: dict[str, Any], path: Path, kind: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return _file_record(path, kind)


def _write_session_readme(
    project: ArrangementProject,
    path: Path,
    *,
    takes_manifest: dict[str, Any],
    model_trace: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# AI Arranger Studio DAW Export",
        "",
        f"Project: `{project.project_id}`",
        f"Active take: `{takes_manifest.get('active_take_id') or '-'}`",
        f"Accepted takes in package: `{takes_manifest.get('count', 0)}`",
        f"Model trace status: `{model_trace.get('status', '-')}`",
        "",
        "## Files",
        "",
        "- `full_arrangement.mid`: multitrack MIDI with conductor, tempo, meter and markers.",
        "- `midi_tracks/`: one MIDI file per arrangement track.",
        "- `full_score.musicxml`: full score for notation editors.",
        "- `full_score.pdf`: full score PDF when MuseScore CLI is available.",
        "- `parts_pdf/`: individual part PDFs when MuseScore CLI is available.",
        "- `validation_report.html`: validation summary.",
        "- `model_trace.json`: accepted active model take trace.",
        "- `takes_manifest.json`: accepted takes included in this final export.",
        "- `arrangement_project.json`: canonical project snapshot.",
        "",
        "Raw model artifacts, pending takes and rejected takes are not part of this export.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return _file_record(path, "session_readme")


def _file_record(path: Path, kind: str, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "status": "created",
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    record.update(extra)
    return record


def _build_export_validation_report(project: ArrangementProject) -> dict[str, Any]:
    duration_issues = project.validate_bar_durations()
    return {
        "status": "pass" if not duration_issues else "fail",
        "errors": [issue.model_dump(mode="json") for issue in duration_issues],
        "warnings": [],
        "metrics": {
            "bars": project.bar_count,
            "tracks": len(project.tracks),
            "note_events": sum(
                1 for track in project.tracks for _ in _iter_track_note_events(track)
            ),
        },
    }


def _run_musescore_export(musescore_path: Path, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(musescore_path), "-o", str(output_path), str(input_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _iter_absolute_note_events(
    project: ArrangementProject,
    track: Track,
) -> Iterable[tuple[float, NoteEvent, float, float]]:
    for bar in track.bars:
        bar_start = _bar_start_beats(project, bar.number)
        bar_end = bar_start + meter_to_quarter_beats(project.meter_at_bar(bar.number))
        for event in bar.events:
            if isinstance(event, NoteEvent):
                yield bar_start + event.start, event, bar_start, bar_end

    project_end = _bar_start_beats(project, project.bar_count + 1) if project.bar_count else 0.0
    for event in track.events:
        if isinstance(event, NoteEvent):
            yield event.start, event, 0.0, max(project_end, event.start + event.duration)


def _iter_track_note_events(track: Track) -> Iterable[NoteEvent]:
    for bar in track.bars:
        for event in bar.events:
            if isinstance(event, NoteEvent):
                yield event
    for event in track.events:
        if isinstance(event, NoteEvent):
            yield event


def _bar_start_beats(project: ArrangementProject, bar_number: int) -> float:
    return sum(meter_to_quarter_beats(project.meter_at_bar(bar)) for bar in range(1, bar_number))


def _bar_start_ticks(project: ArrangementProject, bar_number: int) -> int:
    return round(_bar_start_beats(project, bar_number) * TICKS_PER_BEAT)


def _event_microtiming_ticks(project: ArrangementProject, note_event: NoteEvent) -> int:
    microtiming_ms = _event_microtiming_ms(note_event)
    if microtiming_ms == 0:
        return 0
    beat_ms = 60000 / max(1, _project_tempo(project))
    return round((microtiming_ms / beat_ms) * TICKS_PER_BEAT)


def _event_microtiming_ms(note_event: NoteEvent) -> int:
    for key in ("performance_microtiming_ms", "humanized_timing_ms"):
        value = note_event.annotations.get(key)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return 0


def _clamp_start_tick(start_tick: int, window_start: float, window_end: float) -> int:
    low = round(window_start * TICKS_PER_BEAT)
    high = max(low, round(window_end * TICKS_PER_BEAT) - 1)
    return max(low, min(high, start_tick))


def _absolutize_midi_track(midi_track: mido.MidiTrack) -> None:
    previous_tick = 0
    for message in midi_track:
        absolute_tick = message.time
        message.time = max(0, absolute_tick - previous_tick)
        previous_tick = absolute_tick


def _split_meter(meter_text: str) -> tuple[int, int]:
    numerator_text, denominator_text = meter_text.split("/", maxsplit=1)
    return int(numerator_text), int(denominator_text)


def _midi_key(key_text: str) -> str:
    normalized = key_text.strip()
    if normalized.lower().endswith(" minor"):
        return normalized[:-6].strip() + "m"
    if normalized.lower().endswith(" major"):
        return normalized[:-6].strip()
    return normalized


def _music21_key(key_text: str) -> m21_key.Key:
    normalized = key_text.strip()
    if normalized.lower().endswith(" minor"):
        return m21_key.Key(normalized[:-6].strip(), "minor")
    if normalized.lower().endswith(" major"):
        return m21_key.Key(normalized[:-6].strip(), "major")
    return m21_key.Key(normalized)


def _project_tempo(project: ArrangementProject) -> int:
    if project.tempo_map:
        return project.tempo_map[0].bpm
    if project.generation_spec:
        return project.generation_spec.tempo
    return DEFAULT_TEMPO_BPM


def _project_key(project: ArrangementProject) -> str | None:
    if project.key_map:
        return project.key_map[0].key
    if project.generation_spec:
        return project.generation_spec.key
    return None


def _instrument_info(track: Track, catalog: InstrumentCatalog) -> CatalogInstrument:
    try:
        return catalog.get(track.instrument)
    except KeyError:
        return CatalogInstrument(
            id=track.instrument,
            display_name=track.name or track.instrument,
            family="unknown",
            midi_program=0,
            clef="treble",
            transposition_semitones=0,
            sounding_range=("C0", "C8"),
            comfortable_range=("C1", "C7"),
            polyphonic=True,
            breath_required=False,
        )


def _track_midi_channel(
    track: Track,
    instrument_info: CatalogInstrument,
    allocator: ChannelAllocator,
) -> int:
    if track.channel is not None:
        return track.channel - 1
    if instrument_info.midi_channel is not None:
        return instrument_info.midi_channel - 1
    if track.role == "drums" or instrument_info.family == "percussion":
        return DRUM_CHANNEL_ZERO_BASED
    return allocator.next_channel()


def _track_display_name(track: Track, catalog: InstrumentCatalog) -> str:
    if track.name:
        return track.name
    try:
        return catalog.get(track.instrument).display_name
    except KeyError:
        return track.id


def _track_file_stem(track: Track) -> str:
    if track.role == "drums" or track.instrument == "drum_kit":
        return "drums"
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in track.id)
    return safe or "track"


def _bar_for_number(track: Track, bar_number: int) -> Bar | None:
    for bar in track.bars:
        if bar.number == bar_number:
            return bar
    return None


def _chord_symbols_by_bar(chords: list[ChordSymbol]) -> dict[int, list[ChordSymbol]]:
    grouped: dict[int, list[ChordSymbol]] = defaultdict(list)
    for chord_symbol in chords:
        if chord_symbol.bar is not None:
            grouped[chord_symbol.bar].append(chord_symbol)
    return grouped


def _section_starting_at(sections: list[Section], bar_number: int) -> Section | None:
    for section in sections:
        if section.start_bar == bar_number:
            return section
    return None


def _meter_changes_at_bar(project: ArrangementProject, bar_number: int) -> bool:
    return any(marker.bar == bar_number for marker in project.meter_map)


def _music21_instrument(track: Track, instrument_info: CatalogInstrument) -> instrument.Instrument:
    mapping: dict[str, type[instrument.Instrument]] = {
        "piano": instrument.Piano,
        "double_bass": instrument.Contrabass,
        "drum_kit": instrument.UnpitchedPercussion,
        "alto_sax": instrument.AltoSaxophone,
        "tenor_sax": instrument.TenorSaxophone,
        "trumpet_bflat": instrument.Trumpet,
        "trombone": instrument.Trombone,
        "flute": instrument.Flute,
        "clarinet_bflat": instrument.Clarinet,
        "baritone_sax": instrument.BaritoneSaxophone,
        "tuba": instrument.Tuba,
    }
    instrument_cls = mapping.get(track.instrument, instrument.Instrument)
    m21_instrument = instrument_cls()
    m21_instrument.instrumentName = instrument_info.display_name
    if instrument_info.midi_program is not None:
        m21_instrument.midiProgram = instrument_info.midi_program
    return m21_instrument


class ChannelAllocator:
    def __init__(self) -> None:
        self._next = 0

    def next_channel(self) -> int:
        while self._next == DRUM_CHANNEL_ZERO_BASED:
            self._next += 1
        if self._next > 15:
            raise ValueError("MIDI supports at most 15 pitched channels plus channel 10 drums")
        channel = self._next
        self._next += 1
        return channel
