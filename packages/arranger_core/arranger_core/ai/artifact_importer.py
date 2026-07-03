from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mido
from pydantic import BaseModel, ConfigDict, Field

from arranger_core.ai.artifact_store import ArtifactStore
from arranger_core.music_theory import midi_to_note
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    NoteEvent,
    RestEvent,
    Track,
    meter_to_quarter_beats,
)
from arranger_core.takes.models import ModelArtifactRecord

TICKS_PER_BEAT_FALLBACK = 480
MODEL_IMPORT_QUANTIZE_GRID = 0.25


class ArtifactImportError(ValueError):
    pass


class ImportedModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    project_id: str | None
    backend_id: str
    task: str
    artifact_type: str
    track_id: str | None = None
    bars: list[int] = Field(default_factory=list)
    track: Track | None = None
    imported_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def note_count(self) -> int:
        if self.track is None:
            return 0
        return sum(
            1
            for bar in self.track.bars
            for event in bar.events
            if isinstance(event, NoteEvent)
        )


class ArtifactImporter:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore | None = None,
        imported_root: str | Path | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        if imported_root is None and artifact_store is not None:
            imported_root = artifact_store.root / "imported"
        self.imported_root = Path(imported_root or "outputs/model_artifacts/imported")
        self.imported_root.mkdir(parents=True, exist_ok=True)

    def import_record(
        self,
        record: ModelArtifactRecord,
        *,
        project: ArrangementProject | None = None,
        target_track_id: str | None = None,
        target_bars: list[int] | None = None,
    ) -> ImportedModelArtifact:
        try:
            imported = self._import_record(
                record,
                project=project,
                target_track_id=target_track_id,
                target_bars=target_bars,
            )
        except Exception as exc:
            if self.artifact_store is not None:
                self.artifact_store.mark_rejected(record, reason=str(exc))
            if isinstance(exc, ArtifactImportError):
                raise
            raise ArtifactImportError(str(exc)) from exc

        imported_path = self.imported_root / f"{record.artifact_id}.json"
        imported = imported.model_copy(update={"imported_path": str(imported_path)})
        imported_path.write_text(
            imported.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        if self.artifact_store is not None:
            self.artifact_store.mark_imported(
                record,
                imported_path=imported_path,
                metadata={"note_count": imported.note_count},
            )
        return imported

    def _import_record(
        self,
        record: ModelArtifactRecord,
        *,
        project: ArrangementProject | None,
        target_track_id: str | None,
        target_bars: list[int] | None,
    ) -> ImportedModelArtifact:
        raw_path = Path(record.raw_path)
        if record.artifact_type == "json":
            return self._import_json(record, raw_path)
        if record.artifact_type != "midi":
            raise ArtifactImportError(f"Unsupported artifact type: {record.artifact_type}")
        if not raw_path.exists():
            raise ArtifactImportError(f"Artifact file does not exist: {raw_path}")

        try:
            midi_file = mido.MidiFile(raw_path)
        except Exception as exc:
            raise ArtifactImportError(f"MIDI artifact is not parseable: {exc}") from exc

        bars = target_bars or record.metadata.get("bars") or [1]
        bars = [int(bar) for bar in bars]
        track_id = target_track_id or record.metadata.get("track_id") or "model_track"
        template_track = _find_track(project, track_id) if project is not None else None
        imported_track = Track(
            id=track_id,
            instrument=template_track.instrument if template_track else "piano",
            role=template_track.role if template_track else "model_import",
            name=template_track.name if template_track else track_id,
            channel=template_track.channel if template_track else None,
            bars=_events_to_bars(
                midi_file,
                project=project,
                target_bars=bars,
            ),
            metadata={
                "source": "model_artifact",
                "artifact_id": record.artifact_id,
                "backend_id": record.backend_id,
                "task": record.task,
            },
        )
        return ImportedModelArtifact(
            artifact_id=record.artifact_id,
            project_id=record.project_id,
            backend_id=record.backend_id,
            task=record.task,
            artifact_type=record.artifact_type,
            track_id=track_id,
            bars=bars,
            track=imported_track,
            metadata={"raw_path": record.raw_path},
        )

    def _import_json(self, record: ModelArtifactRecord, raw_path: Path) -> ImportedModelArtifact:
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactImportError(f"JSON artifact is not parseable: {exc}") from exc
        return ImportedModelArtifact(
            artifact_id=record.artifact_id,
            project_id=record.project_id,
            backend_id=record.backend_id,
            task=record.task,
            artifact_type=record.artifact_type,
            metadata={"payload": payload, "raw_path": record.raw_path},
        )


def _events_to_bars(
    midi_file: mido.MidiFile,
    *,
    project: ArrangementProject | None,
    target_bars: list[int],
) -> list[Bar]:
    notes = _midi_notes(midi_file)
    notes_by_bar: dict[int, list[NoteEvent]] = defaultdict(list)
    ticks_per_beat = midi_file.ticks_per_beat or TICKS_PER_BEAT_FALLBACK
    first_bar = min(target_bars)

    for note in notes:
        absolute_beat = note["start_ticks"] / ticks_per_beat
        duration = max(0.25, note["duration_ticks"] / ticks_per_beat)
        bar_number, start = _beat_to_target_bar(
            absolute_beat,
            project=project,
            target_bars=target_bars,
        )
        if bar_number not in target_bars:
            continue
        expected = _bar_duration(project, bar_number)
        if start >= expected:
            continue
        duration = min(duration, expected - start)
        start, duration = _quantize_note_window(
            start,
            duration,
            expected=expected,
            grid=MODEL_IMPORT_QUANTIZE_GRID,
        )
        if duration <= 0:
            continue
        notes_by_bar[bar_number].append(
            NoteEvent(
                pitch=midi_to_note(int(note["note"]), prefer_sharps=False),
                start=round(start, 3),
                duration=round(duration, 3),
                velocity=max(1, min(127, int(note["velocity"]))),
                annotations={
                    "source": "model_artifact",
                    "imported_bar_offset": bar_number - first_bar,
                },
            )
        )

    return [
        Bar(
            number=bar_number,
            meter=project.meter_at_bar(bar_number) if project is not None else None,
            events=_fill_rests(
                notes_by_bar.get(bar_number, []),
                _bar_duration(project, bar_number),
            ),
        )
        for bar_number in target_bars
    ]


def _quantize_note_window(
    start: float,
    duration: float,
    *,
    expected: float,
    grid: float,
) -> tuple[float, float]:
    if expected <= 0:
        return 0.0, 0.0
    quantized_start = round(start / grid) * grid
    quantized_start = max(0.0, min(quantized_start, max(0.0, expected - grid)))
    raw_end = min(expected, start + duration)
    quantized_end = round(raw_end / grid) * grid
    quantized_end = max(quantized_start + grid, quantized_end)
    quantized_end = min(expected, quantized_end)
    return round(quantized_start, 3), round(max(0.0, quantized_end - quantized_start), 3)


def _midi_notes(midi_file: mido.MidiFile) -> list[dict[str, int]]:
    notes: list[dict[str, int]] = []
    for track in midi_file.tracks:
        absolute_tick = 0
        active: dict[tuple[int, int], tuple[int, int]] = {}
        for message in track:
            absolute_tick += int(message.time)
            if not hasattr(message, "type"):
                continue
            if message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)] = (absolute_tick, message.velocity)
            elif message.type in {"note_off", "note_on"}:
                key = (message.channel, message.note)
                if key not in active:
                    continue
                start_tick, velocity = active.pop(key)
                duration_ticks = max(1, absolute_tick - start_tick)
                notes.append(
                    {
                        "start_ticks": start_tick,
                        "duration_ticks": duration_ticks,
                        "note": message.note,
                        "velocity": velocity,
                    }
                )
    notes.sort(key=lambda item: (item["start_ticks"], item["note"]))
    return notes


def _beat_to_target_bar(
    absolute_beat: float,
    *,
    project: ArrangementProject | None,
    target_bars: list[int],
) -> tuple[int, float]:
    elapsed = 0.0
    for bar_number in target_bars:
        duration = _bar_duration(project, bar_number)
        if absolute_beat < elapsed + duration:
            return bar_number, absolute_beat - elapsed
        elapsed += duration
    return (
        target_bars[-1],
        absolute_beat - max(0.0, elapsed - _bar_duration(project, target_bars[-1])),
    )


def _fill_rests(notes: list[NoteEvent], expected: float) -> list[NoteEvent | RestEvent]:
    events: list[NoteEvent | RestEvent] = []
    cursor = 0.0
    for note in sorted(notes, key=lambda item: (item.start, item.pitch)):
        if note.start > cursor + 1e-6:
            events.append(RestEvent(start=round(cursor, 3), duration=round(note.start - cursor, 3)))
        events.append(note)
        cursor = max(cursor, note.start + note.duration)
    if cursor < expected - 1e-6:
        events.append(RestEvent(start=round(cursor, 3), duration=round(expected - cursor, 3)))
    if not events:
        events.append(RestEvent(start=0.0, duration=round(expected, 3)))
    return events


def _bar_duration(project: ArrangementProject | None, bar_number: int) -> float:
    if project is None:
        return 4.0
    return meter_to_quarter_beats(project.meter_at_bar(bar_number))


def _find_track(project: ArrangementProject | None, track_id: str) -> Track | None:
    if project is None:
        return None
    return next((track for track in project.tracks if track.id == track_id), None)
