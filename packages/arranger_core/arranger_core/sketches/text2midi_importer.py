from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mido
from pydantic import BaseModel, ConfigDict, Field

from arranger_core.music_theory import midi_to_note
from arranger_core.prompt_compiler import compile_prompt
from arranger_core.schema import (
    ArrangementProject,
    Bar,
    GenerationSpec,
    KeyMark,
    MeterMark,
    NoteEvent,
    RestEvent,
    Section,
    TempoMark,
    Track,
    meter_to_quarter_beats,
)
from arranger_core.takes.models import ModelArtifactRecord
from arranger_core.validators import validate_project

TICKS_PER_BEAT_FALLBACK = 480
SketchStatus = Literal["sketch_validated", "sketch_uncertain", "sketch_rejected"]
DRUM_ALLOWED_MIDI = (36, 38, 42, 44, 45, 47, 49, 50, 51)


class SketchImportError(ValueError):
    pass


class SketchModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SketchTrackClassification(SketchModel):
    track_id: str
    source_name: str
    role: str
    instrument: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class SketchImportResult(SketchModel):
    sketch_id: str
    status: SketchStatus
    project: ArrangementProject
    classifications: list[SketchTrackClassification] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)

    @property
    def role_confidence(self) -> dict[str, float]:
        return {
            classification.track_id: classification.confidence
            for classification in self.classifications
        }


@dataclass
class MidiNote:
    start_ticks: int
    duration_ticks: int
    note: int
    velocity: int
    channel: int


@dataclass
class MidiTrackData:
    index: int
    name: str
    channel: int | None
    notes: list[MidiNote] = field(default_factory=list)
    programs: list[int] = field(default_factory=list)


class Text2MidiSketchImporter:
    def import_record(
        self,
        record: ModelArtifactRecord,
        *,
        prompt: str,
        seed: int | None,
        sketch_id: str,
    ) -> SketchImportResult:
        if record.artifact_type != "midi":
            raise SketchImportError(
                f"Text2MIDI sketch must be a MIDI artifact, got {record.artifact_type}"
            )
        raw_path = Path(record.raw_path)
        if not raw_path.exists():
            raise SketchImportError(f"MIDI sketch artifact does not exist: {raw_path}")
        try:
            midi_file = mido.MidiFile(raw_path)
        except Exception as exc:
            raise SketchImportError(f"MIDI sketch is not parseable: {exc}") from exc

        source_tracks = _midi_track_data(midi_file)
        source_tracks = [track for track in source_tracks if track.notes]
        if not source_tracks:
            raise SketchImportError("MIDI sketch contains no note events")

        meter = _meter_from_midi(midi_file)
        tempo = _tempo_from_midi(midi_file)
        beats_per_bar = meter_to_quarter_beats(meter)
        ticks_per_beat = midi_file.ticks_per_beat or TICKS_PER_BEAT_FALLBACK
        bar_count = _bar_count(
            source_tracks,
            beats_per_bar=beats_per_bar,
            ticks_per_beat=ticks_per_beat,
        )
        generation_spec = _generation_spec(prompt, seed=seed, tempo=tempo, meter=meter)

        tracks: list[Track] = []
        classifications: list[SketchTrackClassification] = []
        used_track_ids: set[str] = set()
        for source_track in source_tracks:
            classification = _classify_track(source_track)
            track_id = _deduped_track_id(
                classification.instrument
                if classification.role != "unknown"
                else f"sketch_track_{source_track.index + 1:02d}",
                used_track_ids,
            )
            classification = classification.model_copy(update={"track_id": track_id})
            classifications.append(classification)
            tracks.append(
                Track(
                    id=track_id,
                    instrument=classification.instrument,
                    role=classification.role,
                    name=source_track.name,
                    channel=source_track.channel + 1 if source_track.channel is not None else None,
                    bars=_track_bars(
                        source_track,
                        bar_count=bar_count,
                        beats_per_bar=beats_per_bar,
                        ticks_per_beat=ticks_per_beat,
                        meter=meter,
                        classification=classification,
                    ),
                    metadata={
                        "source": "text2midi_sketch",
                        "source_track_index": source_track.index,
                        "source_track_name": source_track.name,
                        "role_confidence": classification.confidence,
                        "classification_reasons": classification.reasons,
                    },
                )
            )

        project = ArrangementProject(
            project_id=sketch_id,
            metadata={
                "project_type": "text2midi_sketch",
                "sketch_id": sketch_id,
                "sketch_status": "pending_validation",
                "source": "text2midi",
                "source_backend": record.backend_id,
                "source_artifact_id": record.artifact_id,
                "prompt": prompt,
                "seed": seed,
                "professional_project": False,
                "requires_manual_review": True,
                "auto_merge_allowed": False,
            },
            generation_spec=generation_spec.model_copy(
                update={
                    "duration_bars": bar_count,
                    "tempo": tempo,
                    "meter": meter,
                    "instruments": _instrument_list(classifications),
                    "constraints": {
                        **generation_spec.constraints,
                        "source": "text2midi_sketch",
                        "artifact_id": record.artifact_id,
                        "experimental": True,
                    },
                }
            ),
            tempo_map=[TempoMark(bar=1, bpm=tempo)],
            key_map=[KeyMark(bar=1, key=generation_spec.key)],
            meter_map=[MeterMark(bar=1, meter=meter)],
            form=[Section(name="Sketch", start_bar=1, end_bar=bar_count, label="sketch")],
            chord_grid=[],
            tracks=tracks,
        )

        validation_report = validate_project(project)
        uncertainty_reasons = _uncertainty_reasons(
            classifications,
            bar_count=bar_count,
            tracks=tracks,
            validation_report=validation_report,
        )
        status: SketchStatus = (
            "sketch_rejected"
            if validation_report["status"] == "fail"
            else "sketch_uncertain"
            if uncertainty_reasons
            else "sketch_validated"
        )
        limitations = [
            "sketch_not_final_arrangement",
            "chord_grid_not_inferred",
            "manual_review_required",
        ]
        project.metadata.update(
            {
                "sketch_status": status,
                "role_confidence": {
                    classification.track_id: classification.confidence
                    for classification in classifications
                },
                "uncertainty_reasons": uncertainty_reasons,
                "limitations": limitations,
            }
        )
        project.validation_report = validation_report
        return SketchImportResult(
            sketch_id=sketch_id,
            status=status,
            project=project,
            classifications=classifications,
            uncertainty_reasons=uncertainty_reasons,
            limitations=limitations,
            validation_report=validation_report,
        )


def _midi_track_data(midi_file: mido.MidiFile) -> list[MidiTrackData]:
    output: list[MidiTrackData] = []
    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        name = f"track_{track_index + 1}"
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        notes_by_channel: dict[int, list[MidiNote]] = defaultdict(list)
        programs_by_channel: dict[int, list[int]] = defaultdict(list)
        channels_seen: set[int] = set()
        for message in track:
            absolute_tick += int(message.time)
            if getattr(message, "type", None) == "track_name":
                name = str(message.name)
                continue
            if not hasattr(message, "channel"):
                continue
            channel = int(message.channel)
            channels_seen.add(channel)
            if message.type == "program_change":
                programs_by_channel[channel].append(int(message.program))
                continue
            if message.type == "note_on" and message.velocity > 0:
                active[(channel, int(message.note))].append((absolute_tick, int(message.velocity)))
                continue
            if message.type not in {"note_off", "note_on"}:
                continue
            key = (channel, int(message.note))
            starts = active.get(key)
            if not starts:
                continue
            start_tick, velocity = starts.pop(0)
            notes_by_channel[channel].append(
                MidiNote(
                    start_ticks=start_tick,
                    duration_ticks=max(1, absolute_tick - start_tick),
                    note=int(message.note),
                    velocity=velocity,
                    channel=channel,
                )
            )

        if not notes_by_channel:
            output.append(
                MidiTrackData(
                    index=track_index,
                    name=name,
                    channel=next(iter(channels_seen), None),
                    programs=[],
                )
            )
            continue

        if len(notes_by_channel) == 1:
            channel, notes = next(iter(notes_by_channel.items()))
            output.append(
                MidiTrackData(
                    index=track_index,
                    name=name,
                    channel=channel,
                    notes=sorted(notes, key=lambda item: (item.start_ticks, item.note)),
                    programs=programs_by_channel.get(channel, []),
                )
            )
            continue

        for channel, notes in sorted(notes_by_channel.items()):
            output.append(
                MidiTrackData(
                    index=track_index,
                    name=f"{name} ch{channel + 1}",
                    channel=channel,
                    notes=sorted(notes, key=lambda item: (item.start_ticks, item.note)),
                    programs=programs_by_channel.get(channel, []),
                )
            )
    return output


def _classify_track(track: MidiTrackData) -> SketchTrackClassification:
    normalized = track.name.lower()
    avg_pitch = sum(note.note for note in track.notes) / len(track.notes) if track.notes else 0.0
    program = track.programs[0] if track.programs else None
    reasons: list[str] = []

    if track.channel == 9 or any(
        token in normalized
        for token in ("drum", "percusion", "percussion", "bateria")
    ):
        reasons.append("drum_channel_or_name")
        return _classification(track, "drums", "drum_kit", 0.95, reasons)
    if "bass" in normalized or "contrabajo" in normalized:
        reasons.append("bass_name")
        return _classification(track, "walking_bass", "double_bass", 0.9, reasons)
    if any(token in normalized for token in ("piano", "keys", "keyboard", "comp")):
        reasons.append("keyboard_name")
        return _classification(track, "comping", "piano", 0.85, reasons)
    if "trombone" in normalized:
        reasons.append("trombone_name")
        return _classification(track, "horn_response", "trombone", 0.85, reasons)
    if "trumpet" in normalized or "trompeta" in normalized:
        reasons.append("trumpet_name")
        return _classification(track, "horn_response", "trumpet_bflat", 0.85, reasons)
    if "tenor" in normalized:
        reasons.append("tenor_sax_name")
        return _classification(track, "melody", "tenor_sax", 0.82, reasons)
    if any(token in normalized for token in ("sax", "lead", "melody", "saxo")):
        reasons.append("lead_or_sax_name")
        return _classification(track, "melody", "alto_sax", 0.8, reasons)
    if "clarinet" in normalized:
        reasons.append("clarinet_name")
        return _classification(track, "melody", "clarinet_bflat", 0.8, reasons)
    if "flute" in normalized or "flauta" in normalized:
        reasons.append("flute_name")
        return _classification(track, "melody", "flute", 0.8, reasons)

    if program is not None:
        program_classification = _classify_program(track, program)
        if program_classification is not None:
            return program_classification

    if avg_pitch < 45:
        reasons.append("low_average_pitch")
        return _classification(track, "walking_bass", "double_bass", 0.58, reasons)

    reasons.append("no_role_signal")
    return _classification(track, "unknown", "piano", 0.2, reasons)


def _classify_program(
    track: MidiTrackData,
    program: int,
) -> SketchTrackClassification | None:
    if 0 <= program <= 7:
        return _classification(track, "comping", "piano", 0.72, ["gm_program_keyboard"])
    if 32 <= program <= 39:
        return _classification(track, "walking_bass", "double_bass", 0.72, ["gm_program_bass"])
    if 56 <= program <= 63:
        instrument = "trombone" if program == 57 else "trumpet_bflat"
        return _classification(track, "horn_response", instrument, 0.68, ["gm_program_brass"])
    if 64 <= program <= 67:
        instrument = "tenor_sax" if program == 66 else "alto_sax"
        return _classification(track, "melody", instrument, 0.68, ["gm_program_sax"])
    if 71 <= program <= 73:
        instrument = "flute" if program == 73 else "clarinet_bflat"
        return _classification(track, "melody", instrument, 0.68, ["gm_program_woodwind"])
    return None


def _classification(
    track: MidiTrackData,
    role: str,
    instrument: str,
    confidence: float,
    reasons: list[str],
) -> SketchTrackClassification:
    return SketchTrackClassification(
        track_id="",
        source_name=track.name,
        role=role,
        instrument=instrument,
        confidence=confidence,
        reasons=reasons,
    )


def _track_bars(
    track: MidiTrackData,
    *,
    bar_count: int,
    beats_per_bar: float,
    ticks_per_beat: int,
    meter: str,
    classification: SketchTrackClassification,
) -> list[Bar]:
    notes_by_bar: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in track.notes:
        absolute_beat = note.start_ticks / ticks_per_beat
        duration = max(0.125, note.duration_ticks / ticks_per_beat)
        bar_number = int(absolute_beat // beats_per_bar) + 1
        if bar_number < 1 or bar_number > bar_count:
            continue
        start = absolute_beat - ((bar_number - 1) * beats_per_bar)
        if start >= beats_per_bar:
            continue
        duration = min(duration, beats_per_bar - start)
        pitch_midi = (
            _nearest_drum_pitch(note.note)
            if classification.role == "drums" or classification.instrument == "drum_kit"
            else note.note
        )
        annotations: dict[str, Any] = {
            "source": "text2midi_sketch",
            "source_track_name": track.name,
            "role": classification.role,
            "role_confidence": classification.confidence,
            "imported_model": True,
            "final_material": False,
        }
        if pitch_midi != note.note:
            annotations["source_midi_note"] = note.note
            annotations["normalized_drum_pitch"] = True
        notes_by_bar[bar_number].append(
            NoteEvent(
                pitch=midi_to_note(pitch_midi, prefer_sharps=False),
                start=round(start, 3),
                duration=round(duration, 3),
                velocity=max(1, min(127, note.velocity)),
                annotations=annotations,
            )
        )
    return [
        Bar(
            number=bar_number,
            meter=meter,
            events=_fill_rests(notes_by_bar.get(bar_number, []), beats_per_bar),
        )
        for bar_number in range(1, bar_count + 1)
    ]


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


def _nearest_drum_pitch(value: int) -> int:
    return min(DRUM_ALLOWED_MIDI, key=lambda allowed: (abs(allowed - value), allowed))


def _bar_count(
    tracks: list[MidiTrackData],
    *,
    beats_per_bar: float,
    ticks_per_beat: int,
) -> int:
    last_beat = 0.0
    for track in tracks:
        for note in track.notes:
            last_beat = max(
                last_beat,
                (note.start_ticks + note.duration_ticks) / ticks_per_beat,
            )
    return max(1, int(math.ceil(last_beat / beats_per_bar)))


def _tempo_from_midi(midi_file: mido.MidiFile) -> int:
    for track in midi_file.tracks:
        for message in track:
            if getattr(message, "type", None) == "set_tempo":
                return max(1, min(320, round(mido.tempo2bpm(message.tempo))))
    return 120


def _meter_from_midi(midi_file: mido.MidiFile) -> str:
    for track in midi_file.tracks:
        for message in track:
            if getattr(message, "type", None) == "time_signature":
                return f"{int(message.numerator)}/{int(message.denominator)}"
    return "4/4"


def _generation_spec(
    prompt: str,
    *,
    seed: int | None,
    tempo: int,
    meter: str,
) -> GenerationSpec:
    try:
        spec = compile_prompt(prompt, seed=seed or 0)
    except Exception:
        spec = GenerationSpec(prompt=prompt, seed=seed or 0)
    return spec.model_copy(
        update={
            "prompt": prompt,
            "seed": seed or 0,
            "tempo": tempo,
            "meter": meter,
        }
    )


def _instrument_list(classifications: list[SketchTrackClassification]) -> list[str]:
    instruments = []
    for classification in classifications:
        if classification.instrument not in instruments:
            instruments.append(classification.instrument)
    return instruments


def _uncertainty_reasons(
    classifications: list[SketchTrackClassification],
    *,
    bar_count: int,
    tracks: list[Track],
    validation_report: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    unknown = [item for item in classifications if item.role == "unknown"]
    low_confidence = [item for item in classifications if item.confidence < 0.6]
    if len(unknown) == len(classifications):
        reasons.append("no_roles_detected")
    elif unknown:
        reasons.append("ambiguous_tracks")
    if low_confidence:
        reasons.append("low_role_confidence")
    note_count = int(validation_report.get("metrics", {}).get("note_events", 0))
    if bar_count > 0 and tracks and note_count / max(1, bar_count * len(tracks)) > 24:
        reasons.append("too_dense")
    if validation_report["status"] == "pass_with_warnings":
        reasons.append("validation_warnings")
    return sorted(set(reasons))


def _deduped_track_id(base: str, used: set[str]) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_").lower()
    normalized = normalized or "sketch_track"
    candidate = normalized
    suffix = 2
    while candidate in used:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
