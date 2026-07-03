from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import mido
import yaml
from music21 import converter

from arranger_core.catalogs import InstrumentCatalog
from arranger_core.chords import ChordParser, ParsedChord
from arranger_core.music_theory import note_to_midi
from arranger_core.release_gate import BLOCKED_LICENSES
from arranger_core.schema import ArrangementProject, Bar, ChordSymbol, NoteEvent, RestEvent, Track
from arranger_core.validators import validate_project

PRO_QUALITY_GATE_VERSION = "0.1.0"
RATING_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}
TRACK_ID_ALIASES = {
    "drums": ("drum_kit",),
    "drum_kit": ("drums",),
    "trumpet": ("trumpet_bflat",),
    "trumpet_bflat": ("trumpet",),
}
SKETCH_ONLY_TASKS = {"generate_full_sketch", "sketch_reference", "text2midi_sketch"}


class ProQualityGate:
    """Professional release gate for generated symbolic MIDI arrangements."""

    def __init__(
        self,
        *,
        thresholds: dict[str, Any] | None = None,
        thresholds_path: str | Path | None = None,
        instrument_catalog: InstrumentCatalog | None = None,
        chord_parser: ChordParser | None = None,
    ) -> None:
        self.thresholds = thresholds or _load_thresholds(thresholds_path)
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()
        self.chord_parser = chord_parser or ChordParser.load_default()

    def evaluate(
        self,
        project: ArrangementProject,
        *,
        validation_report: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        export_manifest: dict[str, Any] | None = None,
        model_trace: dict[str, Any] | None = None,
        takes_manifest: dict[str, Any] | None = None,
        export_mode: Literal["private", "commercial"] = "private",
        min_rating: Literal["A", "B", "C", "D"] = "B",
        required_tracks: list[str] | None = None,
        require_export_files: bool | None = None,
    ) -> dict[str, Any]:
        output_root = Path(output_dir) if output_dir is not None else None
        if require_export_files is None:
            require_export_files = output_root is not None

        validation = validation_report or project.validation_report or validate_project(project)
        export = export_manifest or _read_json(
            output_root / "export_manifest.json" if output_root else None
        )
        trace = model_trace or _read_json(
            output_root / "model_trace.json" if output_root else None
        )
        takes = takes_manifest or _read_json(
            output_root / "takes_manifest.json" if output_root else None
        )

        metrics = _quality_metrics(
            project,
            validation=validation,
            model_trace=trace,
            takes_manifest=takes,
            export_manifest=export,
            output_dir=output_root,
            chord_parser=self.chord_parser,
            instrument_catalog=self.instrument_catalog,
            required_tracks=required_tracks or [],
        )
        score = 1.0
        issues: list[dict[str, Any]] = []

        score = _apply_global_checks(
            project,
            metrics=metrics,
            validation=validation,
            thresholds=self.thresholds,
            required_tracks=required_tracks or [],
            issues=issues,
            score=score,
        )
        score = _apply_role_checks(
            metrics=metrics,
            thresholds=self.thresholds,
            issues=issues,
            score=score,
        )
        score = _apply_model_checks(
            metrics=metrics,
            model_trace=trace,
            takes_manifest=takes,
            export_mode=export_mode,
            thresholds=self.thresholds,
            issues=issues,
            score=score,
        )
        if require_export_files:
            score = _apply_export_checks(
                project,
                output_root=output_root,
                export_manifest=export,
                thresholds=self.thresholds,
                issues=issues,
                score=score,
            )

        score = max(0.0, min(1.0, score))
        rating = _rating_for_score(score, self.thresholds.get("ratings", {}))
        if RATING_ORDER[rating] < RATING_ORDER[min_rating]:
            issues.append(
                _issue(
                    "error",
                    "rating_below_minimum",
                    f"Quality rating {rating} is below required minimum {min_rating}",
                    details={"rating": rating, "min_rating": min_rating},
                )
            )

        errors = [issue for issue in issues if issue["severity"] == "error"]
        warnings = [issue for issue in issues if issue["severity"] == "warning"]
        return {
            "schema_version": "0.1.0",
            "gate_version": PRO_QUALITY_GATE_VERSION,
            "status": "fail" if errors else "pass",
            "score": round(score, 3),
            "rating": rating,
            "min_rating": min_rating,
            "release_candidate": not errors and rating == "A",
            "blocking_errors": [
                f"{issue['code']}: {issue['message']}" for issue in errors
            ],
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }


def _load_thresholds(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _default_thresholds()
    threshold_path = Path(path)
    if not threshold_path.exists():
        return _default_thresholds()
    payload = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else _default_thresholds()


def _default_thresholds() -> dict[str, Any]:
    return {
        "global": {
            "max_blocking_errors": 0,
            "min_tracks": 3,
            "min_note_events": 80,
            "require_full_midi": True,
            "require_musicxml": True,
            "require_model_trace": True,
            "reject_pending_takes_on_export": True,
        },
        "bass": {
            "min_beat1_root_score": 0.55,
            "min_approach_to_next_root_score": 0.35,
            "max_large_leaps": 8,
            "min_active_bar_ratio": 0.70,
        },
        "piano": {
            "max_rootless_violations": 24,
            "min_avg_voicing_size": 2.0,
            "max_voicing_size": 6,
            "max_low_register_notes_below_midi": 40,
        },
        "drums": {
            "min_drum_pitch_count": 3,
            "min_fill_bar_count": 1,
            "min_velocity_stddev": 2.0,
        },
        "melody": {
            "min_breath_rest_count": 2,
            "max_large_leaps": 10,
            "min_active_bar_ratio": 0.35,
        },
        "horns": {
            "min_breath_rest_count": 1,
            "max_large_leaps": 12,
            "max_density_per_bar": 8,
        },
        "ratings": {
            "A": {"min_score": 0.88},
            "B": {"min_score": 0.72},
            "C": {"min_score": 0.55},
            "D": {"min_score": 0.0},
        },
    }


def _quality_metrics(
    project: ArrangementProject,
    *,
    validation: dict[str, Any],
    model_trace: dict[str, Any],
    takes_manifest: dict[str, Any],
    export_manifest: dict[str, Any],
    output_dir: Path | None,
    chord_parser: ChordParser,
    instrument_catalog: InstrumentCatalog,
    required_tracks: list[str],
) -> dict[str, Any]:
    track_metrics = [
        _track_quality_metrics(
            project,
            track,
            chord_parser=chord_parser,
            instrument_catalog=instrument_catalog,
        )
        for track in project.tracks
    ]
    available_tracks = {track.id for track in project.tracks}
    present_required, missing_required = _required_track_status(required_tracks, available_tracks)
    return {
        "project": {
            "project_id": project.project_id,
            "bars": project.bar_count,
            "tracks": len(project.tracks),
            "note_events": sum(track["note_count"] for track in track_metrics),
            "bar_duration_issues": len(project.validate_bar_durations()),
            "required_tracks_present": present_required,
            "missing_required_tracks": missing_required,
        },
        "validation": {
            "status": validation.get("status"),
            "errors": len(validation.get("errors", [])),
            "warnings": len(validation.get("warnings", [])),
        },
        "tracks": track_metrics,
        "roles": _role_summary(track_metrics),
        "model_trace": _model_trace_metrics(model_trace),
        "takes": _takes_metrics(takes_manifest),
        "export": _export_metrics(export_manifest, output_dir),
        "datasets": _dataset_metrics(project),
    }


def _track_quality_metrics(
    project: ArrangementProject,
    track: Track,
    *,
    chord_parser: ChordParser,
    instrument_catalog: InstrumentCatalog,
) -> dict[str, Any]:
    notes_by_bar = _notes_by_bar(track)
    rests = [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, RestEvent)
    ]
    notes = [event for bar_notes in notes_by_bar.values() for event in bar_notes]
    active_bars = sorted(bar_number for bar_number, bar_notes in notes_by_bar.items() if bar_notes)
    midi_notes = [_safe_note_to_midi(note.pitch) for note in notes]
    midi_notes = [midi for midi in midi_notes if midi is not None]
    large_leaps = _large_leaps(midi_notes, limit=_leap_limit(track))
    comfortable_out_count = _comfortable_range_out_count(track, notes, instrument_catalog)
    voicing_metrics = _piano_voicing_metrics(track)
    drum_metrics = _drum_metrics(track, notes)
    breath_rest_count = sum(1 for rest in rests if rest.duration >= 0.5)
    density_per_bar = len(notes) / max(1, project.bar_count)
    base = {
        "track_id": track.id,
        "role": track.role,
        "instrument": track.instrument,
        "channel": track.channel,
        "note_count": len(notes),
        "rest_count": len(rests),
        "breath_rest_count": breath_rest_count,
        "active_bars": active_bars,
        "active_bar_ratio": round(len(active_bars) / max(1, project.bar_count), 3),
        "density_per_bar": round(density_per_bar, 3),
        "large_leaps": large_leaps,
        "min_midi": min(midi_notes) if midi_notes else None,
        "max_midi": max(midi_notes) if midi_notes else None,
        "comfortable_out_count": comfortable_out_count,
        "bar_count": len(track.bars),
        "missing_bars": _missing_bars(project, track),
    }
    if _is_bass_track(track):
        base["bass"] = _bass_metrics(project, track, chord_parser)
    if _is_piano_track(track):
        base["piano"] = voicing_metrics
    if _is_drum_track(track):
        base["drums"] = drum_metrics
    if _is_melodic_track(track) or _is_horn_track(track):
        base["melody"] = {
            "cadence_resolution_score": _cadence_resolution_score(
                project,
                track,
                chord_parser,
            )
        }
    return base


def _apply_global_checks(
    project: ArrangementProject,
    *,
    metrics: dict[str, Any],
    validation: dict[str, Any],
    thresholds: dict[str, Any],
    required_tracks: list[str],
    issues: list[dict[str, Any]],
    score: float,
) -> float:
    global_thresholds = thresholds.get("global", {})
    max_errors = int(global_thresholds.get("max_blocking_errors", 0))
    if metrics["validation"]["errors"] > max_errors or validation.get("status") == "fail":
        score = _penalize(
            issues,
            score,
            severity="error",
            code="blocking_validation_errors",
            message="Project validation contains blocking errors",
            amount=0.45,
            details={
                "errors": metrics["validation"]["errors"],
                "max_errors": max_errors,
            },
        )
    if metrics["project"]["bar_duration_issues"]:
        score = _penalize(
            issues,
            score,
            severity="error",
            code="incomplete_bars",
            message="One or more bars do not fill their expected duration",
            amount=0.35,
            details={"count": metrics["project"]["bar_duration_issues"]},
        )
    if metrics["project"]["tracks"] < int(global_thresholds.get("min_tracks", 3)):
        score = _penalize(
            issues,
            score,
            severity="error",
            code="too_few_tracks",
            message="Project has fewer tracks than required",
            amount=0.2,
            details={
                "tracks": metrics["project"]["tracks"],
                "minimum": int(global_thresholds.get("min_tracks", 3)),
            },
        )
    if metrics["project"]["note_events"] < int(global_thresholds.get("min_note_events", 80)):
        score = _penalize(
            issues,
            score,
            severity="warning",
            code="low_note_count",
            message="Project has a low number of note events",
            amount=0.12,
            details={
                "note_events": metrics["project"]["note_events"],
                "minimum": int(global_thresholds.get("min_note_events", 80)),
            },
        )
    for track in metrics["tracks"]:
        if track["note_count"] == 0:
            score = _penalize(
                issues,
                score,
                severity="error",
                code="empty_track",
                message=f"Track {track['track_id']} has no notes",
                amount=0.12,
                track_id=track["track_id"],
            )
        if track["missing_bars"]:
            score = _penalize(
                issues,
                score,
                severity="error",
                code="missing_track_bars",
                message=f"Track {track['track_id']} is missing bars",
                amount=0.12,
                track_id=track["track_id"],
                details={"missing_bars": track["missing_bars"]},
            )
    if required_tracks and metrics["project"]["missing_required_tracks"]:
        score = _penalize(
            issues,
            score,
            severity="error",
            code="missing_required_tracks",
            message="Project is missing required benchmark tracks",
            amount=0.25,
            details={"missing": metrics["project"]["missing_required_tracks"]},
        )
    return score


def _apply_role_checks(
    *,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    issues: list[dict[str, Any]],
    score: float,
) -> float:
    bass_thresholds = thresholds.get("bass", {})
    piano_thresholds = thresholds.get("piano", {})
    drum_thresholds = thresholds.get("drums", {})
    melody_thresholds = thresholds.get("melody", {})
    horn_thresholds = thresholds.get("horns", {})

    for track in metrics["tracks"]:
        role = str(track["role"])
        if _metric_is_bass(track):
            bass = track.get("bass", {})
            score = _warn_below(
                issues,
                score,
                bass.get("beat1_root_score"),
                float(bass_thresholds.get("min_beat1_root_score", 0.55)),
                code="bass_low_beat1_root_score",
                message="Walking bass does not land on chord roots often enough on beat 1",
                track_id=track["track_id"],
                amount=0.08,
            )
            score = _warn_below(
                issues,
                score,
                bass.get("approach_to_next_root_score"),
                float(bass_thresholds.get("min_approach_to_next_root_score", 0.35)),
                code="bass_low_approach_score",
                message="Walking bass approach notes do not lead clearly into next roots",
                track_id=track["track_id"],
                amount=0.05,
            )
            if track["large_leaps"] > int(bass_thresholds.get("max_large_leaps", 8)):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="bass_large_leaps",
                    message="Walking bass has too many large leaps",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"large_leaps": track["large_leaps"]},
                )
            score = _warn_below(
                issues,
                score,
                track.get("active_bar_ratio"),
                float(bass_thresholds.get("min_active_bar_ratio", 0.70)),
                code="bass_low_active_bar_ratio",
                message="Walking bass leaves too many inactive bars",
                track_id=track["track_id"],
                amount=0.07,
            )

        if _metric_is_piano(track):
            piano = track.get("piano", {})
            score = _warn_below(
                issues,
                score,
                piano.get("avg_voicing_size"),
                float(piano_thresholds.get("min_avg_voicing_size", 2.0)),
                code="piano_thin_voicings",
                message="Piano voicings are thinner than expected",
                track_id=track["track_id"],
                amount=0.05,
            )
            if piano.get("max_voicing_size", 0) > int(
                piano_thresholds.get("max_voicing_size", 6)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="piano_overfull_voicings",
                    message="Piano voicings exceed the configured maximum size",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"max_voicing_size": piano.get("max_voicing_size")},
                )
            if piano.get("rootless_violations", 0) > int(
                piano_thresholds.get("max_rootless_violations", 24)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="piano_rootless_violations",
                    message="Piano rootless voicings contain too many roots",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"rootless_violations": piano.get("rootless_violations")},
                )
            if piano.get("low_register_notes", 0) > int(
                piano_thresholds.get("max_low_register_notes_below_midi", 40)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="piano_low_register_mud",
                    message="Piano comping uses too much low register material",
                    amount=0.06,
                    track_id=track["track_id"],
                    details={"low_register_notes": piano.get("low_register_notes")},
                )

        if _metric_is_drums(track):
            drums = track.get("drums", {})
            if track.get("channel") not in {None, 10}:
                score = _penalize(
                    issues,
                    score,
                    severity="error",
                    code="drums_wrong_channel",
                    message="Drums must use MIDI channel 10",
                    amount=0.20,
                    track_id=track["track_id"],
                    details={"channel": track.get("channel")},
                )
            if drums.get("distinct_pitch_count", 0) < int(
                drum_thresholds.get("min_drum_pitch_count", 3)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="drums_low_pitch_variety",
                    message="Drums use too few kit pieces",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"distinct_pitch_count": drums.get("distinct_pitch_count")},
                )
            if drums.get("fill_bar_count", 0) < int(
                drum_thresholds.get("min_fill_bar_count", 1)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="drums_missing_fills",
                    message="Drums have too few fill bars before section changes",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"fill_bar_count": drums.get("fill_bar_count")},
                )
            if drums.get("velocity_stddev", 0.0) < float(
                drum_thresholds.get("min_velocity_stddev", 2.0)
            ):
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="drums_flat_velocity",
                    message="Drum velocities are too static",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"velocity_stddev": drums.get("velocity_stddev")},
                )

        if role == "melody":
            score = _melody_like_checks(
                issues,
                score,
                track,
                thresholds=melody_thresholds,
                prefix="melody",
            )
        elif role in {"horn_response", "horn_responses"}:
            score = _melody_like_checks(
                issues,
                score,
                track,
                thresholds=horn_thresholds,
                prefix="horn",
            )
            max_density = float(horn_thresholds.get("max_density_per_bar", 8))
            if track["density_per_bar"] > max_density:
                score = _penalize(
                    issues,
                    score,
                    severity="warning",
                    code="horn_density_too_high",
                    message="Horn response density is too high",
                    amount=0.04,
                    track_id=track["track_id"],
                    details={"density_per_bar": track["density_per_bar"]},
                )
    return score


def _melody_like_checks(
    issues: list[dict[str, Any]],
    score: float,
    track: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    prefix: str,
) -> float:
    score = _warn_below(
        issues,
        score,
        track.get("breath_rest_count"),
        float(thresholds.get("min_breath_rest_count", 1)),
        code=f"{prefix}_low_breath_rests",
        message="Breath instrument has too few playable rests",
        track_id=track["track_id"],
        amount=0.04,
    )
    if track["large_leaps"] > int(thresholds.get("max_large_leaps", 12)):
        score = _penalize(
            issues,
            score,
            severity="warning",
            code=f"{prefix}_large_leaps",
            message="Melodic line has too many large leaps",
            amount=0.04,
            track_id=track["track_id"],
            details={"large_leaps": track["large_leaps"]},
        )
    if "min_active_bar_ratio" in thresholds:
        score = _warn_below(
            issues,
            score,
            track.get("active_bar_ratio"),
            float(thresholds.get("min_active_bar_ratio", 0.35)),
            code=f"{prefix}_low_active_bar_ratio",
            message="Melodic line leaves too much empty space",
            track_id=track["track_id"],
            amount=0.04,
        )
    cadence_score = (track.get("melody") or {}).get("cadence_resolution_score")
    if cadence_score is not None and cadence_score < 0.5:
        score = _penalize(
            issues,
            score,
            severity="warning",
            code=f"{prefix}_weak_cadence_resolution",
            message="Cadences do not resolve clearly into chord tones",
            amount=0.03,
            track_id=track["track_id"],
            details={"cadence_resolution_score": cadence_score},
        )
    if track.get("comfortable_out_count", 0) > 0:
        score = _penalize(
            issues,
            score,
            severity="warning",
            code=f"{prefix}_outside_comfortable_range",
            message="Melodic line leaves the comfortable instrument range",
            amount=0.03,
            track_id=track["track_id"],
            details={"comfortable_out_count": track.get("comfortable_out_count", 0)},
        )
    return score


def _apply_model_checks(
    *,
    metrics: dict[str, Any],
    model_trace: dict[str, Any],
    takes_manifest: dict[str, Any],
    export_mode: str,
    thresholds: dict[str, Any],
    issues: list[dict[str, Any]],
    score: float,
) -> float:
    global_thresholds = thresholds.get("global", {})
    if global_thresholds.get("require_model_trace", True) and not model_trace:
        score = _penalize(
            issues,
            score,
            severity="error",
            code="missing_model_trace",
            message="model_trace.json is required for professional generation",
            amount=0.2,
        )
    if model_trace and not isinstance(model_trace.get("model_artifacts", []), list):
        score = _penalize(
            issues,
            score,
            severity="error",
            code="invalid_model_trace",
            message="model_trace.json must contain a model_artifacts list",
            amount=0.2,
        )

    for artifact in metrics["model_trace"]["artifacts"]:
        task = artifact.get("task", "")
        backend_id = artifact.get("backend_id", "")
        commercial_use = artifact.get("commercial_use", "unknown")
        if any(token in f"{backend_id} {task}".lower() for token in ("audio", "wav", "mp3")):
            score = _penalize(
                issues,
                score,
                severity="error",
                code="audio_backend_in_symbolic_release",
                message="Audio backends are not allowed in symbolic professional exports",
                amount=0.35,
                details={"backend_id": backend_id, "task": task},
            )
        if "text2midi" in backend_id.lower() and str(task).lower() not in SKETCH_ONLY_TASKS:
            score = _penalize(
                issues,
                score,
                severity="error",
                code="text2midi_used_as_final",
                message="Text2MIDI may only be used as sketch_reference, not final material",
                amount=0.35,
                details={"backend_id": backend_id, "task": task},
            )
        if commercial_use == "forbidden":
            score = _penalize(
                issues,
                score,
                severity="error",
                code="model_license_forbidden",
                message="A model artifact is forbidden for release use",
                amount=0.35,
                details={"backend_id": backend_id},
            )
        elif export_mode == "commercial" and commercial_use != "allowed":
            score = _penalize(
                issues,
                score,
                severity="error",
                code="model_license_incompatible",
                message="Commercial export requires model artifacts with commercial_use=allowed",
                amount=0.35,
                details={"backend_id": backend_id, "commercial_use": commercial_use},
            )
        elif commercial_use in {"", "unknown", "review_required"}:
            score = _penalize(
                issues,
                score,
                severity="warning",
                code="model_license_review_required",
                message="Model license requires review before commercial release",
                amount=0.01,
                details={"backend_id": backend_id, "commercial_use": commercial_use},
            )

    if metrics["model_trace"]["text2midi_used_in_final"]:
        score = _penalize(
            issues,
            score,
            severity="error",
            code="text2midi_sketch_used_in_final",
            message="Text2MIDI sketch material was marked as final output",
            amount=0.35,
        )

    if global_thresholds.get("reject_pending_takes_on_export", True):
        for take in _take_list(takes_manifest):
            if take.get("status") == "pending":
                score = _penalize(
                    issues,
                    score,
                    severity="error",
                    code="pending_take_present",
                    message="Pending takes block professional release exports",
                    amount=0.3,
                    details={"take_id": take.get("take_id")},
                )
    for dataset in metrics["datasets"]["entries"]:
        license_name = str(dataset.get("license", "")).strip().lower()
        if license_name in BLOCKED_LICENSES:
            score = _penalize(
                issues,
                score,
                severity="error",
                code="dataset_license_blocked",
                message="A dataset manifest entry has a blocked or missing license",
                amount=0.35,
                details=dataset,
            )
        if export_mode == "commercial" and (
            dataset.get("commercial_training") != "allowed"
            or dataset.get("local_learning_only")
        ):
            score = _penalize(
                issues,
                score,
                severity="error",
                code="dataset_commercial_use_incompatible",
                message="Commercial export requires commercially approved datasets",
                amount=0.35,
                details=dataset,
            )
    return score


def _apply_export_checks(
    project: ArrangementProject,
    *,
    output_root: Path | None,
    export_manifest: dict[str, Any],
    thresholds: dict[str, Any],
    issues: list[dict[str, Any]],
    score: float,
) -> float:
    if output_root is None:
        return _penalize(
            issues,
            score,
            severity="error",
            code="missing_output_dir",
            message="An output directory is required for final professional gate checks",
            amount=0.25,
        )
    global_thresholds = thresholds.get("global", {})
    if export_manifest.get("status") != "exported":
        score = _penalize(
            issues,
            score,
            severity="error",
            code="export_manifest_not_exported",
            message="export_manifest.json is not marked as exported",
            amount=0.25,
            details={"status": export_manifest.get("status")},
        )
    if global_thresholds.get("require_full_midi", True):
        score = _check_midi_file(
            output_root / "full_arrangement.mid",
            issues=issues,
            score=score,
            code="full_midi_missing_or_invalid",
            message="full_arrangement.mid must exist and be parseable",
            amount=0.25,
        )
    midi_track_files = list((output_root / "midi_tracks").glob("*.mid"))
    if len(midi_track_files) < len(project.tracks):
        score = _penalize(
            issues,
            score,
            severity="error",
            code="midi_track_exports_missing",
            message="One MIDI file per arrangement track is required",
            amount=0.2,
            details={"expected": len(project.tracks), "actual": len(midi_track_files)},
        )
    else:
        for midi_track_path in midi_track_files:
            score = _check_midi_file(
                midi_track_path,
                issues=issues,
                score=score,
                code="track_midi_invalid",
                message="Track MIDI export must be parseable",
                amount=0.03,
            )
    if global_thresholds.get("require_musicxml", True):
        musicxml_path = output_root / "full_score.musicxml"
        if not musicxml_path.exists() or musicxml_path.stat().st_size <= 0:
            score = _penalize(
                issues,
                score,
                severity="error",
                code="musicxml_missing",
                message="full_score.musicxml must exist",
                amount=0.25,
            )
        else:
            try:
                converter.parse(musicxml_path)
            except Exception as exc:
                score = _penalize(
                    issues,
                    score,
                    severity="error",
                    code="musicxml_not_parseable",
                    message="full_score.musicxml must be parseable",
                    amount=0.25,
                    details={"error": str(exc)},
                )
    return score


def _warn_below(
    issues: list[dict[str, Any]],
    score: float,
    value: Any,
    threshold: float,
    *,
    code: str,
    message: str,
    track_id: str,
    amount: float,
) -> float:
    numeric = _safe_float(value)
    if numeric is None or numeric >= threshold:
        return score
    return _penalize(
        issues,
        score,
        severity="warning",
        code=code,
        message=message,
        amount=amount,
        track_id=track_id,
        details={"value": numeric, "threshold": threshold},
    )


def _penalize(
    issues: list[dict[str, Any]],
    score: float,
    *,
    severity: Literal["error", "warning"],
    code: str,
    message: str,
    amount: float,
    track_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> float:
    issues.append(
        _issue(
            severity,
            code,
            message,
            track_id=track_id,
            details=details,
        )
    )
    return score - amount


def _issue(
    severity: Literal["error", "warning"],
    code: str,
    message: str,
    *,
    track_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "validator": "ProQualityGate",
        "code": code,
        "message": message,
        "track_id": track_id,
        "details": details or {},
    }


def _check_midi_file(
    path: Path,
    *,
    issues: list[dict[str, Any]],
    score: float,
    code: str,
    message: str,
    amount: float,
) -> float:
    if not path.exists() or path.stat().st_size <= 0:
        return _penalize(
            issues,
            score,
            severity="error",
            code=code,
            message=message,
            amount=amount,
            details={"path": str(path)},
        )
    try:
        mido.MidiFile(path)
    except Exception as exc:
        return _penalize(
            issues,
            score,
            severity="error",
            code=code,
            message=message,
            amount=amount,
            details={"path": str(path), "error": str(exc)},
        )
    return score


def _notes_by_bar(track: Track) -> dict[int, list[NoteEvent]]:
    grouped: dict[int, list[NoteEvent]] = {}
    for bar in track.bars:
        grouped[bar.number] = [
            event for event in bar.events if isinstance(event, NoteEvent)
        ]
    return grouped


def _bass_metrics(
    project: ArrangementProject,
    track: Track,
    chord_parser: ChordParser,
) -> dict[str, Any]:
    chords_by_bar = _chords_by_bar(project.chord_grid)
    beat1_checked = 0
    beat1_root_hits = 0
    approach_checked = 0
    approach_hits = 0
    for bar in track.bars:
        active = _active_chord(chords_by_bar, bar.number, 0.0, chord_parser)
        if active is None:
            continue
        beat1_notes = [
            event
            for event in bar.events
            if isinstance(event, NoteEvent) and abs(event.start) <= 1e-6
        ]
        if beat1_notes:
            beat1_checked += 1
            if any(
                (_safe_note_to_midi(note.pitch) or -999) % 12 == active.root_pc
                for note in beat1_notes
            ):
                beat1_root_hits += 1

        next_root = _next_bar_root(chords_by_bar, bar.number, chord_parser)
        if next_root is None:
            continue
        last_note = _last_note_in_bar(bar)
        if last_note is None:
            continue
        approach_checked += 1
        midi_note = _safe_note_to_midi(last_note.pitch)
        if midi_note is None:
            continue
        distance = _pitch_class_distance(midi_note % 12, next_root)
        role = str(last_note.annotations.get("bass_role") or "")
        if distance <= 2 or role.startswith("approach"):
            approach_hits += 1
    return {
        "beat1_root_score": _ratio(beat1_root_hits, beat1_checked, default=1.0),
        "beat1_root_checked": beat1_checked,
        "approach_to_next_root_score": _ratio(
            approach_hits,
            approach_checked,
            default=1.0,
        ),
        "approach_checked": approach_checked,
    }


def _piano_voicing_metrics(track: Track) -> dict[str, Any]:
    voicing_sizes: list[int] = []
    rootless_violations = 0
    low_register_notes = 0
    for bar in track.bars:
        for notes in _notes_by_slot(bar).values():
            voicing_sizes.append(len(notes))
            midi_notes = []
            root_pc = None
            for event in notes:
                midi_note = _safe_note_to_midi(event.pitch)
                if midi_note is None:
                    continue
                midi_notes.append(midi_note)
                if midi_note < 40:
                    low_register_notes += 1
                if event.annotations.get("root_pc") is not None:
                    root_pc = int(event.annotations["root_pc"])
            if root_pc is not None and any(midi % 12 == root_pc for midi in midi_notes):
                rootless_violations += 1
    return {
        "avg_voicing_size": round(sum(voicing_sizes) / len(voicing_sizes), 3)
        if voicing_sizes
        else 0.0,
        "max_voicing_size": max(voicing_sizes, default=0),
        "rootless_violations": rootless_violations,
        "low_register_notes": low_register_notes,
    }


def _drum_metrics(track: Track, notes: list[NoteEvent]) -> dict[str, Any]:
    midi_notes = [_safe_note_to_midi(note.pitch) for note in notes]
    midi_notes = [midi for midi in midi_notes if midi is not None]
    velocities = [note.velocity for note in notes]
    return {
        "distinct_pitch_count": len(set(midi_notes)),
        "velocity_stddev": round(_stddev(velocities), 3),
        "fill_bar_count": sum(
            1
            for bar in track.bars
            if bar.metadata.get("fill")
            or any(
                isinstance(event, NoteEvent) and event.annotations.get("fill")
                for event in bar.events
            )
        ),
    }


def _cadence_resolution_score(
    project: ArrangementProject,
    track: Track,
    chord_parser: ChordParser,
) -> float | None:
    cadence_bars = sorted({section.end_bar for section in project.form} or {project.bar_count})
    checked = 0
    hits = 0
    chords_by_bar = _chords_by_bar(project.chord_grid)
    for bar_number in cadence_bars:
        bar = _bar_for_number(track, bar_number)
        if bar is None:
            continue
        last_note = _last_note_in_bar(bar)
        if last_note is None:
            continue
        active = _active_chord(chords_by_bar, bar_number, last_note.start, chord_parser)
        if active is None:
            continue
        checked += 1
        midi_note = _safe_note_to_midi(last_note.pitch)
        if midi_note is not None and midi_note % 12 in active.all_pitch_classes:
            hits += 1
    if checked == 0:
        return None
    return _ratio(hits, checked, default=1.0)


def _role_summary(track_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, Any]] = {}
    for track in track_metrics:
        role = str(track["role"])
        entry = roles.setdefault(
            role,
            {
                "tracks": 0,
                "note_count": 0,
                "avg_active_bar_ratio": 0.0,
            },
        )
        entry["tracks"] += 1
        entry["note_count"] += track["note_count"]
        entry["avg_active_bar_ratio"] += track["active_bar_ratio"]
    for entry in roles.values():
        entry["avg_active_bar_ratio"] = round(
            entry["avg_active_bar_ratio"] / max(1, entry["tracks"]),
            3,
        )
    return roles


def _model_trace_metrics(model_trace: dict[str, Any]) -> dict[str, Any]:
    artifacts = [
        artifact
        for artifact in model_trace.get("model_artifacts", [])
        if isinstance(artifact, dict)
    ]
    return {
        "present": bool(model_trace),
        "artifact_count": len(artifacts),
        "backends": sorted(
            {
                str(artifact.get("backend_id"))
                for artifact in artifacts
                if artifact.get("backend_id")
            }
        ),
        "artifacts": [
            {
                "backend_id": str(artifact.get("backend_id") or ""),
                "task": str(artifact.get("task") or ""),
                "commercial_use": str(artifact.get("commercial_use") or "unknown"),
                "validation_status": str(
                    artifact.get("validation_status")
                    or artifact.get("validation_result")
                    or ""
                ),
            }
            for artifact in artifacts
        ],
        "text2midi_used_in_final": bool(model_trace.get("text2midi_used_in_final")),
    }


def _takes_metrics(takes_manifest: dict[str, Any]) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    takes = _take_list(takes_manifest)
    for take in takes:
        statuses[str(take.get("status") or "unknown")] += 1
    return {
        "present": bool(takes_manifest),
        "count": len(takes),
        "statuses": dict(sorted(statuses.items())),
        "pending": statuses.get("pending", 0),
        "accepted": statuses.get("accepted", 0),
        "rejected": statuses.get("rejected", 0),
    }


def _export_metrics(export_manifest: dict[str, Any], output_dir: Path | None) -> dict[str, Any]:
    files = [
        item for item in export_manifest.get("files", []) if isinstance(item, dict)
    ]
    return {
        "present": bool(export_manifest),
        "status": export_manifest.get("status"),
        "file_count": len(files),
        "midi_track_files": sum(1 for item in files if item.get("kind") == "midi_track"),
        "output_dir": str(output_dir) if output_dir else None,
    }


def _dataset_metrics(project: ArrangementProject) -> dict[str, Any]:
    paths = _dataset_manifest_paths(project)
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = _read_json(path)
        for index, entry in enumerate(payload.get("entries", [])):
            if not isinstance(entry, dict):
                continue
            entries.append(
                {
                    "path": str(path),
                    "index": index,
                    "license": entry.get("license"),
                    "commercial_training": entry.get("commercial_training"),
                    "local_learning_only": bool(entry.get("local_learning_only")),
                }
            )
    return {"manifest_paths": paths, "missing_paths": missing, "entries": entries}


def _metric_is_bass(track: dict[str, Any]) -> bool:
    return str(track.get("role")) == "walking_bass" or str(track.get("instrument")) == "double_bass"


def _metric_is_piano(track: dict[str, Any]) -> bool:
    return str(track.get("instrument")) == "piano" or str(track.get("role")) in {
        "comping",
        "piano",
        "piano_comping",
    }


def _metric_is_drums(track: dict[str, Any]) -> bool:
    return str(track.get("role")) == "drums" or str(track.get("instrument")) == "drum_kit"


def _is_bass_track(track: Track) -> bool:
    return track.role == "walking_bass" or track.instrument == "double_bass"


def _is_piano_track(track: Track) -> bool:
    return track.instrument == "piano" or track.role in {"comping", "piano", "piano_comping"}


def _is_drum_track(track: Track) -> bool:
    return track.role == "drums" or track.instrument == "drum_kit"


def _is_melodic_track(track: Track) -> bool:
    return track.role == "melody"


def _is_horn_track(track: Track) -> bool:
    return track.role in {"horn_response", "horn_responses"}


def _comfortable_range_out_count(
    track: Track,
    notes: list[NoteEvent],
    instrument_catalog: InstrumentCatalog,
) -> int:
    try:
        instrument = instrument_catalog.get(track.instrument)
    except KeyError:
        return 0
    low = note_to_midi(instrument.comfortable_range[0])
    high = note_to_midi(instrument.comfortable_range[1])
    count = 0
    for event in notes:
        midi_note = _safe_note_to_midi(event.pitch)
        if midi_note is not None and (midi_note < low or midi_note > high):
            count += 1
    return count


def _large_leaps(midi_notes: list[int], *, limit: int) -> int:
    count = 0
    previous: int | None = None
    for midi_note in midi_notes:
        if previous is not None and abs(midi_note - previous) > limit:
            count += 1
        previous = midi_note
    return count


def _leap_limit(track: Track) -> int:
    if track.role in {"melody", "horn_response", "horn_responses"}:
        return 12
    if track.role == "walking_bass":
        return 8
    return 16


def _missing_bars(project: ArrangementProject, track: Track) -> list[int]:
    present = {bar.number for bar in track.bars}
    return [bar for bar in range(1, project.bar_count + 1) if bar not in present]


def _chords_by_bar(chords: list[ChordSymbol]) -> dict[int, list[ChordSymbol]]:
    grouped: dict[int, list[ChordSymbol]] = defaultdict(list)
    for chord in chords:
        if chord.bar is not None:
            grouped[chord.bar].append(chord)
    for items in grouped.values():
        items.sort(key=lambda chord: chord.beat)
    return dict(grouped)


def _active_chord(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    start: float,
    parser: ChordParser,
) -> ParsedChord | None:
    chord_symbols = chords_by_bar.get(bar_number, [])
    if not chord_symbols:
        return None
    active = chord_symbols[0]
    for chord in chord_symbols:
        if chord.beat - 1.0 <= start + 1e-6:
            active = chord
        else:
            break
    try:
        return parser.parse(active.symbol)
    except ValueError:
        return None


def _next_bar_root(
    chords_by_bar: dict[int, list[ChordSymbol]],
    bar_number: int,
    parser: ChordParser,
) -> int | None:
    for candidate in range(bar_number + 1, bar_number + 5):
        chord_symbols = chords_by_bar.get(candidate, [])
        if not chord_symbols:
            continue
        try:
            return parser.parse(chord_symbols[0].symbol).root_pc
        except ValueError:
            continue
    return None


def _last_note_in_bar(bar: Bar) -> NoteEvent | None:
    notes = [event for event in bar.events if isinstance(event, NoteEvent)]
    if not notes:
        return None
    return max(notes, key=lambda event: (event.start + event.duration, event.start))


def _notes_by_slot(bar: Bar) -> dict[tuple[float, float, int], list[NoteEvent]]:
    grouped: dict[tuple[float, float, int], list[NoteEvent]] = defaultdict(list)
    for event in bar.events:
        if isinstance(event, NoteEvent):
            grouped[(event.start, event.duration, event.voice)].append(event)
    return dict(grouped)


def _bar_for_number(track: Track, bar_number: int) -> Bar | None:
    for bar in track.bars:
        if bar.number == bar_number:
            return bar
    return None


def _pitch_class_distance(left: int, right: int) -> int:
    distance = abs((left - right) % 12)
    return min(distance, 12 - distance)


def _ratio(numerator: int, denominator: int, *, default: float) -> float:
    if denominator == 0:
        return default
    return round(numerator / denominator, 3)


def _stddev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _safe_note_to_midi(pitch: str) -> int | None:
    try:
        return note_to_midi(pitch)
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rating_for_score(score: float, ratings: dict[str, Any]) -> Literal["A", "B", "C", "D"]:
    for rating in ("A", "B", "C", "D"):
        threshold = float((ratings.get(rating) or {}).get("min_score", 0.0))
        if score >= threshold:
            return rating
    return "D"


def _required_track_status(
    required_tracks: list[str],
    available_track_ids: set[str],
) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for required in required_tracks:
        resolved = _resolve_track_id(str(required), available_track_ids)
        if resolved in available_track_ids:
            present.append(str(required))
        else:
            missing.append(str(required))
    return sorted(present), sorted(missing)


def _resolve_track_id(requested_track_id: str, available_track_ids: set[str]) -> str:
    if requested_track_id in available_track_ids:
        return requested_track_id
    for alias in TRACK_ID_ALIASES.get(requested_track_id, ()):
        if alias in available_track_ids:
            return alias
    return requested_track_id


def _take_list(takes_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        take for take in takes_manifest.get("takes", []) if isinstance(take, dict)
    ]


def _dataset_manifest_paths(project: ArrangementProject) -> list[str]:
    candidates: list[Any] = []
    metadata = project.metadata if isinstance(project.metadata, dict) else {}
    candidates.extend(
        [
            metadata.get("dataset_manifest_path"),
            metadata.get("dataset_manifest_paths"),
            metadata.get("dataset_manifests"),
        ]
    )
    if project.generation_spec is not None:
        constraints = project.generation_spec.constraints
        candidates.extend(
            [
                constraints.get("dataset_manifest_path"),
                constraints.get("dataset_manifest_paths"),
                constraints.get("dataset_manifests"),
            ]
        )
    paths: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            paths.append(candidate)
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.append(str(item["path"]))
        elif isinstance(candidate, dict) and isinstance(candidate.get("path"), str):
            paths.append(str(candidate["path"]))
    return sorted(set(paths))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
