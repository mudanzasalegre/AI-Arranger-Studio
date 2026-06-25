from __future__ import annotations

from typing import Any

from arranger_core.schema import ArrangementProject, NoteEvent
from arranger_core.validators import validate_project


class ValidationGate:
    def validate_candidate(
        self,
        *,
        base_project: ArrangementProject,
        candidate_project: ArrangementProject,
        target_track_id: str | None = None,
        target_bars: list[int] | None = None,
        locked_tracks: list[str] | None = None,
        expected_material: bool = True,
    ) -> dict[str, Any]:
        target_bars_set = set(target_bars or [])
        locked_tracks_set = set(locked_tracks or [])
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        errors.extend(
            _outside_target_errors(
                base_project,
                candidate_project,
                target_track_id=target_track_id,
                target_bars=target_bars_set,
                locked_tracks=locked_tracks_set,
            )
        )
        if expected_material and target_track_id is not None:
            if _target_note_count(candidate_project, target_track_id, target_bars_set) == 0:
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "empty_target_material",
                        "Model candidate contains no notes in the target range.",
                        track_id=target_track_id,
                    )
                )

        project_report = validate_project(candidate_project)
        for issue in project_report.get("errors", []):
            errors.append({**issue, "validator": f"Candidate{issue.get('validator', '')}"})
        warnings.extend(project_report.get("warnings", []))

        status = "fail" if errors else "pass_with_warnings" if warnings else "pass"
        return {
            "status": status,
            "project_id": candidate_project.project_id,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "errors": len(errors),
                "warnings": len(warnings),
                "target_track_id": target_track_id,
                "target_bars": sorted(target_bars_set),
                "candidate_note_events": _note_count(candidate_project),
            },
        }


def _outside_target_errors(
    base_project: ArrangementProject,
    candidate_project: ArrangementProject,
    *,
    target_track_id: str | None,
    target_bars: set[int],
    locked_tracks: set[str],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    base_tracks = {track.id: track for track in base_project.tracks}
    candidate_tracks = {track.id: track for track in candidate_project.tracks}

    for track_id in sorted(set(base_tracks) | set(candidate_tracks)):
        base_track = base_tracks.get(track_id)
        candidate_track = candidate_tracks.get(track_id)
        if base_track is None or candidate_track is None:
            if track_id in locked_tracks:
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "locked_track_modified",
                        f"Locked track {track_id!r} was added or removed.",
                        track_id=track_id,
                    )
                )
            elif track_id != target_track_id:
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "non_target_track_modified",
                        f"Track {track_id!r} was added or removed outside the requested target.",
                        track_id=track_id,
                    )
                )
            elif candidate_track is None:
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "target_track_missing",
                        f"Target track {track_id!r} is missing from the candidate.",
                        track_id=track_id,
                    )
                )
            continue

        base_bars = {bar.number: bar for bar in base_track.bars}
        candidate_bars = {bar.number: bar for bar in candidate_track.bars}
        if track_id in locked_tracks and _bars_dump(base_bars) != _bars_dump(candidate_bars):
            errors.append(
                _gate_issue(
                    "ValidationGate",
                    "locked_track_modified",
                    f"Locked track {track_id!r} was modified.",
                    track_id=track_id,
                )
            )
            continue

        if track_id != target_track_id:
            if _bars_dump(base_bars) != _bars_dump(candidate_bars):
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "non_target_track_modified",
                        f"Track {track_id!r} changed outside the requested target.",
                        track_id=track_id,
                    )
                )
            continue

        for bar_number in sorted(set(base_bars) | set(candidate_bars)):
            if bar_number in target_bars:
                continue
            base_bar = base_bars.get(bar_number)
            candidate_bar = candidate_bars.get(bar_number)
            if _dump(base_bar) != _dump(candidate_bar):
                errors.append(
                    _gate_issue(
                        "ValidationGate",
                        "outside_target_bars_modified",
                        f"Track {track_id!r} bar {bar_number} changed outside target bars.",
                        track_id=track_id,
                        bar_number=bar_number,
                    )
                )
    return errors


def _target_note_count(
    project: ArrangementProject,
    track_id: str,
    target_bars: set[int],
) -> int:
    for track in project.tracks:
        if track.id != track_id:
            continue
        return sum(
            1
            for bar in track.bars
            if not target_bars or bar.number in target_bars
            for event in bar.events
            if isinstance(event, NoteEvent)
        )
    return 0


def _note_count(project: ArrangementProject) -> int:
    return sum(
        1
        for track in project.tracks
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    )


def _bars_dump(bars: dict[int, Any]) -> dict[int, Any]:
    return {bar_number: _dump(bar) for bar_number, bar in bars.items()}


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _gate_issue(
    validator: str,
    code: str,
    message: str,
    *,
    track_id: str | None = None,
    bar_number: int | None = None,
) -> dict[str, Any]:
    return {
        "severity": "error",
        "validator": validator,
        "code": code,
        "message": message,
        "track_id": track_id,
        "bar_number": bar_number,
        "details": {},
    }
