from __future__ import annotations

from arranger_core.ai.artifact_importer import ImportedModelArtifact
from arranger_core.performance import PerformanceMapper
from arranger_core.schema import ArrangementProject, Bar, RestEvent, Track, meter_to_quarter_beats


class ProjectMergeError(ValueError):
    pass


class ProjectMerger:
    def merge(
        self,
        project: ArrangementProject,
        imported: ImportedModelArtifact,
        *,
        target_track_id: str | None = None,
        target_bars: list[int] | None = None,
        locked_tracks: list[str] | None = None,
    ) -> ArrangementProject:
        if imported.track is None:
            raise ProjectMergeError("Imported artifact has no track material to merge")

        locked = set(locked_tracks or [])
        track_id = target_track_id or imported.track_id or imported.track.id
        if track_id in locked:
            raise ProjectMergeError(f"Target track is locked: {track_id}")

        bars = target_bars or imported.bars or [bar.number for bar in imported.track.bars]
        if not bars:
            raise ProjectMergeError("No target bars supplied")

        candidate = project.model_copy(deep=True)
        candidate.metadata = {
            **candidate.metadata,
            "pending_model_merge": {
                "artifact_id": imported.artifact_id,
                "backend_id": imported.backend_id,
                "task": imported.task,
                "target_track_id": track_id,
                "target_bars": bars,
            },
        }

        target_track = next((track for track in candidate.tracks if track.id == track_id), None)
        if target_track is None:
            target_track = _new_track_from_import(project, imported.track, track_id)
            candidate.tracks.append(target_track)

        imported_bars = {bar.number: bar for bar in imported.track.bars}
        target_by_bar = {bar.number: bar for bar in target_track.bars}
        for bar_number in bars:
            imported_bar = imported_bars.get(bar_number)
            if imported_bar is None:
                imported_bar = _empty_bar(project, bar_number)
            target_by_bar[bar_number] = imported_bar.model_copy(deep=True)
        target_track.bars = [target_by_bar[number] for number in sorted(target_by_bar)]
        return PerformanceMapper().apply(candidate, default_source="rule_based")


def _new_track_from_import(
    project: ArrangementProject,
    imported_track: Track,
    track_id: str,
) -> Track:
    existing_bars = {
        bar_number: _empty_bar(project, bar_number)
        for bar_number in range(1, max(1, project.bar_count) + 1)
    }
    for bar in imported_track.bars:
        existing_bars[bar.number] = bar.model_copy(deep=True)
    return imported_track.model_copy(
        update={
            "id": track_id,
            "bars": [existing_bars[number] for number in sorted(existing_bars)],
        },
        deep=True,
    )


def _empty_bar(project: ArrangementProject, bar_number: int) -> Bar:
    expected = project.meter_at_bar(bar_number)
    return Bar(
        number=bar_number,
        meter=expected,
        events=[RestEvent(start=0.0, duration=meter_to_quarter_beats(expected))],
    )
