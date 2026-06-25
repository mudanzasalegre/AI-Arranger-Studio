from __future__ import annotations

from arranger_core import (
    ArtifactImporter,
    ArtifactStore,
    GenerationSpec,
    ProjectMerger,
    generate_arrangement,
)
from model_backends import MockSymbolicBackend, ModelGenerationRequest


def test_project_merger_only_replaces_target_track_bars(tmp_path):
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=303),
        project_id="merge-target",
    )
    before = project.model_dump(mode="json")
    imported = _import_mock_track(project, tmp_path, target_bars=[1])

    candidate = ProjectMerger().merge(
        project,
        imported,
        target_track_id="alto_sax",
        target_bars=[1],
    )
    after = candidate.model_dump(mode="json")

    assert before["metadata"]["song_plan"] == after["metadata"]["song_plan"]
    assert before["tracks"] != after["tracks"]
    for before_track, after_track in zip(before["tracks"], after["tracks"], strict=True):
        if before_track["id"] != "alto_sax":
            assert before_track == after_track
            continue
        before_bars = {bar["number"]: bar for bar in before_track["bars"]}
        after_bars = {bar["number"]: bar for bar in after_track["bars"]}
        assert before_bars[1] != after_bars[1]
        for bar_number in range(2, project.bar_count + 1):
            assert before_bars[bar_number] == after_bars[bar_number]


def _import_mock_track(project, tmp_path, *, target_bars):
    backend = MockSymbolicBackend(output_dir=tmp_path / "backend_raw")
    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="merge-artifact",
            track_id="alto_sax",
            bars=target_bars,
            seed=303,
        )
    )
    store = ArtifactStore(tmp_path / "model_artifacts")
    record = store.store_generation_result(result, project_id=project.project_id)[0]
    return ArtifactImporter(artifact_store=store).import_record(
        record,
        project=project,
        target_track_id="alto_sax",
        target_bars=target_bars,
    )
