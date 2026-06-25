from __future__ import annotations

import json

from arranger_core import (
    GenerationSpec,
    RuleBasedArranger,
    SongPlan,
    SongPlanner,
    WalkingBassGenerator,
    export_project,
    generate_harmony_project,
)


def test_song_plan_serializes_and_is_seed_deterministic(tmp_path):
    spec = GenerationSpec(
        style="hard_bop",
        form="minor_blues_12",
        ensemble="jazz_sextet",
        seed=161,
    )
    project = generate_harmony_project(spec, project_id="song-plan-deterministic")

    first = SongPlanner().plan(spec, project)
    second = SongPlanner().plan(spec, project)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len(first.sections) == 3
    assert {section.function for section in first.sections} == {
        "head_statement",
        "response",
        "turnaround",
    }
    assert len({section.energy for section in first.sections}) > 1

    plan_path = tmp_path / "song_plan.json"
    first.save_json(plan_path)
    loaded = SongPlan.load_json(plan_path)
    assert loaded.model_dump(mode="json") == first.model_dump(mode="json")


def test_aaba_bridge_has_distinct_energy():
    spec = GenerationSpec(style="swing", form="aaba_32", ensemble="jazz_quartet_alto", seed=162)
    project = generate_harmony_project(spec, project_id="song-plan-aaba")

    plan = SongPlanner().plan(spec, project)
    bridge = next(section for section in plan.sections if section.function == "bridge")
    non_bridge = [section for section in plan.sections if section.function != "bridge"]

    assert bridge.energy > max(section.energy for section in non_bridge)
    assert bridge.groove_feel == "swing"
    assert bridge.harmonic_rhythm == "active"


def test_ballad_plan_keeps_head_and_ending_lower_density():
    spec = GenerationSpec(
        style="jazz_ballad",
        form="ballad_aaba_32",
        ensemble="jazz_quartet_alto",
        seed=163,
    )
    project = generate_harmony_project(spec, project_id="song-plan-ballad")

    plan = SongPlanner().plan(spec, project)
    intro = plan.sections[0]
    bridge = next(section for section in plan.sections if section.function == "bridge")
    ending = plan.sections[-1]

    assert intro.energy < bridge.energy
    assert ending.energy < bridge.energy
    assert intro.role_densities["comping"] < bridge.role_densities["comping"]
    assert ending.role_densities["drums"] < bridge.role_densities["drums"]
    assert plan.ending_strategy == "soft_tag"


def test_generators_receive_global_song_plan_and_project_metadata_exports_it(tmp_path):
    spy_bass = SpyBassGenerator()
    arranger = RuleBasedArranger(bass_generator=spy_bass)
    project = arranger.generate(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=164),
        project_id="song-plan-context",
    )

    assert spy_bass.received_song_plan is not None
    assert spy_bass.received_song_plan.song_id == "song-plan-context"
    song_plan_metadata = project.metadata["song_plan"]
    assert song_plan_metadata["song_id"] == "song-plan-context"
    assert song_plan_metadata["sections"]
    assert song_plan_metadata["phrases"]
    assert song_plan_metadata["groove_map"]["feel"] == "swing"
    assert song_plan_metadata["groove_map"]["beat_grid"]

    phrase_ids = {phrase["id"] for phrase in song_plan_metadata["phrases"]}
    assert all(
        phrase_id in phrase_ids
        for section in song_plan_metadata["sections"]
        for phrase_id in section["phrase_ids"]
    )

    manifest = export_project(project, tmp_path, include_pdf=False)
    kinds = {file["kind"] for file in manifest["files"]}
    assert "song_plan_json" in kinds
    exported_plan = json.loads((tmp_path / "song_plan.json").read_text(encoding="utf-8"))
    exported_project = json.loads(
        (tmp_path / "arrangement_project.json").read_text(encoding="utf-8")
    )
    assert exported_plan == exported_project["metadata"]["song_plan"]
    assert exported_plan["sections"] == song_plan_metadata["sections"]
    assert exported_plan["phrases"] == song_plan_metadata["phrases"]
    assert exported_plan["groove_map"] == song_plan_metadata["groove_map"]


class SpyBassGenerator:
    role = "walking_bass"

    def __init__(self) -> None:
        self.received_song_plan = None
        self._delegate = WalkingBassGenerator()

    def generate(self, context):
        self.received_song_plan = context.song_plan
        return self._delegate.generate(context)
