from __future__ import annotations

from arranger_core import (
    AIDrumsGenerator,
    AIHornResponseGenerator,
    AIMelodyGenerator,
    AIPianoCompingGenerator,
    AIWalkingBassGenerator,
    DeterministicRoleModelBackend,
    GenerationSpec,
    RuleBasedArranger,
    export_project,
)


def test_custom_role_model_generators_replace_all_roles_without_export_changes(tmp_path):
    backend = DeterministicRoleModelBackend()
    arranger = RuleBasedArranger(
        drums_generator=AIDrumsGenerator(backend, model_mode="external_model"),
        bass_generator=AIWalkingBassGenerator(backend, model_mode="custom_model"),
        piano_generator=AIPianoCompingGenerator(backend, model_mode="custom_model"),
        melody_generator=AIMelodyGenerator(backend, model_mode="external_model"),
        horn_response_generator=AIHornResponseGenerator(backend, model_mode="custom_model"),
    )

    project = arranger.generate(
        GenerationSpec(
            ensemble="jazz_sextet",
            form="minor_blues_12",
            style="hard_bop",
            seed=1901,
            constraints={"humanize": False},
        ),
        project_id="custom-role-model-interface",
    )
    manifest = export_project(project, tmp_path / "export", include_pdf=False)

    assert project.metadata["arranger"] == "hybrid_rule_model_v0"
    assert project.validate_bar_durations() == []
    assert manifest["status"] == "exported"
    assert (tmp_path / "export/full_arrangement.mid").exists()

    tracks_by_id = {track.id: track for track in project.tracks}
    assert tracks_by_id["drum_kit"].metadata["role_model_mode"] == "external_model"
    assert tracks_by_id["double_bass"].metadata["role_model_mode"] == "custom_model"
    assert tracks_by_id["piano"].metadata["role_model_mode"] == "custom_model"
    assert tracks_by_id["alto_sax"].metadata["role_model_mode"] == "external_model"
    assert tracks_by_id["trumpet_bflat"].metadata["role_model_mode"] == "custom_model"
    assert tracks_by_id["trombone"].metadata["role_model_mode"] == "custom_model"
    assert all(
        "custom_model" in track.metadata["modes_available"]
        and "external_model" in track.metadata["modes_available"]
        for track in tracks_by_id.values()
    )
    assert all(
        track.metadata["model_backend"] == "deterministic-role-model-placeholder"
        for track in tracks_by_id.values()
    )
