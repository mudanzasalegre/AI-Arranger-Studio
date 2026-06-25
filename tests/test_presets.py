from __future__ import annotations

from arranger_core import (
    PresetLibrary,
    compile_prompt,
    export_project,
    generate_arrangement,
    validate_project,
)
from music21 import converter

REQUIRED_PRESETS = {
    "jazz_hard_bop_minor_blues_sextet",
    "jazz_bebop_blues_quintet",
    "jazz_swing_aaba_quartet",
    "jazz_ballad_quartet",
    "jazz_modal_quintet",
    "jazz_bossa_nova_quartet",
    "jazz_waltz_trio",
    "jazz_funk_straight_eighth_quintet",
}


def test_required_generation_presets_and_evaluation_pack_load():
    library = PresetLibrary.load_default()

    assert {preset.id for preset in library.list_presets()} == REQUIRED_PRESETS
    assert len(library.evaluation_pack()) == 20
    assert {
        prompt.preset_id for prompt in library.evaluation_pack()
    }.issubset(REQUIRED_PRESETS)
    assert all(
        preset.spec.constraints["preset_id"] == preset.id
        for preset in library.list_presets()
    )


def test_evaluation_prompts_compile_to_expected_styles():
    library = PresetLibrary.load_default()
    expected_style_by_preset = {
        preset.id: preset.spec.style for preset in library.list_presets()
    }

    for item in library.evaluation_pack():
        compiled = compile_prompt(item.prompt, seed=item.seed)
        assert compiled.style == expected_style_by_preset[item.preset_id]


def test_each_preset_generates_valid_complete_project_and_exports(tmp_path):
    library = PresetLibrary.load_default()

    for preset in library.list_presets():
        project = generate_arrangement(preset.spec, project_id=preset.id)
        report = validate_project(project)

        assert project.project_id == preset.id
        assert project.bar_count == preset.spec.duration_bars
        assert len(project.tracks) == len(preset.spec.instruments)
        assert project.validate_bar_durations() == []
        assert report["errors"] == []

        manifest = export_project(
            project,
            tmp_path / preset.id,
            include_pdf=False,
        )
        kinds = {file["kind"] for file in manifest["files"]}
        assert {"midi_full", "midi_track", "musicxml_full"} <= kinds
        converter.parse(tmp_path / preset.id / "full_score.musicxml")


def test_style_markers_are_present_in_generated_presets():
    library = PresetLibrary.load_default()
    generated = {
        preset.id: generate_arrangement(preset.spec, project_id=preset.id)
        for preset in library.list_presets()
    }

    hard_bop = generated["jazz_hard_bop_minor_blues_sextet"]
    assert [track.id for track in hard_bop.tracks] == [
        "drum_kit",
        "double_bass",
        "piano",
        "alto_sax",
        "trumpet_bflat",
        "trombone",
    ]
    assert any(track.role == "horn_response" for track in hard_bop.tracks)

    bossa_drums = _track(generated["jazz_bossa_nova_quartet"], "drum_kit")
    assert bossa_drums.metadata["groove"] == "bossa"

    funk_drums = _track(generated["jazz_funk_straight_eighth_quintet"], "drum_kit")
    assert funk_drums.metadata["groove"] == "funk"

    waltz = generated["jazz_waltz_trio"]
    assert waltz.generation_spec is not None
    assert waltz.generation_spec.meter == "3/4"
    assert _track(waltz, "drum_kit").metadata["groove"] == "waltz"
    assert all(len(bar.events) == 3 for bar in _track(waltz, "double_bass").bars)


def _track(project, track_id):
    return next(track for track in project.tracks if track.id == track_id)
