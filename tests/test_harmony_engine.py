from arranger_core import (
    GenerationSpec,
    HarmonyFormEngine,
    compile_prompt,
    degree_to_chord_symbol,
    export_project,
    generate_harmony_project,
    parse_key,
)
from music21 import converter


def test_degree_to_chord_symbol_transposes_minor_key_degrees():
    key = parse_key("C minor")

    assert degree_to_chord_symbol("i-7", key) == "Cm7"
    assert degree_to_chord_symbol("iv-7", key) == "Fm7"
    assert degree_to_chord_symbol("iim7b5", key) == "Dm7b5"
    assert degree_to_chord_symbol("V7alt", key) == "G7alt"
    assert degree_to_chord_symbol("bVII13", key) == "Bb13"


def test_minor_blues_generates_12_bar_chord_grid_without_variations_at_low_complexity():
    spec = GenerationSpec(
        key="C minor",
        form="minor_blues_12",
        duration_bars=12,
        complexity=0.2,
        seed=11,
    )

    project = generate_harmony_project(spec, project_id="minor-blues-low")

    assert project.bar_count == 12
    assert len(project.form) == 1
    assert [chord.symbol for chord in project.chord_grid[:4]] == ["Cm7", "Fm7", "Cm7", "Cm7"]
    assert project.metadata["applied_variations"] == []
    assert project.tracks[0].role == "lead_sheet"


def test_supported_forms_generate_expected_bar_counts():
    cases = [
        ("jazz_blues_12", 12),
        ("minor_blues_12", 12),
        ("aaba_32", 32),
        ("rhythm_changes_like_32", 32),
        ("modal_vamp_16", 16),
        ("ballad_aaba_32", 32),
        ("sixteen_bar", 16),
    ]

    for form, expected_bars in cases:
        project = generate_harmony_project(
            GenerationSpec(form=form, key="C minor", complexity=0.4, seed=23)
        )

        assert project.bar_count == expected_bars
        assert max(chord.bar or 0 for chord in project.chord_grid) == expected_bars
        assert all(chord.symbol for chord in project.chord_grid)


def test_modal_vamp_can_generate_16_bars():
    spec = GenerationSpec(
        style="modal_jazz",
        key="D minor",
        form="modal_vamp",
        duration_bars=16,
        complexity=0.3,
        seed=3,
    )

    project = generate_harmony_project(spec)

    assert project.bar_count == 16
    assert len({chord.bar for chord in project.chord_grid}) == 16
    assert project.form[0].name == "Modal Vamp"


def test_aaba_and_rhythm_changes_like_generate_32_bars():
    aaba = generate_harmony_project(
        GenerationSpec(key="F major", form="aaba_32", complexity=0.3, seed=1)
    )
    rhythm = generate_harmony_project(
        GenerationSpec(key="Bb major", form="rhythm_changes_like", complexity=0.3, seed=1)
    )

    assert aaba.bar_count == 32
    assert rhythm.bar_count == 32
    assert [section.name for section in aaba.form] == ["A1", "A2", "Bridge", "A3"]
    assert "I Got Rhythm" not in str(rhythm.model_dump(mode="json"))


def test_high_complexity_variations_are_seed_reproducible():
    spec = GenerationSpec(
        key="C minor",
        form="minor_blues_12",
        complexity=0.9,
        seed=1234,
    )

    first = HarmonyFormEngine().generate(spec)
    second = HarmonyFormEngine().generate(spec)
    other_seed = HarmonyFormEngine().generate(spec.model_copy(update={"seed": 4321}))

    assert [chord.model_dump(mode="json") for chord in first.chord_grid] == [
        chord.model_dump(mode="json") for chord in second.chord_grid
    ]
    assert [chord.model_dump(mode="json") for chord in first.chord_grid] != [
        chord.model_dump(mode="json") for chord in other_seed.chord_grid
    ]
    assert {"turnaround", "backdoor_cadence"}.issubset(set(first.applied_variations))
    assert any(
        chord.metadata.get("variation") == "tritone_substitution"
        for chord in first.chord_grid
    )


def test_compiled_prompt_can_generate_harmony_project():
    spec = compile_prompt(
        "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, "
        "trompeta, trombon, piano, contrabajo y bateria",
        seed=1234,
    )
    project = generate_harmony_project(spec)

    assert project.generation_spec == spec
    assert project.bar_count == 12
    assert project.chord_grid[0].symbol == "Cm7"


def test_harmony_chord_grid_exports_to_musicxml(tmp_path):
    spec = GenerationSpec(
        key="C minor",
        form="minor_blues_12",
        complexity=0.75,
        seed=1234,
        instruments=["piano"],
    )
    project = generate_harmony_project(spec, project_id="harmony-export")

    export_project(project, tmp_path, include_pdf=False)

    musicxml_path = tmp_path / "full_score.musicxml"
    converter.parse(musicxml_path)
    xml_text = musicxml_path.read_text(encoding="utf-8")

    assert "<harmony" in xml_text
    assert "<root-step>C</root-step>" in xml_text
    assert "minor-seventh" in xml_text
    assert "Lead Sheet" in xml_text
