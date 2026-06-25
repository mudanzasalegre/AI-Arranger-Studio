import pytest
from arranger_core import (
    ChordParser,
    InstrumentCatalog,
    MusicConfigLoader,
    PatternLibrary,
    ProgressionLibrary,
    ScaleCatalog,
    StyleProfileCatalog,
    midi_to_note,
    note_to_midi,
    pitch_class,
)


@pytest.mark.parametrize(
    ("symbol", "quality", "root", "bass"),
    [
        ("F#m7b5", "half_diminished", "F#", None),
        ("B7alt", "dominant", "B", None),
        ("Ebmaj7#11", "major_triad", "Eb", None),
        ("G13b9", "dominant", "G", None),
        ("CmMaj9", "minor_triad", "C", None),
        ("D7#5#9/Ab", "dominant", "D", "Ab"),
    ],
)
def test_chord_parser_accepts_required_complex_jazz_chords(symbol, quality, root, bass):
    parsed = ChordParser.load_default().parse(symbol)

    assert parsed.symbol == symbol
    assert parsed.quality == quality
    assert parsed.root == root
    assert parsed.bass == bass


def test_chord_parser_calculates_chord_tones_and_tensions():
    parser = ChordParser.load_default()

    half_diminished = parser.parse("F#m7b5")
    assert half_diminished.chord_tone_pcs == [
        pitch_class("F#"),
        pitch_class("A"),
        pitch_class("C"),
        pitch_class("E"),
    ]

    lydian_major = parser.parse("Ebmaj7#11")
    assert lydian_major.chord_tone_pcs == [
        pitch_class("Eb"),
        pitch_class("G"),
        pitch_class("Bb"),
        pitch_class("D"),
    ]
    assert pitch_class("A") in lydian_major.tension_pcs

    altered = parser.parse("B7alt")
    for note_name in ["C", "D", "F", "G"]:
        assert pitch_class(note_name) in altered.tension_pcs

    slash = parser.parse("D7#5#9/Ab")
    assert slash.bass_pc == pitch_class("Ab")
    assert pitch_class("A") in slash.chord_tone_pcs
    assert pitch_class("A#") not in slash.chord_tone_pcs
    assert pitch_class("A#") in slash.alteration_pcs
    assert pitch_class("F") in slash.alteration_pcs

    minor_major = parser.parse("CmMaj9")
    assert pitch_class("B") in minor_major.chord_tone_pcs
    assert pitch_class("Bb") not in minor_major.chord_tone_pcs
    assert pitch_class("D") in minor_major.tension_pcs


def test_scale_catalog_loads_scales_and_returns_notes():
    scales = ScaleCatalog.load_default()

    assert scales.get("altered") == [0, 1, 3, 4, 6, 8, 10]
    assert scales.notes("C", "minor_blues") == ["C", "Eb", "F", "Gb", "G", "Bb"]
    assert scales.pitch_classes("F#", "dorian")[0] == pitch_class("F#")


def test_instrument_catalog_loads_ensembles_and_transposes_instruments():
    instruments = InstrumentCatalog.load_default()

    sextet = instruments.instruments_for_ensemble("jazz_sextet")
    assert [instrument.id for instrument in sextet] == [
        "drum_kit",
        "double_bass",
        "piano",
        "alto_sax",
        "trumpet_bflat",
        "trombone",
    ]

    assert instruments.get("trumpet_bflat").transposition_semitones == -2
    assert instruments.written_to_sounding("trumpet_bflat", "C4") == "Bb3"
    assert instruments.sounding_to_written("trumpet_bflat", "Bb3") == "C4"
    assert instruments.written_to_sounding("alto_sax", "C4") == "Eb3"
    assert instruments.written_to_sounding("double_bass", "C3") == "C2"
    assert midi_to_note(note_to_midi("C4") - 2) == "Bb3"


def test_progressions_styles_and_patterns_load_from_yaml():
    progressions = ProgressionLibrary.load_default()
    minor_blues = progressions.get("minor_blues_12")
    assert minor_blues.bars == 12
    assert minor_blues.degrees[0] == "i-7"

    styles = StyleProfileCatalog.load_default()
    hard_bop = styles.get("hard_bop")
    assert hard_bop.family == "jazz"
    assert hard_bop.roles["bass"] == "walking_bass"
    assert styles.get("modal_jazz").harmony["allow_static_vamps"] is True

    patterns = PatternLibrary.load_default()
    assert "medium_hard_bop" in patterns.category("comping_rhythms")
    assert patterns.get("walking_bass_cells", "approach_from_below")["beats"][0] == "root"
    assert patterns.get("drum_grooves", "swing_ride_basic")["feel"] == "swing"


def test_config_loader_can_load_named_yaml_file():
    loader = MusicConfigLoader()
    data = loader.load_yaml("instruments.yaml")

    assert "instruments" in data
    assert "piano" in data["instruments"]
