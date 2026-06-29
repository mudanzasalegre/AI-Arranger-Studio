from __future__ import annotations

from arranger_core import (
    DRUM_ENGINE_VERSION,
    DrumsEngine,
    GenerationSpec,
    NoteEvent,
    Track,
    generate_arrangement,
)
from arranger_core.schema import ArrangementProject, Bar, ChordSymbol, Section, TempoMark


def test_drums_engine_follows_groove_map_setups_fills_and_horn_hits() -> None:
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_sextet",
            form="minor_blues_12",
            style="hard_bop",
            seed=10101,
        ),
        project_id="drums-engine-form",
    )
    drums = _track(project, "drum_kit")
    groove_map = project.metadata["song_plan"]["groove_map"]
    validation = drums.metadata["drums_validation"]

    assert drums.metadata["drums_engine_version"] == DRUM_ENGINE_VERSION
    assert drums.metadata["drums_engine_mode"] == "rule_based"
    assert drums.metadata["groove"] == "swing"
    assert validation["status"] == "pass"
    assert validation["metrics"]["fill_bars"] == groove_map["fill_bars"]
    assert validation["metrics"]["setup_bars"] == groove_map["setup_bars"]
    assert validation["metrics"]["horn_hit_bars"] == groove_map["horn_hit_bars"]
    assert validation["metrics"]["unique_bar_signatures"] >= 5
    assert validation["metrics"]["density_stdev"] > 0

    for bar_number in groove_map["setup_bars"]:
        assert any(
            event.annotations.get("setup")
            for event in _notes(drums.bars[bar_number - 1])
        )
    for bar_number in groove_map["horn_hit_bars"]:
        assert any(
            event.annotations.get("horn_hit_support")
            for event in _notes(drums.bars[bar_number - 1])
        )
    assert any(event.annotations.get("performance_applied") for event in _all_notes(drums))


def test_drums_engine_generates_distinct_style_grooves() -> None:
    bossa = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="bossa_32",
            style="bossa_nova",
            seed=10102,
        ),
        project_id="drums-bossa",
    )
    bossa_drums = _track(bossa, "drum_kit")
    assert bossa_drums.metadata["groove"] == "bossa"
    assert any(
        event.annotations.get("cross_stick")
        for event in _all_notes(bossa_drums)
    )

    funk = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="modal_vamp_16",
            style="funk_jazz",
            seed=10103,
        ),
        project_id="drums-funk",
    )
    funk_drums = _track(funk, "drum_kit")
    assert funk_drums.metadata["groove"] == "funk"
    assert any(event.annotations.get("backbeat") for event in _all_notes(funk_drums))
    assert any(event.annotations.get("kick_lock") for event in _all_notes(funk_drums))

    waltz = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="jazz_waltz_32",
            style="jazz_waltz",
            meter="3/4",
            seed=10104,
        ),
        project_id="drums-waltz",
    )
    waltz_drums = _track(waltz, "drum_kit")
    assert waltz_drums.metadata["groove"] == "waltz"
    assert all(bar.metadata["feel"] == "waltz" for bar in waltz_drums.bars)
    assert any(event.annotations.get("waltz_comp") for event in _all_notes(waltz_drums))


def test_drums_engine_retrieval_keeps_timekeeping_and_records_pattern() -> None:
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=10105,
            constraints={
                "pattern_index": {
                    "patterns": [
                        {
                            "id": "drum_groove_test",
                            "category": "drum_grooves",
                            "role": "drums",
                            "style": "hard_bop",
                            "quality": 5,
                            "weight": 1.0,
                            "usable_for_pattern_extraction": True,
                            "payload": {
                                "bar": 1,
                                "meter": "4/4",
                                "events": [
                                    {"beat": 0.0, "pitch": 36, "velocity": 82},
                                    {"beat": 1.0, "pitch": 38, "velocity": 76},
                                    {"beat": 2.5, "pitch": 51, "velocity": 70},
                                ],
                            },
                        }
                    ]
                }
            },
        ),
        project_id="drums-retrieval",
    )
    drums = _track(project, "drum_kit")
    retrieval_bars = [
        bar
        for bar in drums.bars
        if bar.metadata.get("drums_source") == "retrieval"
    ]

    assert drums.metadata["drums_engine_mode"] == "retrieval"
    assert drums.metadata["learned_pattern_id"] == "drum_groove_test"
    assert retrieval_bars
    assert any(event.annotations.get("drum") == "learned" for event in _all_notes(drums))
    assert any(
        event.annotations.get("retrieval_backbone")
        for bar in retrieval_bars
        for event in _notes(bar)
    )
    assert drums.metadata["drums_validation"]["status"] == "pass"


def test_drums_validator_warns_on_flat_pattern_without_fills() -> None:
    bars = [
        Bar(
            number=bar_number,
            events=[
                NoteEvent(pitch="D#3", start=start, duration=0.5, annotations={"drum": "ride"})
                for start in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
            ],
        )
        for bar_number in range(1, 9)
    ]
    track = Track(id="drum_kit", instrument="drum_kit", role="drums", bars=bars)
    project = ArrangementProject(
        project_id="flat-drums",
        generation_spec=GenerationSpec(duration_bars=8),
        tempo_map=[TempoMark(bar=1, bpm=132)],
        form=[Section(name="A", start_bar=1, end_bar=8)],
        chord_grid=[ChordSymbol(symbol="Cm7", bar=1, beat=1.0)],
        tracks=[track],
    )

    report = DrumsEngine().validate_track(project, track)
    warning_codes = {warning["code"] for warning in report.warnings}

    assert report.status == "pass"
    assert "missing_fills" in warning_codes
    assert "flat_drum_language" in warning_codes


def _track(project, track_id: str) -> Track:
    return next(track for track in project.tracks if track.id == track_id)


def _notes(bar: Bar) -> list[NoteEvent]:
    return [event for event in bar.events if isinstance(event, NoteEvent)]


def _all_notes(track: Track) -> list[NoteEvent]:
    return [event for bar in track.bars for event in _notes(bar)]
