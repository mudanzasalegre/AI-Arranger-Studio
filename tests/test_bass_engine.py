from __future__ import annotations

from typing import Any

from arranger_core import (
    BASS_ENGINE_VERSION,
    BassEngine,
    ChordParser,
    GenerationSpec,
    NoteEvent,
    RuleBasedArranger,
    Track,
    WalkingBassGenerator,
    generate_arrangement,
    note_to_midi,
)
from arranger_core.schema import Bar


def test_bass_engine_walking_line_resolves_into_next_roots() -> None:
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=9101,
            constraints={"bass_retrieval": False},
        ),
        project_id="bass-engine-walking",
    )

    bass = _track(project, "double_bass")
    validation = bass.metadata["bass_validation"]

    assert bass.metadata["bass_engine_version"] == BASS_ENGINE_VERSION
    assert bass.metadata["bass_engine_mode"] == "rule_based"
    assert bass.metadata["bass_line_style"] == "walking_bass"
    assert validation["status"] == "pass"
    assert validation["metrics"]["root_on_downbeat_ratio"] == 1.0
    assert validation["metrics"]["approach_resolution_ratio"] >= 0.65
    assert validation["metrics"]["max_leap_semitones"] <= 14

    parser = ChordParser.load_default()
    chords_by_bar = _chords_by_bar(project.chord_grid)
    for bar in bass.bars[:-1]:
        last_note = _notes(bar)[-1]
        next_symbol = _first_chord_after(chords_by_bar, bar.number, project.bar_count).symbol
        next_root = parser.parse(next_symbol).root_pc
        assert _pc_distance(note_to_midi(last_note.pitch) % 12, next_root) <= 2


def test_bass_engine_supports_bossa_two_feel_modal_and_waltz() -> None:
    bossa = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="bossa_32",
            style="bossa_nova",
            seed=9102,
        ),
        project_id="bass-engine-bossa",
    )
    bossa_bass = _track(bossa, "double_bass")
    assert bossa_bass.metadata["bass_line_style"] == "bossa_bass"
    assert bossa_bass.metadata["bass_validation"]["status"] == "pass"
    assert [event.start for event in _notes(bossa_bass.bars[0])] == [0.0, 1.5, 2.5]

    ballad = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="aaba_32",
            style="jazz_ballad",
            density="low",
            seed=9103,
        ),
        project_id="bass-engine-two-feel",
    )
    ballad_bass = _track(ballad, "double_bass")
    assert ballad_bass.metadata["bass_line_style"] == "two_feel"
    assert ballad_bass.metadata["bass_validation"]["status"] == "pass"
    assert all(len(_notes(bar)) == 2 for bar in ballad_bass.bars)
    assert {event.duration for event in _notes(ballad_bass.bars[0])} == {2.0}

    modal = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="modal_vamp_16",
            style="modal_jazz",
            seed=9104,
        ),
        project_id="bass-engine-modal",
    )
    modal_bass = _track(modal, "double_bass")
    assert modal_bass.metadata["bass_line_style"] == "pedal_modal"
    assert modal_bass.metadata["bass_validation"]["status"] == "pass"
    assert any(
        event.annotations.get("bass_role") == "pedal_root"
        for event in _notes(modal_bass.bars[0])
    )

    waltz = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="jazz_waltz_32",
            style="jazz_waltz",
            meter="3/4",
            seed=9105,
        ),
        project_id="bass-engine-waltz",
    )
    waltz_bass = _track(waltz, "double_bass")
    assert waltz_bass.metadata["bass_line_style"] == "waltz_bass"
    assert waltz_bass.metadata["bass_validation"]["status"] == "pass"
    assert all(len(_notes(bar)) == 3 for bar in waltz_bass.bars)


def test_bass_engine_retrieval_adapts_pattern_and_records_ledger() -> None:
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=9106,
            constraints={
                "pattern_index": {
                    "patterns": [
                        {
                            "id": "bass_cell_test",
                            "category": "walking_bass_cells",
                            "role": "walking_bass",
                            "style": "hard_bop",
                            "quality": 5,
                            "weight": 1.0,
                            "usable_for_pattern_extraction": True,
                            "payload": {
                                "pitch_intervals": [0, 7, 10, 0],
                                "rhythm": [0.0, 1.0, 2.0, 3.0],
                            },
                        }
                    ]
                }
            },
        ),
        project_id="bass-engine-retrieval",
    )

    bass = _track(project, "double_bass")
    ledger = bass.metadata["bass_line_ledger"]

    assert bass.metadata["bass_engine_mode"] == "retrieval"
    assert bass.metadata["bass_source"] == "retrieval"
    assert bass.metadata["learned_pattern_id"] == "bass_cell_test"
    assert bass.metadata["bass_validation"]["status"] == "pass"
    assert ledger["entries"][0]["source_pattern_id"] == "bass_cell_test"
    assert _notes(bass.bars[0])[-1].annotations["bass_role"] == "approach_next_root"


def test_bass_engine_rejects_erratic_ai_and_falls_back_to_rule_based() -> None:
    class ErraticBassBackend:
        def generate_bass_track(
            self,
            *,
            project: Any,
            base_track: Track,
            context: Any,
        ) -> Track:
            _ = project, context
            bars = [
                Bar(
                    number=bar.number,
                    events=[
                        NoteEvent(
                            pitch="E1" if index % 2 == 0 else "C4",
                            start=float(index),
                            duration=1.0,
                            velocity=70,
                            annotations={"bass_role": "ai_candidate"},
                        )
                        for index in range(4)
                    ],
                )
                for bar in base_track.bars
            ]
            return base_track.model_copy(
                update={"bars": bars, "metadata": {**base_track.metadata, "ai": True}}
            )

    engine = BassEngine(ai_backend=ErraticBassBackend())
    arranger = RuleBasedArranger(bass_generator=WalkingBassGenerator(engine=engine))
    project = arranger.generate(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            seed=9107,
            constraints={"bass_engine_mode": "ai_infill", "bass_retrieval": False},
        ),
        project_id="bass-engine-ai-fallback",
    )

    bass = _track(project, "double_bass")

    assert bass.metadata["bass_engine_mode"] == "rule_based"
    assert bass.metadata["fallback_reason"] == "ai_validation_failed"
    assert bass.metadata["bass_validation"]["status"] == "pass"
    assert not any(
        event.annotations.get("bass_role") == "ai_candidate"
        for event in _all_notes(bass)
    )


def test_bass_validator_flags_erratic_contour() -> None:
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            seed=9108,
            constraints={"bass_retrieval": False},
        ),
        project_id="bass-validator-source",
    )
    bass = _track(project, "double_bass")
    bad_bars = [
        bass.bars[0].model_copy(
            update={
                "events": [
                    NoteEvent(
                        pitch="E1",
                        start=0.0,
                        duration=1.0,
                        annotations={"bass_role": "root"},
                    ),
                    NoteEvent(
                        pitch="C4",
                        start=1.0,
                        duration=1.0,
                        annotations={"bass_role": "walking"},
                    ),
                    NoteEvent(
                        pitch="E1",
                        start=2.0,
                        duration=1.0,
                        annotations={"bass_role": "walking"},
                    ),
                    NoteEvent(
                        pitch="C4",
                        start=3.0,
                        duration=1.0,
                        annotations={"bass_role": "walking"},
                    ),
                ]
            }
        )
    ]
    bad_track = bass.model_copy(update={"bars": bad_bars})

    report = BassEngine().validate_track(
        project.model_copy(update={"tracks": [bad_track], "form": project.form[:1]}),
        bad_track,
    )

    assert report.status == "fail"
    assert any(error["code"] == "bass_erratic_contour" for error in report.errors)


def _track(project, track_id: str) -> Track:
    return next(track for track in project.tracks if track.id == track_id)


def _notes(bar: Bar) -> list[NoteEvent]:
    return [event for event in bar.events if isinstance(event, NoteEvent)]


def _all_notes(track: Track) -> list[NoteEvent]:
    return [event for bar in track.bars for event in _notes(bar)]


def _chords_by_bar(chord_grid) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for chord in chord_grid:
        if chord.bar is None:
            continue
        grouped.setdefault(chord.bar, []).append(chord)
    for chords in grouped.values():
        chords.sort(key=lambda chord: chord.beat)
    return grouped


def _first_chord_after(chords_by_bar: dict[int, list], bar_number: int, max_bar: int):
    for next_bar in range(bar_number + 1, max_bar + 1):
        chords = chords_by_bar.get(next_bar, [])
        if chords:
            return chords[0]
    return chords_by_bar[bar_number][0]


def _pc_distance(first: int, second: int) -> int:
    diff = abs((first - second) % 12)
    return min(diff, 12 - diff)
