from __future__ import annotations

from arranger_core import (
    PIANO_COMPING_ENGINE_VERSION,
    Bar,
    GenerationSpec,
    NoteEvent,
    PianoCompingEngine,
    Track,
    generate_arrangement,
    note_to_midi,
)


def test_piano_comping_engine_controls_density_register_and_voice_leading():
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=920,
            constraints={"piano_retrieval": False},
        ),
        project_id="piano-comping-control",
    )
    piano = _track(project, "piano")
    validation = piano.metadata["piano_comping_validation"]

    assert piano.metadata["piano_comping_engine_version"] == PIANO_COMPING_ENGINE_VERSION
    assert piano.metadata["piano_comping_mode"] == "rule_based"
    assert validation["status"] == "pass"
    assert validation["metrics"]["max_notes_per_bar"] <= 12
    assert validation["metrics"]["max_voicing_size"] <= 4
    assert validation["metrics"]["max_span_semitones"] <= 28
    assert validation["metrics"]["max_voice_leading_semitones"] <= 18
    assert validation["metrics"]["root_doublings"] == 0
    assert project.validate_bar_durations() == []

    piano_notes = _notes(piano)
    assert piano_notes
    assert all(
        note_to_midi(event.pitch) % 12 != event.annotations["root_pc"]
        for event in piano_notes
        if event.annotations.get("root_pc") is not None
    )
    assert any(event.annotations.get("performance_applied") for event in piano_notes)
    assert piano.metadata["piano_comping_ledger"]["entries"]


def test_piano_comping_engine_selects_voicing_styles_by_context():
    ballad = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="jazz_ballad",
            seed=921,
            constraints={"piano_retrieval": False},
        ),
        project_id="piano-ballad",
    )
    modal = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="modal_jazz",
            seed=922,
            constraints={"piano_retrieval": False},
        ),
        project_id="piano-modal",
    )
    shell = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            density="low",
            seed=923,
            constraints={"piano_retrieval": False, "piano_voicing": "shell"},
        ),
        project_id="piano-shell",
    )

    ballad_piano = _track(ballad, "piano")
    modal_piano = _track(modal, "piano")
    shell_piano = _track(shell, "piano")

    assert ballad_piano.metadata["voicing_style"] == "spread"
    assert modal_piano.metadata["voicing_style"] == "quartal"
    assert shell_piano.metadata["voicing_style"] == "shell"
    assert ballad_piano.metadata["piano_comping_validation"]["status"] == "pass"
    assert modal_piano.metadata["piano_comping_validation"]["status"] == "pass"
    assert shell_piano.metadata["piano_comping_validation"]["status"] == "pass"
    assert shell_piano.metadata["piano_comping_validation"]["metrics"]["max_voicing_size"] <= 2


def test_piano_comping_engine_uses_retrieval_pattern_when_valid():
    pattern = {
        "id": "fixture-piano-voicing",
        "category": "piano_voicings",
        "role": "comping",
        "style": "hard_bop",
        "quality": 5,
        "weight": 1.0,
        "usable_for_pattern_extraction": True,
        "payload": {"relative_notes": [0, 4, 7, 10], "density": 4},
    }
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=924,
            constraints={"pattern_index": {"patterns": [pattern]}},
        ),
        project_id="piano-retrieval",
    )
    piano = _track(project, "piano")

    assert piano.metadata["piano_comping_mode"] == "retrieval"
    assert piano.metadata["piano_comping_source"] == "retrieval"
    assert piano.metadata["learned_pattern_id"] == "fixture-piano-voicing"
    assert piano.metadata["piano_comping_validation"]["status"] == "pass"
    assert any(
        event.annotations.get("learned_pattern_id") == "fixture-piano-voicing"
        for event in _notes(piano)
    )


def test_piano_comping_validator_rejects_muddy_dense_root_doubling():
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            seed=925,
            constraints={"piano_retrieval": False},
        ),
        project_id="piano-bad-validation-context",
    )
    bad_track = Track(
        id="piano",
        instrument="piano",
        role="comping",
        bars=[
            Bar(
                number=1,
                events=[
                    NoteEvent(
                        pitch=pitch,
                        start=0.0,
                        duration=1.0,
                        annotations={"source_chord": "Cm7"},
                    )
                    for pitch in ("C2", "E2", "G2", "Bb2", "D3", "F3")
                ],
            )
        ],
    )

    report = PianoCompingEngine().validate_track(project, bad_track)
    error_codes = {issue["code"] for issue in report.errors}

    assert report.status == "fail"
    assert "piano_polyphony" in error_codes
    assert "piano_register" in error_codes
    assert "piano_root_duplication" in error_codes


def _track(project, track_id):
    return next(track for track in project.tracks if track.id == track_id)


def _notes(track):
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
