import pytest
from arranger_core import (
    ArrangementProject,
    Bar,
    ChordSymbol,
    GenerationSpec,
    MusicValidationError,
    NoteEvent,
    RestEvent,
    Section,
    TempoMark,
    Track,
    export_project,
    generate_arrangement,
    validate_project,
)


def test_generated_arrangement_validation_report_passes():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=808),
        project_id="valid-arrangement",
    )

    report = validate_project(project)

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["by_track"] == {}
    assert report["metrics"]["harmony_score"] is not None


def test_bar_duration_and_range_errors_are_reported_by_track_and_bar():
    project = _one_track_project(
        Track(
            id="alto_sax",
            instrument="alto_sax",
            role="melody",
            bars=[
                Bar(
                    number=1,
                    events=[
                        NoteEvent(pitch="C7", start=0, duration=1),
                        RestEvent(start=1, duration=2),
                    ],
                )
            ],
        )
    )

    report = validate_project(project)

    assert report["status"] == "fail"
    assert _has_issue(report, "BarDurationValidator", "bar_duration")
    assert _has_issue(report, "InstrumentRangeValidator", "outside_absolute_range")
    assert "alto_sax" in report["by_track"]
    assert "alto_sax:1" in report["by_bar"]


def test_transposition_errors_are_reported_for_impossible_written_pitch():
    project = _one_track_project(
        Track(
            id="trumpet_bflat",
            instrument="trumpet_bflat",
            role="melody",
            bars=[
                Bar(
                    number=1,
                    events=[
                        NoteEvent(pitch="C-1", start=0, duration=1),
                        RestEvent(start=1, duration=3),
                    ],
                )
            ],
        )
    )

    report = validate_project(project)

    assert report["status"] == "fail"
    assert _has_issue(
        report,
        "TranspositionValidator",
        "sounding_pitch_out_of_midi_range",
    )


def test_harmony_and_breath_warnings_do_not_fail_report():
    project = _one_track_project(
        Track(
            id="alto_sax",
            instrument="alto_sax",
            role="melody",
            bars=[
                Bar(number=bar, events=[NoteEvent(pitch="F#4", start=0, duration=4)])
                for bar in range(1, 6)
            ],
        ),
        bars=5,
    )

    report = validate_project(project)

    assert report["status"] == "pass_with_warnings"
    assert _has_issue(report, "HarmonyValidator", "low_harmony_score")
    assert _has_issue(report, "BreathValidator", "no_breaths")
    assert _has_issue(report, "BreathValidator", "phrase_too_long")


def test_piano_voicing_and_drum_validator_cases_are_reported():
    project = ArrangementProject(
        project_id="voicing-drum-check",
        generation_spec=GenerationSpec(duration_bars=1),
        tempo_map=[TempoMark(bar=1, bpm=132)],
        form=[Section(name="Head", start_bar=1, end_bar=1)],
        chord_grid=[ChordSymbol(symbol="Cmaj7", bar=1, beat=1)],
        tracks=[
            Track(
                id="piano",
                instrument="piano",
                role="comping",
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(pitch=pitch, start=0, duration=4)
                            for pitch in ["C3", "E3", "G3", "B3", "D4", "F#4", "A4"]
                        ],
                    )
                ],
            ),
            Track(
                id="drum_kit",
                instrument="drum_kit",
                role="drums",
                channel=1,
                bars=[
                    Bar(
                        number=1,
                        events=[
                            NoteEvent(
                                pitch="A5",
                                start=0,
                                duration=0.5,
                                annotations={"drum": "bad"},
                            ),
                            RestEvent(start=0.5, duration=3.5),
                        ],
                    )
                ],
            ),
        ],
    )

    report = validate_project(project)

    assert report["status"] == "fail"
    assert _has_issue(report, "PianoPlayabilityValidator", "too_many_notes")
    assert _has_issue(report, "DrumValidator", "wrong_midi_channel")
    assert _has_issue(report, "DrumValidator", "unsupported_drum_pitch")


def test_strict_export_blocks_serious_validation_errors_and_writes_report(tmp_path):
    project = _one_track_project(
        Track(
            id="alto_sax",
            instrument="alto_sax",
            role="melody",
            bars=[
                Bar(
                    number=1,
                    events=[
                        NoteEvent(pitch="C7", start=0, duration=1),
                        RestEvent(start=1, duration=3),
                    ],
                )
            ],
        )
    )

    with pytest.raises(MusicValidationError) as exc_info:
        export_project(project, tmp_path, include_pdf=False)

    assert exc_info.value.report["status"] == "fail"
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "validation_report.html").exists()
    assert not (tmp_path / "full_arrangement.mid").exists()


def test_report_only_export_records_validation_and_export_reports(tmp_path):
    project = _one_track_project(
        Track(
            id="alto_sax",
            instrument="alto_sax",
            role="melody",
            bars=[
                Bar(
                    number=1,
                    events=[
                        NoteEvent(pitch="C7", start=0, duration=1),
                        RestEvent(start=1, duration=3),
                    ],
                )
            ],
        )
    )

    manifest = export_project(
        project,
        tmp_path,
        include_pdf=False,
        validation_policy="report_only",
    )

    assert manifest["status"] == "exported"
    assert project.validation_report["status"] == "fail"
    assert any(file["kind"] == "validation_report_html" for file in manifest["files"])
    assert (tmp_path / "full_score.musicxml").exists()


def _one_track_project(track: Track, *, bars: int = 1) -> ArrangementProject:
    return ArrangementProject(
        project_id="validator-broken",
        generation_spec=GenerationSpec(duration_bars=bars),
        tempo_map=[TempoMark(bar=1, bpm=132)],
        form=[Section(name="Head", start_bar=1, end_bar=bars)],
        chord_grid=[
            ChordSymbol(symbol="Cm7", bar=bar, beat=1)
            for bar in range(1, bars + 1)
        ],
        tracks=[track],
    )


def _has_issue(report, validator, code):
    return any(
        issue["validator"] == validator and issue["code"] == code
        for issue in [*report["errors"], *report["warnings"]]
    )
