from __future__ import annotations

from arranger_core import (
    ProQualityGate,
    compile_prompt,
    export_project,
    generate_arrangement,
    validate_project,
)


def test_pro_quality_gate_passes_exported_arrangement(tmp_path):
    project = generate_arrangement(
        compile_prompt(
            "hard bop minor blues sextet, alto sax trumpet trombone piano bass drums",
            seed=3600,
        ),
        project_id="quality_gate_pass",
    )
    output_dir = tmp_path / "exported"
    export_manifest = export_project(
        project,
        output_dir,
        include_pdf=False,
        validation_policy="strict",
        export_mode="private",
    )
    report = ProQualityGate(thresholds=_test_thresholds()).evaluate(
        project,
        validation_report=project.validation_report,
        output_dir=output_dir,
        export_manifest=export_manifest,
        export_mode="private",
        min_rating="B",
        required_tracks=["drums", "double_bass", "piano", "alto_sax"],
    )

    assert report["status"] == "pass"
    assert report["rating"] in {"A", "B"}
    assert report["blocking_errors"] == []
    assert report["metrics"]["export"]["status"] == "exported"


def test_pro_quality_gate_blocks_empty_required_track(tmp_path):
    project = generate_arrangement(
        compile_prompt("hard bop minor blues trio, piano bass drums", seed=3601),
        project_id="quality_gate_empty",
    )
    project.tracks[0].bars = [
        bar.model_copy(update={"events": []}) for bar in project.tracks[0].bars
    ]
    validation = validate_project(project)

    report = ProQualityGate(thresholds=_test_thresholds()).evaluate(
        project,
        validation_report=validation,
        export_mode="private",
        min_rating="B",
        required_tracks=["drums", "double_bass", "piano"],
        require_export_files=False,
    )

    assert report["status"] == "fail"
    assert any(error.startswith("empty_track:") for error in report["blocking_errors"])


def test_pro_quality_gate_blocks_commercial_review_required_model():
    project = generate_arrangement(
        compile_prompt("hard bop minor blues trio, piano bass drums", seed=3602),
        project_id="quality_gate_commercial",
    )
    validation = validate_project(project)

    report = ProQualityGate(thresholds=_test_thresholds()).evaluate(
        project,
        validation_report=validation,
        model_trace={
            "schema_version": "0.1.0",
            "model_artifacts": [
                {
                    "backend_id": "midigpt",
                    "task": "infill_bars",
                    "commercial_use": "review_required",
                }
            ],
        },
        export_mode="commercial",
        min_rating="B",
        require_export_files=False,
    )

    assert report["status"] == "fail"
    assert any("model_license_incompatible" in error for error in report["blocking_errors"])


def _test_thresholds() -> dict:
    return {
        "global": {
            "max_blocking_errors": 0,
            "min_tracks": 3,
            "min_note_events": 20,
            "require_full_midi": True,
            "require_musicxml": True,
            "require_model_trace": False,
            "reject_pending_takes_on_export": True,
        },
        "bass": {
            "min_beat1_root_score": 0.25,
            "min_approach_to_next_root_score": 0.0,
            "max_large_leaps": 16,
            "min_active_bar_ratio": 0.5,
        },
        "piano": {
            "max_rootless_violations": 64,
            "min_avg_voicing_size": 1.0,
            "max_voicing_size": 8,
            "max_low_register_notes_below_midi": 999,
        },
        "drums": {
            "min_drum_pitch_count": 1,
            "min_fill_bar_count": 0,
            "min_velocity_stddev": 0.0,
        },
        "melody": {
            "min_breath_rest_count": 0,
            "max_large_leaps": 99,
            "min_active_bar_ratio": 0.0,
        },
        "horns": {
            "min_breath_rest_count": 0,
            "max_large_leaps": 99,
            "max_density_per_bar": 99,
        },
        "ratings": {
            "A": {"min_score": 0.88},
            "B": {"min_score": 0.72},
            "C": {"min_score": 0.55},
            "D": {"min_score": 0.0},
        },
    }
