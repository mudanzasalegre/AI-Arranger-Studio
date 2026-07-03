from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "models" / "professional_generation_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("professional_generation_benchmark", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_case_quality_gates_pass_with_required_files(tmp_path):
    module = _load_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "full_arrangement.mid").write_bytes(b"MThd")
    (case_dir / "full_score.musicxml").write_text("<score-partwise />", encoding="utf-8")

    gate = module._case_quality_gates(
        case_dir=case_dir,
        generated={
            "project": {
                "tracks": [
                    {"id": "drum_kit"},
                    {"id": "double_bass"},
                    {"id": "piano"},
                    {"id": "trumpet_bflat"},
                ]
            }
        },
        validation={"errors": [], "warnings": []},
        quality_gates={
            "validation_errors_max": 0,
            "validation_warnings_max": 25,
            "min_tracks": 3,
            "require_full_midi": True,
            "require_musicxml": True,
            "require_model_trace_if_ai_used": True,
            "require_no_pending_takes_in_export": True,
        },
        required_tracks=["drums", "double_bass", "piano", "trumpet"],
        model_trace={"status": "no_model_artifacts", "model_artifacts": []},
        takes_manifest={"takes": [{"status": "accepted"}]},
        package_names=set(module.REQUIRED_PACKAGE_FILES),
        accepted_artifact_ids=[],
    )

    assert gate["blocking_errors"] == []
    assert gate["missing_required_tracks"] == []
    assert gate["required_tracks_present"] == ["double_bass", "drums", "piano", "trumpet"]


def test_case_quality_gates_fail_on_pending_take_and_missing_track(tmp_path):
    module = _load_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "full_arrangement.mid").write_bytes(b"MThd")
    (case_dir / "full_score.musicxml").write_text("<score-partwise />", encoding="utf-8")

    gate = module._case_quality_gates(
        case_dir=case_dir,
        generated={"project": {"tracks": [{"id": "drums"}, {"id": "piano"}]}},
        validation={"errors": [], "warnings": []},
        quality_gates={"min_tracks": 3, "require_no_pending_takes_in_export": True},
        required_tracks=["drums", "double_bass", "piano"],
        model_trace={},
        takes_manifest={"takes": [{"take_id": "take_1", "status": "pending"}]},
        package_names={"full_arrangement.mid"},
        accepted_artifact_ids=[],
    )

    assert any("missing required tracks" in error for error in gate["blocking_errors"])
    assert any("Export includes pending takes" in error for error in gate["blocking_errors"])
    assert any("package.zip missing required files" in error for error in gate["blocking_errors"])


def test_copy_export_files_preserves_manifest_relative_paths(tmp_path):
    module = _load_module()
    project_dir = tmp_path / "api" / "projects" / "bench"
    case_dir = tmp_path / "benchmark" / "bench"
    midi_tracks = project_dir / "midi_tracks"
    midi_tracks.mkdir(parents=True)
    (project_dir / "full_arrangement.mid").write_bytes(b"MThd")
    (midi_tracks / "piano.mid").write_bytes(b"MTrk")
    (project_dir / "export_manifest.json").write_text("{}", encoding="utf-8")
    manifest = {
        "files": [
            {"kind": "midi_full", "path": str(project_dir / "full_arrangement.mid")},
            {"kind": "midi_track", "path": str(midi_tracks / "piano.mid")},
            {
                "kind": "pdf_full",
                "path": str(project_dir / "full_score.pdf"),
                "status": "skipped",
            },
        ]
    }

    copied_manifest = module._copy_export_files(project_dir, case_dir, manifest)

    assert (case_dir / "full_arrangement.mid").exists()
    assert (case_dir / "midi_tracks" / "piano.mid").exists()
    assert not (case_dir / "full_score.pdf").exists()
    assert copied_manifest["copied_files"] == [
        "export_manifest.json",
        "full_arrangement.mid",
        "midi_tracks/piano.mid",
    ]


def test_final_artifact_statuses_require_validated_or_rejected(tmp_path):
    module = _load_module()
    artifact_root = tmp_path / "model_artifacts"
    artifact_root.mkdir()
    (artifact_root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"artifact_id": "artifact_valid", "status": "validated"},
                    {"artifact_id": "artifact_rejected", "status": "rejected"},
                ]
            }
        ),
        encoding="utf-8",
    )

    statuses = module._assert_final_artifact_statuses(
        artifact_root,
        ["artifact_valid", "artifact_rejected"],
    )

    assert statuses == {
        "artifact_valid": "validated",
        "artifact_rejected": "rejected",
    }


def test_downloaded_package_names_can_be_checked(tmp_path):
    module = _load_module()
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("full_arrangement.mid", "midi")
        archive.writestr("full_score.musicxml", "xml")

    assert module._zip_names(archive_path) == {
        "full_arrangement.mid",
        "full_score.musicxml",
    }
