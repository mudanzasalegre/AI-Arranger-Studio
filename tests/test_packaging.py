from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_makefile_exposes_objective_13_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in [
        "demo-jazz:",
        "export-demo:",
        "validate-demo:",
        "zip-demo:",
        "package-smoke:",
        "demo-check: package-smoke",
    ]:
        assert target in makefile


def test_demo_jazz_validate_and_zip_flow_without_pdf(tmp_path):
    output_dir = tmp_path / "demo"
    zip_path = tmp_path / "demo.zip"

    _run_script(
        "demo_jazz.py",
        "--output-dir",
        str(output_dir),
        "--no-include-pdf",
        "--no-clean",
    )
    _run_script(
        "validate_demo.py",
        "--output-dir",
        str(output_dir),
        "--no-generate-if-missing",
        "--no-include-pdf",
    )
    _run_script(
        "zip_demo.py",
        "--output-dir",
        str(output_dir),
        "--zip-path",
        str(zip_path),
        "--no-generate-if-missing",
        "--no-include-pdf",
    )

    summary = json.loads((output_dir / "demo_jazz_summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_dir / "validate_demo_summary.json").read_text(encoding="utf-8")
    )
    zip_summary = json.loads((output_dir / "zip_demo_summary.json").read_text(encoding="utf-8"))

    assert summary["preset_id"] == "jazz_hard_bop_minor_blues_sextet"
    assert summary["pdf_status"] == "skipped"
    assert validation["errors"] == 0
    assert zip_summary["status"] == "pass"
    assert zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    assert {
        "arrangement_project.json",
        "export_manifest.json",
        "full_arrangement.mid",
        "full_score.musicxml",
        "midi_tracks/drums.mid",
        "validation_report.json",
        "validation_report.html",
    } <= names


def _run_script(script_name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
