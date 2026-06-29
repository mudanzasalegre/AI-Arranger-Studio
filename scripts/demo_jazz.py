from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from arranger_core import (
    PresetLibrary,
    export_project,
    find_musescore_cli,
    generate_arrangement,
    validate_project,
)
from music21 import converter

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "obj13_demo_jazz"
DEFAULT_PRESET_ID = "jazz_hard_bop_minor_blues_sextet"

DEMO_REQUIRED_RELATIVE_FILES = {
    "arrangement_project.json",
    "export_manifest.json",
    "full_arrangement.mid",
    "full_score.musicxml",
    "generation_spec.json",
    "midi_tracks/alto_sax.mid",
    "midi_tracks/double_bass.mid",
    "midi_tracks/drums.mid",
    "midi_tracks/piano.mid",
    "midi_tracks/trombone.mid",
    "midi_tracks/trumpet_bflat.mid",
    "model_trace.json",
    "session_readme.md",
    "takes_manifest.json",
    "validation_report.html",
    "validation_report.json",
}


def build_demo(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    preset_id: str = DEFAULT_PRESET_ID,
    include_pdf: bool = True,
    clean: bool = True,
    require_pdf_when_available: bool = True,
) -> dict[str, Any]:
    """Generate and export the canonical Objective 13 demo package."""

    output_path = Path(output_dir)
    if clean:
        _clean_output_dir(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    preset = PresetLibrary.load_default().get(preset_id)
    project = generate_arrangement(preset.spec, project_id=preset.id)
    report = validate_project(project)
    if report["errors"]:
        raise RuntimeError(f"{preset.id} validation errors: {report['errors']}")

    musescore = find_musescore_cli()
    manifest = export_project(project, output_path, include_pdf=include_pdf)
    converter.parse(output_path / "full_score.musicxml")

    if (
        include_pdf
        and require_pdf_when_available
        and musescore is not None
        and manifest["pdf_status"] != "created"
    ):
        raise RuntimeError("MuseScore CLI is available but PDF export was not created")

    missing = sorted(
        relative_path
        for relative_path in DEMO_REQUIRED_RELATIVE_FILES
        if not (output_path / relative_path).exists()
    )
    if missing:
        raise RuntimeError(f"Demo export is missing required files: {missing}")

    summary = {
        "status": "pass",
        "preset_id": preset.id,
        "project_id": project.project_id,
        "style": preset.spec.style,
        "form": preset.spec.form,
        "key": preset.spec.key,
        "tempo": preset.spec.tempo,
        "bars": project.bar_count,
        "tracks": len(project.tracks),
        "validation_status": report["status"],
        "warnings": len(report["warnings"]),
        "output_dir": str(output_path),
        "pdf_status": manifest["pdf_status"],
        "musescore_cli": str(musescore) if musescore else None,
        "exported_files": len(manifest["files"]),
        "required_files": sorted(DEMO_REQUIRED_RELATIVE_FILES),
    }
    (output_path / "demo_jazz_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _clean_output_dir(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_outputs = OUTPUTS_ROOT.resolve()
    if resolved_path == resolved_outputs or not resolved_path.is_relative_to(resolved_outputs):
        raise RuntimeError(f"Refusing to clean path outside outputs/: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the canonical jazz demo package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset-id", default=DEFAULT_PRESET_ID)
    parser.add_argument(
        "--include-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PDFs when MuseScore CLI is available.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the output directory before generating. Cleaning is limited to outputs/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_demo(
        output_dir=args.output_dir,
        preset_id=args.preset_id,
        include_pdf=args.include_pdf,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
