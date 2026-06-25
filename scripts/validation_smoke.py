from __future__ import annotations

import json
from pathlib import Path

from arranger_core import GenerationSpec, export_project, generate_arrangement, validate_project

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/obj8_validation_demo"


def main() -> None:
    spec = GenerationSpec(
        ensemble="jazz_sextet",
        form="minor_blues_12",
        complexity=0.65,
        seed=4808,
    )
    project = generate_arrangement(spec, project_id="obj8-validation-demo")
    preflight_report = validate_project(project)
    if preflight_report["status"] == "fail":
        raise RuntimeError("Generated arrangement failed preflight validation")

    manifest = export_project(project, OUTPUT_DIR, include_pdf=False)
    json_path = OUTPUT_DIR / "validation_report.json"
    html_path = OUTPUT_DIR / "validation_report.html"
    report = json.loads(json_path.read_text(encoding="utf-8"))

    if not html_path.exists():
        raise RuntimeError("Validation HTML report was not written")
    if report["status"] == "fail":
        raise RuntimeError("Exported arrangement has validation errors")
    if report["metrics"].get("midi_track_files") != len(project.tracks):
        raise RuntimeError("Export validation did not see every MIDI track file")

    print(
        "Validation smoke OK: "
        f"{project.bar_count} bars, {len(project.tracks)} tracks, "
        f"status={report['status']}, files={len(manifest['files'])}"
    )


if __name__ == "__main__":
    main()
