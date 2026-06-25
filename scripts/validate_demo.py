from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arranger_core import (
    ArrangementProject,
    merge_validation_reports,
    validate_export_package,
    validate_project,
    write_validation_html,
    write_validation_json,
)
from demo_jazz import DEFAULT_OUTPUT_DIR, build_demo


def validate_demo(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    generate_if_missing: bool = True,
    include_pdf: bool = True,
) -> dict[str, Any]:
    """Validate the canonical demo project and its exported file package."""

    output_path = Path(output_dir)
    project_path = output_path / "arrangement_project.json"
    manifest_path = output_path / "export_manifest.json"
    if generate_if_missing and (not project_path.exists() or not manifest_path.exists()):
        build_demo(output_dir=output_path, include_pdf=include_pdf, clean=False)

    if not project_path.exists():
        raise FileNotFoundError(f"Missing demo project: {project_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")

    project = ArrangementProject.load_json(project_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_report = validate_project(project)
    export_report = validate_export_package(project, manifest, output_path)
    report = merge_validation_reports(project_report, export_report)

    write_validation_json(report, output_path / "validation_report.json")
    write_validation_html(report, output_path / "validation_report.html")

    summary = {
        "status": report["status"],
        "project_id": project.project_id,
        "output_dir": str(output_path),
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
        "metrics": report["metrics"],
    }
    (output_path / "validate_demo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["errors"]:
        raise RuntimeError(f"Demo validation failed: {report['errors']}")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the canonical jazz demo package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generate-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the demo first when exported files are missing.",
    )
    parser.add_argument(
        "--include-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate PDFs if a missing demo has to be created.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = validate_demo(
        output_dir=args.output_dir,
        generate_if_missing=args.generate_if_missing,
        include_pdf=args.include_pdf,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
