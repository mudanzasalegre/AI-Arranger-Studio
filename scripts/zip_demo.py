from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from demo_jazz import DEFAULT_OUTPUT_DIR, DEMO_REQUIRED_RELATIVE_FILES, build_demo
from validate_demo import validate_demo


def create_demo_zip(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    zip_path: str | Path | None = None,
    generate_if_missing: bool = True,
    include_pdf: bool = True,
) -> dict[str, Any]:
    """Create and verify a ZIP archive for the canonical demo package."""

    output_path = Path(output_dir)
    project_path = output_path / "arrangement_project.json"
    manifest_path = output_path / "export_manifest.json"
    if generate_if_missing and (not project_path.exists() or not manifest_path.exists()):
        build_demo(output_dir=output_path, include_pdf=include_pdf, clean=False)

    validate_demo(
        output_dir=output_path,
        generate_if_missing=generate_if_missing,
        include_pdf=include_pdf,
    )

    archive_path = Path(zip_path) if zip_path is not None else output_path.with_suffix(".zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(path for path in output_path.rglob("*") if path.is_file()):
            if file_path.resolve() == archive_resolved:
                continue
            archive.write(file_path, file_path.relative_to(output_path).as_posix())

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    missing = sorted(DEMO_REQUIRED_RELATIVE_FILES - names)
    if missing:
        raise RuntimeError(f"Demo ZIP is missing required files: {missing}")

    summary = {
        "status": "pass",
        "output_dir": str(output_path),
        "zip_path": str(archive_path),
        "file_count": len(names),
        "bytes": archive_path.stat().st_size,
        "required_files": sorted(DEMO_REQUIRED_RELATIVE_FILES),
    }
    (output_path / "zip_demo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the canonical jazz demo ZIP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=None)
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
    summary = create_demo_zip(
        output_dir=args.output_dir,
        zip_path=args.zip_path,
        generate_if_missing=args.generate_if_missing,
        include_pdf=args.include_pdf,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
