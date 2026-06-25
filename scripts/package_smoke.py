from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from demo_jazz import DEFAULT_OUTPUT_DIR, DEFAULT_PRESET_ID, build_demo
from validate_demo import validate_demo
from zip_demo import create_demo_zip


def run_package_smoke(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    zip_path: str | Path | None = None,
    preset_id: str = DEFAULT_PRESET_ID,
    include_pdf: bool = True,
    clean: bool = True,
) -> dict[str, Any]:
    """Run the full generation, validation and ZIP smoke test."""

    output_path = Path(output_dir)
    demo_summary = build_demo(
        output_dir=output_path,
        preset_id=preset_id,
        include_pdf=include_pdf,
        clean=clean,
    )
    validation_summary = validate_demo(
        output_dir=output_path,
        generate_if_missing=False,
        include_pdf=include_pdf,
    )
    zip_summary = create_demo_zip(
        output_dir=output_path,
        zip_path=zip_path,
        generate_if_missing=False,
        include_pdf=include_pdf,
    )

    summary = {
        "status": "pass",
        "output_dir": str(output_path),
        "zip_path": zip_summary["zip_path"],
        "steps": {
            "demo_jazz": demo_summary,
            "validate_demo": validation_summary,
            "zip_demo": zip_summary,
        },
    }
    (output_path / "package_smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Objective 13 packaging smoke test end to end."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=None)
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
    summary = run_package_smoke(
        output_dir=args.output_dir,
        zip_path=args.zip_path,
        preset_id=args.preset_id,
        include_pdf=args.include_pdf,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
