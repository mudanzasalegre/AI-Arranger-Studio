from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for package in ("arranger_core", "model_backends", "training"):
    package_path = str(ROOT / "packages" / package)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from arranger_core import ProQualityGate, load_project_json  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = _repo_path(args.output_dir) if args.output_dir else None
    project_path = _repo_path(args.project) if args.project else _project_path(output_dir)
    project = load_project_json(project_path)
    report = ProQualityGate(thresholds_path=_repo_path(args.thresholds)).evaluate(
        project,
        validation_report=_read_json(_repo_path(args.validation_report))
        if args.validation_report
        else None,
        output_dir=output_dir,
        export_manifest=_read_json(_repo_path(args.export_manifest))
        if args.export_manifest
        else None,
        model_trace=_read_json(_repo_path(args.model_trace)) if args.model_trace else None,
        takes_manifest=_read_json(_repo_path(args.takes_manifest))
        if args.takes_manifest
        else None,
        export_mode=args.export_mode,
        min_rating=args.min_rating,
        required_tracks=args.required_track,
        require_export_files=not args.pre_export,
    )

    json_path = _report_path(args.report_json, output_dir, "pro_quality_report.json")
    md_path = _report_path(args.report_md, output_dir, "pro_quality_report.md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(report_markdown(report, project_path=project_path), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the professional quality gate.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--thresholds", default="configs/quality_thresholds.pro.yaml")
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--export-manifest", default=None)
    parser.add_argument("--model-trace", default=None)
    parser.add_argument("--takes-manifest", default=None)
    parser.add_argument("--export-mode", choices=["private", "commercial"], default="private")
    parser.add_argument("--min-rating", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--required-track", action="append", default=[])
    parser.add_argument("--pre-export", action="store_true")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    return parser.parse_args(argv)


def report_markdown(report: dict[str, Any], *, project_path: Path) -> str:
    metrics = report.get("metrics", {})
    project = metrics.get("project", {})
    lines = [
        "# Pro Quality Gate Report",
        "",
        f"Project: `{project_path}`",
        f"Status: `{report.get('status')}`",
        f"Rating: `{report.get('rating')}`",
        f"Score: `{report.get('score')}`",
        f"Release candidate: `{report.get('release_candidate')}`",
        "",
        "## Project",
        "",
        f"- Bars: `{project.get('bars')}`",
        f"- Tracks: `{project.get('tracks')}`",
        f"- Note events: `{project.get('note_events')}`",
        f"- Validation errors: `{metrics.get('validation', {}).get('errors')}`",
        f"- Validation warnings: `{metrics.get('validation', {}).get('warnings')}`",
        "",
        "## Blocking Errors",
        "",
    ]
    blocking = report.get("blocking_errors", [])
    if blocking:
        lines.extend(f"- {item}" for item in blocking)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(
            f"- `{item.get('code')}`"
            + (f" ({item.get('track_id')})" if item.get("track_id") else "")
            + f": {item.get('message')}"
            for item in warnings
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Tracks", ""])
    lines.append("| Track | Role | Notes | Active bars | Large leaps |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for track in metrics.get("tracks", []):
        lines.append(
            "| "
            f"`{track.get('track_id')}` | "
            f"`{track.get('role')}` | "
            f"{track.get('note_count')} | "
            f"{track.get('active_bar_ratio')} | "
            f"{track.get('large_leaps')} |"
        )
    return "\n".join(lines) + "\n"


def _project_path(output_dir: Path | None) -> Path:
    if output_dir is None:
        raise SystemExit("Pass --project or --output-dir.")
    return output_dir / "arrangement_project.json"


def _report_path(raw: str | None, output_dir: Path | None, default_name: str) -> Path:
    if raw:
        path = _repo_path(raw)
    elif output_dir is not None:
        path = output_dir / default_name
    else:
        path = ROOT / "outputs" / "pro_benchmarks" / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
