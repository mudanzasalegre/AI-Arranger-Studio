from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
for package in ("arranger_core", "model_backends", "training"):
    package_path = str(ROOT / "packages" / package)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from arranger_core import (  # noqa: E402
    ProfessionalGenerationOptions,
    ProfessionalGenerationOrchestrator,
    ProQualityGate,
    load_project_json,
)

RATING_POINTS = {"A": 4, "B": 3, "C": 2, "D": 1}
COMMERCIAL_BLOCK_CODES = {
    "model_license_forbidden",
    "model_license_incompatible",
    "dataset_license_blocked",
    "dataset_commercial_use_incompatible",
}
DEFAULT_BENCHMARKS = [
    {
        "id": "hard_bop_minor_blues_sextet",
        "prompt": (
            "hard bop nocturno en Do menor, 132 bpm, blues menor de 12 compases, "
            "sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria, "
            "bajo caminante, piano rootless, bateria swing y shout chorus final"
        ),
        "seed": 3601,
        "required_tracks": [
            "drums",
            "double_bass",
            "piano",
            "alto_sax",
            "trumpet",
            "trombone",
        ],
    },
    {
        "id": "bebop_blues_quintet",
        "prompt": (
            "bebop blues en Fa, 184 bpm, quinteto con saxo alto, trompeta, piano, "
            "contrabajo y bateria, lineas agiles, hits de metales y walking bass"
        ),
        "seed": 3602,
        "required_tracks": ["drums", "double_bass", "piano", "alto_sax", "trumpet"],
    },
    {
        "id": "modal_jazz_quartet",
        "prompt": (
            "jazz modal en Re dorico, 112 bpm, cuarteto con saxo tenor, piano, "
            "contrabajo y bateria, vamp amplio, voicings cuartales y espacio"
        ),
        "seed": 3603,
        "required_tracks": ["drums", "double_bass", "piano", "tenor_sax"],
    },
    {
        "id": "jazz_ballad_trio",
        "prompt": (
            "balada jazz en Mi bemol mayor, 72 bpm, trio de piano, contrabajo y bateria, "
            "armonia rica, voicings abiertos, escobillas y respiracion amplia"
        ),
        "seed": 3604,
        "required_tracks": ["drums", "double_bass", "piano"],
    },
    {
        "id": "bossa_quartet",
        "prompt": (
            "bossa nova jazz en Sol mayor, 126 bpm, cuarteto con flauta, piano, "
            "contrabajo y bateria, bajo bossa, comping sincopado y melodia suave"
        ),
        "seed": 3605,
        "required_tracks": ["drums", "double_bass", "piano", "flute"],
    },
]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_root = _repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    benchmarks = _load_benchmarks(_repo_path(args.benchmark_config))[: args.max_demos]
    orchestrator = ProfessionalGenerationOrchestrator(repo_root=ROOT)
    quality_gate = ProQualityGate(thresholds_path=_repo_path(args.quality_thresholds))

    case_summaries = [
        _run_case(
            item,
            orchestrator=orchestrator,
            quality_gate=quality_gate,
            output_root=output_root,
            args=args,
        )
        for item in benchmarks
    ]
    summary = _aggregate_summary(
        case_summaries,
        output_root=output_root,
        min_demos=args.min_demos,
        min_average_rating=args.min_average_rating,
        export_mode=args.export_mode,
    )
    (output_root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "benchmark_summary.md").write_text(
        benchmark_markdown(summary),
        encoding="utf-8",
    )
    (output_root / "release_candidate_report.json").write_text(
        json.dumps(_release_candidate_report(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "release_candidate_report.md").write_text(
        release_candidate_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the professional benchmark gate.")
    parser.add_argument(
        "--benchmark-config",
        default="configs/professional_benchmark_gate.pro.yaml",
    )
    parser.add_argument("--ai-config", default="configs/ai_models.pro.yaml")
    parser.add_argument("--quality-thresholds", default="configs/quality_thresholds.pro.yaml")
    parser.add_argument("--output-root", default="outputs/pro_benchmarks")
    parser.add_argument("--profile", default="benchmark_pro")
    parser.add_argument("--export-mode", choices=["private", "commercial"], default="private")
    parser.add_argument("--min-rating", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--min-average-rating", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--max-demos", type=int, default=5)
    parser.add_argument("--min-demos", type=int, default=5)
    parser.add_argument("--max-ai-attempts", type=int, default=3)
    parser.add_argument("--include-pdf", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument(
        "--use-llm-planner",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-custom-role-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-midigpt-infill",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-text2midi-sketch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _run_case(
    item: dict[str, Any],
    *,
    orchestrator: ProfessionalGenerationOrchestrator,
    quality_gate: ProQualityGate,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = f"benchmark_{item['id']}"
    result = orchestrator.generate(
        ProfessionalGenerationOptions(
            prompt=str(item["prompt"]),
            profile=args.profile,
            seed=int(item.get("seed", 0)),
            run_id=run_id,
            output_root=str(output_root),
            ai_config_path=args.ai_config,
            quality_thresholds_path=args.quality_thresholds,
            export_mode=args.export_mode,
            include_pdf=args.include_pdf,
            min_rating=args.min_rating,
            use_llm_planner=args.use_llm_planner,
            use_rule_based_base=True,
            use_custom_role_models=args.use_custom_role_models,
            use_midigpt_infill=args.use_midigpt_infill,
            use_text2midi_sketch=args.use_text2midi_sketch,
            max_ai_attempts=args.max_ai_attempts,
            clean=not args.no_clean,
        )
    )
    output_dir = Path(result.output_dir)
    project = load_project_json(output_dir / "arrangement_project.json")
    report = quality_gate.evaluate(
        project,
        validation_report=result.validation,
        output_dir=output_dir,
        export_manifest=result.export_manifest,
        model_trace=result.model_trace,
        takes_manifest=result.takes_manifest,
        export_mode=args.export_mode,
        min_rating=args.min_rating,
        required_tracks=[str(track_id) for track_id in item.get("required_tracks", [])],
        require_export_files=True,
    )
    (output_dir / "pro_quality_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    blocking_errors = list(report.get("blocking_errors", []))
    if result.status != "ok":
        blocking_errors.append(f"orchestrator_status: {result.status}")
    status = "ok" if result.status == "ok" and report["status"] == "pass" else "fail"
    return {
        "id": item["id"],
        "status": status,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "prompt": item["prompt"],
        "seed": item.get("seed", 0),
        "required_tracks": item.get("required_tracks", []),
        "orchestrator_status": result.status,
        "rating": report.get("rating"),
        "score": report.get("score"),
        "blocking_errors": blocking_errors,
        "warning_count": len(report.get("warnings", [])),
        "model_backends": report.get("metrics", {}).get("model_trace", {}).get("backends", []),
        "exported": bool(result.export_manifest),
        "commercial_blockers": _commercial_blockers(report),
        "quality_report": str(output_dir / "pro_quality_report.json"),
        "files": result.files,
    }


def _aggregate_summary(
    cases: list[dict[str, Any]],
    *,
    output_root: Path,
    min_demos: int,
    min_average_rating: str,
    export_mode: str,
) -> dict[str, Any]:
    ratings = [case.get("rating") for case in cases if case.get("rating") in RATING_POINTS]
    average_points = (
        round(sum(RATING_POINTS[str(rating)] for rating in ratings) / len(ratings), 3)
        if ratings
        else 0.0
    )
    generated = len(cases)
    blocking_error_count = sum(len(case.get("blocking_errors", [])) for case in cases)
    commercial_blocker_count = sum(len(case.get("commercial_blockers", [])) for case in cases)
    min_average_points = RATING_POINTS[min_average_rating]
    status = (
        "ok"
        if generated >= min_demos
        and blocking_error_count == 0
        and average_points >= min_average_points
        and commercial_blocker_count == 0
        else "fail"
    )
    return {
        "schema_version": "0.1.0",
        "status": status,
        "output_root": str(output_root),
        "export_mode": export_mode,
        "generated_demos": generated,
        "min_demos": min_demos,
        "blocking_error_count": blocking_error_count,
        "commercial_blocker_count": commercial_blocker_count,
        "ratings": dict(sorted(_rating_counts(cases).items())),
        "average_rating_points": average_points,
        "average_rating": _rating_from_points(average_points),
        "min_average_rating": min_average_rating,
        "release_candidate_count": sum(1 for case in cases if case.get("rating") == "A"),
        "cases": cases,
    }


def benchmark_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Professional Benchmark Gate",
        "",
        f"Status: `{summary['status']}`",
        f"Generated demos: `{summary['generated_demos']}`",
        f"Blocking errors: `{summary['blocking_error_count']}`",
        f"Average rating: `{summary['average_rating']}` ({summary['average_rating_points']})",
        f"Commercial blockers: `{summary['commercial_blocker_count']}`",
        "",
        "| Demo | Status | Rating | Score | Warnings | Output |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            "| "
            f"`{case['id']}` | "
            f"{case['status']} | "
            f"{case.get('rating')} | "
            f"{case.get('score')} | "
            f"{case.get('warning_count')} | "
            f"`{case.get('output_dir')}` |"
        )
    lines.extend(["", "## Blocking Errors", ""])
    blockers = [
        (case["id"], error)
        for case in summary["cases"]
        for error in case.get("blocking_errors", [])
    ]
    if blockers:
        lines.extend(f"- `{case_id}`: {error}" for case_id, error in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def release_candidate_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Release Candidate Report",
        "",
        f"Benchmark status: `{summary['status']}`",
        f"Release candidates: `{summary['release_candidate_count']}`",
        f"Average rating: `{summary['average_rating']}`",
        "",
        "## Candidate Demos",
        "",
    ]
    candidates = [case for case in summary["cases"] if case.get("rating") == "A"]
    if candidates:
        lines.extend(
            f"- `{case['id']}`: `{case.get('output_dir')}`" for case in candidates
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Gate Criteria", ""])
    lines.extend(
        [
            f"- At least {summary['min_demos']} demos: `{summary['generated_demos']}`",
            f"- Blocking errors: `{summary['blocking_error_count']}`",
            f"- Average rating >= {summary['min_average_rating']}: `{summary['average_rating']}`",
            f"- Commercial blockers: `{summary['commercial_blocker_count']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _release_candidate_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "status": summary["status"],
        "release_candidate_count": summary["release_candidate_count"],
        "average_rating": summary["average_rating"],
        "average_rating_points": summary["average_rating_points"],
        "criteria": {
            "generated_demos": summary["generated_demos"],
            "min_demos": summary["min_demos"],
            "blocking_error_count": summary["blocking_error_count"],
            "commercial_blocker_count": summary["commercial_blocker_count"],
            "min_average_rating": summary["min_average_rating"],
        },
        "release_candidates": [
            {
                "id": case["id"],
                "rating": case.get("rating"),
                "score": case.get("score"),
                "output_dir": case.get("output_dir"),
            }
            for case in summary["cases"]
            if case.get("rating") == "A"
        ],
    }


def _commercial_blockers(report: dict[str, Any]) -> list[str]:
    return [
        str(issue.get("code"))
        for issue in report.get("errors", [])
        if issue.get("code") in COMMERCIAL_BLOCK_CODES
    ]


def _rating_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {rating: 0 for rating in ("A", "B", "C", "D")}
    for case in cases:
        rating = case.get("rating")
        if rating in counts:
            counts[str(rating)] += 1
    return counts


def _rating_from_points(points: float) -> str:
    if points >= 3.5:
        return "A"
    if points >= 2.5:
        return "B"
    if points >= 1.5:
        return "C"
    return "D"


def _load_benchmarks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return list(DEFAULT_BENCHMARKS)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    benchmarks = payload.get("benchmarks") if isinstance(payload, dict) else None
    if not isinstance(benchmarks, list):
        return list(DEFAULT_BENCHMARKS)
    return [dict(item) for item in benchmarks if isinstance(item, dict)]


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
