from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("arranger_core", "dataset_tools", "model_backends", "midi_models"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from arranger_core import GenerationSpec, LlmPlanner, generate_arrangement  # noqa: E402
from model_backends.planner.ollama_planner_backend import OllamaPlannerBackend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/api")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument(
        "--prompt",
        default=(
            "Hard bop minor blues in C minor, 132 BPM, jazz sextet. "
            "Use style hard_bop, form minor_blues_12, ensemble jazz_sextet, "
            "meter 4/4, and instruments drum_kit, double_bass, piano, alto_sax, "
            "trumpet_bflat and trombone. Plan a clear head, horn response and "
            "turnaround; return JSON only."
        ),
    )
    args = parser.parse_args()

    provider = OllamaPlannerBackend(
        model_name=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )
    availability = {
        "available": provider.is_available(),
        "reason": provider.unavailable_reason,
    }
    project = generate_arrangement(
        GenerationSpec(
            style="hard_bop",
            form="minor_blues_12",
            ensemble="jazz_sextet",
            key="C minor",
            tempo=132,
            seed=args.seed,
        ),
        project_id="ollama_planner_smoke",
    )
    result = LlmPlanner(provider=provider).plan(
        prompt=args.prompt,
        project=project,
        locked_tracks=[],
        locked_sections=[],
        seed=args.seed,
    )
    report = {
        "status": "ok" if result.status == "ok" and result.planner == "llm" else "fail",
        "model": args.model,
        "base_url": args.base_url,
        "availability": availability,
        "planner": result.planner,
        "fallback_used": result.fallback_used,
        "validation": result.validation,
        "attempts": [attempt.model_dump(mode="json") for attempt in result.attempts],
        "song_plan_patch": result.song_plan_patch.model_dump(mode="json"),
        "song_plan_sections": [
            section.model_dump(mode="json") for section in result.song_plan.sections
        ],
    }
    report_path = ROOT / "outputs/model_smoke/ollama_planner_smoke_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
