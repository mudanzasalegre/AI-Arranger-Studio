from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing PyYAML") from exc

ROOT = Path(__file__).resolve().parents[2]

def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))

def score_metrics(metrics: dict, thresholds: dict) -> dict:
    validation = metrics.get("validation", {})
    tracks = metrics.get("tracks", [])
    score = 1.0
    errors = []
    warnings = []

    if validation.get("errors", 0) > thresholds["global"]["max_blocking_errors"]:
        score -= 0.4
        errors.append("blocking_validation_errors")

    if metrics.get("project", {}).get("note_events", 0) < thresholds["global"]["min_note_events"]:
        score -= 0.2
        warnings.append("low_note_count")

    roles = {track.get("role"): track for track in tracks}
    bass = roles.get("walking_bass")
    if bass:
        min_root = thresholds["bass"]["min_beat1_root_score"]
        if bass.get("beat1_root_score") is not None and bass["beat1_root_score"] < min_root:
            score -= 0.15
            warnings.append("bass_weak_roots")

    melody = roles.get("melody")
    if melody and melody.get("breath_rest_count", 0) < thresholds["melody"]["min_breath_rest_count"]:
        score -= 0.1
        warnings.append("melody_not_breathable")

    score = max(0.0, min(1.0, score))
    rating = "D"
    for candidate in ("A", "B", "C", "D"):
        if score >= thresholds["ratings"][candidate]["min_score"]:
            rating = candidate
            break
    return {"score": round(score, 3), "rating": rating, "errors": errors, "warnings": warnings}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--thresholds", default="configs/quality_thresholds.pro.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics_path = ROOT / args.metrics if not Path(args.metrics).is_absolute() else Path(args.metrics)
    thresholds_path = ROOT / args.thresholds if not Path(args.thresholds).is_absolute() else Path(args.thresholds)
    report = score_metrics(load(metrics_path), load(thresholds_path))
    output = Path(args.output) if args.output else metrics_path.with_name("quality_report.json")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["rating"] == "D" or report["errors"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
