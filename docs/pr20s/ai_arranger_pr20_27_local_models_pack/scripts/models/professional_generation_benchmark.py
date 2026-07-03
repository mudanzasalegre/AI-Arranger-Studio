from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PyYAML. Run requirements.txt.") from exc

ROOT = Path(__file__).resolve().parents[2]


def post(client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    try:
        data = response.json()
    except Exception:
        data = {"text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"POST {url} failed {response.status_code}: {data}")
    return data


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/professional_benchmarks.yaml")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--use-ai", action="store_true")
    parser.add_argument("--backend", default="midigpt")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing httpx. Run requirements.txt.") from exc

    config_path = ROOT / args.config
    config = load_config(config_path)
    output_root = ROOT / config.get("output_root", "outputs/professional_benchmark")
    output_root.mkdir(parents=True, exist_ok=True)

    base = args.api.rstrip("/")
    summaries = []
    with httpx.Client(timeout=args.timeout) as client:
        client.get(base + "/health").raise_for_status()
        for item in config.get("benchmarks", []):
            project_id = item["id"]
            bench_summary: dict[str, Any] = {"id": project_id, "steps": []}
            generated = post(
                client,
                base + "/v1/projects/generate",
                {
                    "project_id": project_id,
                    "prompt": item["prompt"],
                    "seed": item.get("seed", 0),
                    "options": {"export": True, "validate": True, "include_pdf": False},
                },
            )
            bench_summary["steps"].append({"step": "generate", "status": "ok"})
            bench_summary["generation"] = generated

            if args.use_ai:
                tracks = generated.get("project", {}).get("tracks", [])
                available_track_ids = {track.get("id") for track in tracks}
                for target in item.get("ai_infill_targets", []):
                    track_id = target["track_id"]
                    if track_id not in available_track_ids:
                        bench_summary["steps"].append({"step": "ai_infill", "status": "skipped", "reason": f"track missing {track_id}"})
                        continue
                    try:
                        take = post(
                            client,
                            base + f"/v1/projects/{project_id}/ai/infill",
                            {
                                "backend": args.backend,
                                "track_id": track_id,
                                "bars": target["bars"],
                                "instruction": target.get("instruction", "professional benchmark infill"),
                                "seed": item.get("seed", 0) + 100,
                            },
                        )
                        take_id = take.get("take", {}).get("take_id")
                        if take_id:
                            post(client, base + f"/v1/projects/{project_id}/takes/{take_id}/accept", {})
                        bench_summary["steps"].append({"step": "ai_infill", "status": "ok", "track_id": track_id, "take_id": take_id})
                    except Exception as exc:  # keep benchmark going; summary will mark warning
                        bench_summary["steps"].append({"step": "ai_infill", "status": "failed", "track_id": track_id, "error": str(exc)})

            exported = post(client, base + f"/v1/projects/{project_id}/export", {"include_pdf": False})
            validation = client.get(base + f"/v1/projects/{project_id}/validation").json()
            bench_summary["export"] = exported
            bench_summary["validation"] = validation
            bench_summary["status"] = "ok" if not validation.get("errors") else "fail"
            summaries.append(bench_summary)

    aggregate = {
        "status": "ok" if all(item.get("status") == "ok" for item in summaries) else "fail",
        "config": str(config_path),
        "count": len(summaries),
        "benchmarks": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(summary_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "ok":
        raise SystemExit(1)


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Professional generation benchmark", "", f"Status: **{summary['status']}**", ""]
    for item in summary["benchmarks"]:
        errors = len(item.get("validation", {}).get("errors", []))
        warnings = len(item.get("validation", {}).get("warnings", []))
        lines.append(f"- `{item['id']}`: {item.get('status')} — errors={errors}, warnings={warnings}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
