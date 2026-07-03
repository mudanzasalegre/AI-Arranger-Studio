from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "pro_benchmarks" / "ollama_planner_endpoint_smoke"


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_error = output_dir / "response_error.json"
    if stale_error.exists():
        stale_error.unlink()
    _configure_environment(output_dir)
    _ensure_import_paths()

    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    project_id = f"ollama_pr32_{uuid4().hex[:8]}"
    generated = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
            "seed": 3232,
            "project_id": project_id,
            "options": {"validate": True},
        },
    )
    if generated.status_code != 200:
        _write_json(
            output_dir / "response_error.json",
            {
                "stage": "project_generate",
                "status_code": generated.status_code,
                "body": _body(generated),
            },
        )
        raise SystemExit(1)

    response = client.post(
        f"/v1/projects/{project_id}/ai/plan",
        json={
            "prompt": (
                "Plan a hard bop minor blues head with horn responses, turnaround, "
                "and medium-high rhythm section energy. Return symbolic planning JSON only."
            ),
            "mode": "create_or_patch_plan",
            "seed": 3233,
        },
    )
    if response.status_code != 200:
        _write_json(
            output_dir / "response_error.json",
            {
                "stage": "ai_plan",
                "status_code": response.status_code,
                "body": _body(response),
            },
        )
        raise SystemExit(1)

    payload = response.json()
    report = {
        "status": "ok"
        if payload.get("planner") == "llm" and payload.get("fallback_used") is False
        else "fail",
        "project_id": project_id,
        "planner": payload.get("planner"),
        "fallback_used": payload.get("fallback_used"),
        "validation": payload.get("validation"),
        "attempts": payload.get("attempts"),
        "plan_version": payload.get("plan_version"),
        "files": payload.get("files"),
    }
    _write_json(output_dir / "endpoint_response.json", payload)
    _write_json(output_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


def _configure_environment(output_dir: Path) -> None:
    os.environ.setdefault("AI_MODELS_CONFIG", str(ROOT / "configs" / "ai_models.pro.yaml"))
    os.environ.setdefault(
        "LOCAL_MODEL_RUNTIME_CONFIG",
        str(ROOT / "configs" / "local_model_runtime.pro.yaml"),
    )
    os.environ["AI_ARRANGER_API_STORAGE"] = str(output_dir / "api_storage")


def _ensure_import_paths() -> None:
    for relative in (
        "apps/api",
        "packages/arranger_core",
        "packages/dataset_tools",
        "packages/midi_models",
        "packages/model_backends",
        "packages/training",
    ):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _body(response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text


if __name__ == "__main__":
    main()
