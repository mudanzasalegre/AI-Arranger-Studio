from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "pro_benchmarks" / "midigpt_infill_smoke"


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_environment(output_dir)
    _ensure_import_paths()

    from app.main import app
    from fastapi.testclient import TestClient

    project_id = f"midigpt_pr30_{uuid4().hex[:8]}"
    client = TestClient(app)

    generated = client.post(
        "/v1/projects/generate",
        json={
            "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
            "seed": 3030,
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

    project_dir = Path(os.environ["AI_ARRANGER_API_STORAGE"]) / "projects" / project_id
    base_project_path = project_dir / "arrangement_project.json"
    shutil.copy2(base_project_path, output_dir / "base_project.json")

    response = client.post(
        f"/v1/projects/{project_id}/ai/infill",
        json={
            "backend": "midigpt",
            "track_id": "alto_sax",
            "bars": [1],
            "instruction": "bebop alto sax infill, medium high density, clear cadence",
            "density": "medium_high",
            "temperature": 1.0,
            "seed": 3031,
        },
    )
    if response.status_code != 200:
        _write_json(
            output_dir / "response_error.json",
            {
                "stage": "midigpt_infill",
                "status_code": response.status_code,
                "body": _body(response),
            },
        )
        raise SystemExit(1)

    payload = response.json()
    _copy_required(Path(payload["context_midi_path"]), output_dir / "context.mid")
    _copy_required(Path(payload["artifact"]["raw_path"]), output_dir / "generated_raw.mid")
    _copy_required(
        Path(payload["take"]["project_snapshot_path"]),
        output_dir / "candidate_project.json",
    )
    _write_json(output_dir / "validation_report.json", payload["validation"])

    takes_response = client.get(f"/v1/projects/{project_id}/takes")
    if takes_response.status_code != 200:
        _write_json(
            output_dir / "response_error.json",
            {
                "stage": "takes_manifest",
                "status_code": takes_response.status_code,
                "body": _body(takes_response),
            },
        )
        raise SystemExit(1)
    _write_json(output_dir / "take_manifest.json", takes_response.json())

    report = {
        "status": "ok",
        "project_id": project_id,
        "output_dir": str(output_dir),
        "validation_status": payload["validation"]["status"],
        "artifact_status": payload["artifact"]["status"],
        "take_id": payload["take"]["take_id"],
        "files": {
            "base_project": str(output_dir / "base_project.json"),
            "context_midi": str(output_dir / "context.mid"),
            "generated_raw_midi": str(output_dir / "generated_raw.mid"),
            "candidate_project": str(output_dir / "candidate_project.json"),
            "validation_report": str(output_dir / "validation_report.json"),
            "take_manifest": str(output_dir / "take_manifest.json"),
        },
    }
    _write_json(output_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


def _configure_environment(output_dir: Path) -> None:
    os.environ.setdefault("AI_MODELS_CONFIG", str(ROOT / "configs" / "ai_models.pro.yaml"))
    os.environ.setdefault(
        "LOCAL_MODEL_RUNTIME_CONFIG",
        str(ROOT / "configs" / "local_model_runtime.pro.yaml"),
    )
    os.environ["AI_ARRANGER_API_STORAGE"] = str(output_dir / "api_storage")
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "hf_cache" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "models" / "hf_cache" / "hub"))


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


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Required smoke file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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
