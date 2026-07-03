from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "pro_benchmarks" / "text2midi_sketch_import_smoke"


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
    prompt = (
        "Create a MIDI sketch in hard bop style, key C minor, tempo 132 BPM, "
        "meter 4/4, form 12-bar minor blues, using alto saxophone, piano, "
        "double bass, and drums."
    )
    response = client.post(
        "/v1/ai/text-to-midi-sketch",
        json={
            "backend": "text2midi",
            "prompt": prompt,
            "seed": 3131,
            "metadata": {
                "max_len": 512,
                "device": os.environ.get("AI_DEVICE", "auto"),
                "smoke_id": f"pr31_{uuid4().hex[:8]}",
            },
        },
    )
    if response.status_code != 200:
        _write_json(
            output_dir / "response_error.json",
            {
                "stage": "text2midi_sketch",
                "status_code": response.status_code,
                "body": _body(response),
            },
        )
        raise SystemExit(1)

    payload = response.json()
    status = payload["status"]
    if status not in {"sketch_ready", "sketch_uncertain"}:
        _write_json(output_dir / "response_error.json", payload)
        raise SystemExit(f"Unexpected sketch status: {status}")

    sketch_id = payload["sketch_id"]
    sketch_dir = Path(os.environ["AI_ARRANGER_API_STORAGE"]) / "sketches" / sketch_id
    raw_path = Path(payload["artifact"]["raw_path"])
    project_path = sketch_dir / "arrangement_project.json"

    _copy_required(raw_path, output_dir / "generated_sketch.mid")
    _copy_required(project_path, output_dir / "arrangement_project.json")
    _write_json(output_dir / "validation_report.json", payload["validation"])
    _write_json(output_dir / "sketch_metadata.json", payload)

    report = {
        "status": "ok",
        "sketch_status": status,
        "sketch_id": sketch_id,
        "output_dir": str(output_dir),
        "files": {
            "generated_midi": str(output_dir / "generated_sketch.mid"),
            "arrangement_project": str(output_dir / "arrangement_project.json"),
            "validation_report": str(output_dir / "validation_report.json"),
            "sketch_metadata": str(output_dir / "sketch_metadata.json"),
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
