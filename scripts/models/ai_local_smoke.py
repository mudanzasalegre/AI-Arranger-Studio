from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FINAL_ARTIFACT_STATUSES = {"validated", "rejected"}
SUMMARY_PATH = ROOT / "outputs/model_smoke/ai_local_smoke_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local model smoke suite against a running API."
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--runtime-config", default="configs/local_model_runtime.yaml")
    parser.add_argument("--api-storage", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--use-midigpt", action="store_true", help="Force MIDI-GPT checks.")
    parser.add_argument("--skip-midigpt", action="store_true")
    parser.add_argument("--use-text2midi", action="store_true", help="Force Text2MIDI checks.")
    parser.add_argument("--skip-text2midi", action="store_true")
    parser.add_argument("--skip-local-scripts", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument(
        "--allow-pending-takes-in-export",
        action="store_true",
        help="Do not fail when exported takes_manifest contains pending takes.",
    )
    args = parser.parse_args()

    load_dotenv_defaults(ROOT / ".env")

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing httpx. Run `python -m pip install -r requirements.txt`.") from exc

    base = args.api.rstrip("/")
    summary: dict[str, Any] = {
        "status": "pending",
        "api": base,
        "steps": [],
        "artifact_ids": [],
        "project_id": None,
    }

    if not args.skip_local_scripts:
        _run_local_script(
            summary,
            "models_check",
            [
                sys.executable,
                "scripts/models/check_local_model_runtime.py",
                "--config",
                args.runtime_config,
            ],
        )

    with httpx.Client(timeout=args.timeout) as client:
        health = _get(client, base + "/health")
        _add_step(summary, "health", "ok", response=health)

        models_payload = _get(client, base + "/v1/ai/models")
        models = _models_by_id(models_payload)
        _add_step(
            summary,
            "models",
            "ok",
            count=models_payload.get("count"),
            enabled_backends=sorted(
                model_id for model_id, model in models.items() if model.get("enabled")
            ),
            models=_compact_models(models_payload.get("models", [])),
        )

        planner_enabled = _should_run_backend(models, "local_llm_planner")
        midigpt_enabled = _should_run_backend(
            models,
            "midigpt",
            force=args.use_midigpt,
            skip=args.skip_midigpt,
        )
        text2midi_enabled = _should_run_backend(
            models,
            "text2midi",
            force=args.use_text2midi,
            skip=args.skip_text2midi,
        )

        if not args.skip_local_scripts:
            _run_optional_script(
                summary,
                "ollama_planner_smoke",
                [sys.executable, "scripts/models/smoke_ollama_planner.py"],
                enabled=planner_enabled,
                reason="local_llm_planner disabled",
            )
            _run_optional_script(
                summary,
                "midigpt_load_smoke",
                [sys.executable, "scripts/models/smoke_midigpt.py", "--model-dim", "4"],
                enabled=midigpt_enabled,
                reason="midigpt disabled",
            )
            _run_optional_script(
                summary,
                "text2midi_checkpoint_smoke",
                [sys.executable, "scripts/models/smoke_text2midi.py"],
                enabled=text2midi_enabled,
                reason="text2midi disabled",
            )
            _run_optional_script(
                summary,
                "miditok_tokenization_smoke",
                [sys.executable, "scripts/models/smoke_miditok.py"],
                enabled=importlib.util.find_spec("miditok") is not None,
                reason="miditok not installed",
            )

        project_id = f"local_smoke_{int(time.time())}"
        summary["project_id"] = project_id
        generated = _post(
            client,
            base + "/v1/projects/generate",
            {
                "project_id": project_id,
                "prompt": (
                    "hard bop nocturne in C minor, 132 bpm, minor blues, sextet "
                    "with alto sax, trumpet, trombone, piano, double bass and drums"
                ),
                "seed": 2601,
                "options": {"export": False, "validate": True, "include_pdf": False},
            },
        )
        _add_step(
            summary,
            "generate_project",
            "ok",
            project_id=project_id,
            response=_compact_project_response(generated),
        )

        plan = _post(
            client,
            base + f"/v1/projects/{project_id}/ai/plan",
            {"prompt": "make the shout chorus more intense without changing tracks", "seed": 2602},
        )
        _add_step(
            summary,
            "ai_plan",
            "ok",
            planner=plan.get("planner"),
            fallback_used=plan.get("fallback_used"),
            validation=(plan.get("validation") or {}).get("status"),
            response=_compact_plan_response(plan),
        )

        tracks = generated.get("project", {}).get("tracks", [])
        target_track = _target_track_id(tracks)
        if target_track is None:
            raise RuntimeError("No target track found for infill smoke")

        mock_take = _post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": "mock_symbolic",
                "track_id": target_track,
                "bars": [1],
                "instruction": "mock infill smoke",
                "seed": 2603,
            },
        )
        _collect_artifact_id(summary, mock_take)
        _add_step(
            summary,
            "mock_infill",
            "ok",
            take_id=_take_id(mock_take),
            artifact_id=_artifact_id(mock_take),
            response=_compact_take_response(mock_take),
        )

        mock_take_id = _take_id(mock_take)
        if mock_take_id is None:
            raise RuntimeError("Mock infill did not return a take id")
        accepted = _post(
            client,
            base + f"/v1/projects/{project_id}/takes/{mock_take_id}/accept",
            {},
        )
        _add_step(
            summary,
            "accept_mock_take",
            "ok",
            take_id=mock_take_id,
            response=_compact_take_response(accepted),
        )

        reject_take = _post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": "mock_symbolic",
                "track_id": target_track,
                "bars": [2],
                "instruction": "mock reject smoke",
                "seed": 2604,
            },
        )
        _collect_artifact_id(summary, reject_take)
        reject_take_id = _take_id(reject_take)
        if reject_take_id is None:
            raise RuntimeError("Reject smoke infill did not return a take id")
        rejected = _post(
            client,
            base + f"/v1/projects/{project_id}/takes/{reject_take_id}/reject",
            {},
        )
        _add_step(
            summary,
            "reject_mock_take",
            "ok",
            take_id=reject_take_id,
            artifact_id=_artifact_id(reject_take),
            response=_compact_take_response(rejected),
        )

        invalid = _post_expect_error(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": "mock_symbolic",
                "track_id": target_track,
                "bars": [3],
                "instruction": "invalid midi artifact quarantine smoke",
                "seed": 2605,
                "metadata": {"mock_artifact": "invalid_midi"},
            },
            expected_status=422,
        )
        invalid_artifact_id = _artifact_id_from_error(invalid)
        if invalid_artifact_id:
            summary["artifact_ids"].append(invalid_artifact_id)
        _add_step(
            summary,
            "artifact_quarantine_rejects_invalid_midi",
            "ok",
            artifact_id=invalid_artifact_id,
            response=invalid,
        )

        if midigpt_enabled:
            midigpt_take = _post(
                client,
                base + f"/v1/projects/{project_id}/ai/infill",
                {
                    "backend": "midigpt",
                    "track_id": target_track,
                    "bars": [4, 5],
                    "instruction": "local MIDI-GPT smoke infill, medium density",
                    "seed": 2606,
                },
            )
            _collect_artifact_id(summary, midigpt_take)
            midigpt_take_id = _take_id(midigpt_take)
            _add_step(
                summary,
                "midigpt_infill",
                "ok",
                take_id=midigpt_take_id,
                artifact_id=_artifact_id(midigpt_take),
                response=_compact_take_response(midigpt_take),
            )
            if midigpt_take_id:
                accept_midigpt = _post(
                    client,
                    base + f"/v1/projects/{project_id}/takes/{midigpt_take_id}/accept",
                    {},
                )
                _add_step(
                    summary,
                    "accept_midigpt_take",
                    "ok",
                    take_id=midigpt_take_id,
                    response=_compact_take_response(accept_midigpt),
                )
        else:
            _add_step(summary, "midigpt_infill", "skipped", reason="midigpt disabled")

        if text2midi_enabled:
            sketch = _post(
                client,
                base + "/v1/ai/text-to-midi-sketch",
                {
                    "backend": "text2midi",
                    "prompt": (
                        "Hard bop minor blues in C minor, 132 BPM, with piano, "
                        "double bass, drums and alto sax."
                    ),
                    "seed": 2607,
                },
            )
            _collect_artifact_id(summary, sketch)
            _add_step(
                summary,
                "text2midi_sketch",
                "ok",
                sketch_id=sketch.get("sketch_id"),
                artifact_id=_artifact_id(sketch),
                sketch_status=sketch.get("status"),
                response=_compact_sketch_response(sketch),
            )
        else:
            _add_step(summary, "text2midi_sketch", "skipped", reason="text2midi disabled")

        takes = _get(client, base + f"/v1/projects/{project_id}/takes")
        _assert_no_pending_takes(takes)
        _add_step(
            summary,
            "takes_accept_reject",
            "ok",
            count=takes.get("count"),
            active_take_id=takes.get("active_take_id"),
            statuses=_take_status_counts(takes),
        )

        export = _post(
            client,
            base + f"/v1/projects/{project_id}/export",
            {"include_pdf": False, "validation_policy": "strict"},
        )
        _add_step(
            summary,
            "export",
            "ok",
            file_count=len(export.get("files", [])),
            validation=(export.get("validation") or {}).get("status"),
            response=_compact_export_response(export),
        )

        if not args.skip_zip:
            zip_response = client.get(base + f"/v1/projects/{project_id}/zip")
            zip_response.raise_for_status()
            if zip_response.content[:2] != b"PK":
                raise RuntimeError("Project ZIP response is not a zip archive")
            zip_path = ROOT / f"outputs/model_smoke/{project_id}.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_response.content)
            _add_step(
                summary,
                "export_zip",
                "ok",
                zip_path=str(zip_path),
                bytes=len(zip_response.content),
            )

    api_storage = _api_storage_root(args.api_storage)
    project_dir = api_storage / "projects" / str(summary["project_id"])
    exported_takes_manifest = _read_optional_json(project_dir / "takes_manifest.json")
    if exported_takes_manifest:
        if not args.allow_pending_takes_in_export:
            _assert_export_has_no_pending_takes(exported_takes_manifest)
        _add_step(
            summary,
            "export_takes_manifest",
            "ok",
            export_policy=exported_takes_manifest.get("export_policy"),
            count=exported_takes_manifest.get("count"),
            statuses=_take_status_counts(exported_takes_manifest),
        )
    else:
        _add_step(
            summary,
            "export_takes_manifest",
            "skipped",
            reason=f"not found at {project_dir / 'takes_manifest.json'}",
        )

    artifact_root = _find_artifact_root(args.artifact_root, api_storage, summary["artifact_ids"])
    artifact_statuses = _assert_final_artifact_statuses(
        artifact_root,
        [str(item) for item in summary["artifact_ids"]],
    )
    _add_step(
        summary,
        "artifact_statuses_final",
        "ok",
        artifact_root=str(artifact_root),
        statuses=artifact_statuses,
    )

    summary["status"] = "ok"
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_dotenv_defaults(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def _get(client: Any, url: str) -> dict[str, Any]:
    response = client.get(url)
    try:
        data = response.json()
    except Exception:
        data = {"text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"GET {url} failed {response.status_code}: {data}")
    return data


def _post(client: Any, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    try:
        data = response.json()
    except Exception:
        data = {"text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"POST {url} failed {response.status_code}: {data}")
    return data


def _post_expect_error(
    client: Any,
    url: str,
    payload: dict[str, Any],
    *,
    expected_status: int,
) -> dict[str, Any]:
    response = client.post(url, json=payload)
    try:
        data = response.json()
    except Exception:
        data = {"text": response.text}
    if response.status_code != expected_status:
        raise RuntimeError(
            f"POST {url} expected {expected_status}, got {response.status_code}: {data}"
        )
    return data


def _add_step(summary: dict[str, Any], step: str, status: str, **fields: Any) -> None:
    summary["steps"].append({"step": step, "status": status, **fields})


def _run_optional_script(
    summary: dict[str, Any],
    step: str,
    command: list[str],
    *,
    enabled: bool,
    reason: str,
) -> None:
    if not enabled:
        _add_step(summary, step, "skipped", reason=reason)
        return
    _run_local_script(summary, step, command)


def _run_local_script(summary: dict[str, Any], step: str, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = "ok" if completed.returncode == 0 else "failed"
    _add_step(
        summary,
        step,
        status,
        command=command,
        returncode=completed.returncode,
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{step} failed with exit code {completed.returncode}")


def _models_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(model.get("id")): model
        for model in payload.get("models", [])
        if isinstance(model, dict) and model.get("id")
    }


def _should_run_backend(
    models: dict[str, dict[str, Any]],
    backend_id: str,
    *,
    force: bool = False,
    skip: bool = False,
) -> bool:
    if skip:
        return False
    if force:
        return True
    model = models.get(backend_id)
    return bool(model and model.get("enabled") and model.get("status") != "disabled")


def _target_track_id(tracks: list[dict[str, Any]]) -> str | None:
    return next(
        (
            str(track["id"])
            for track in tracks
            if track.get("role") in {"melody", "horn_response"}
        ),
        str(tracks[0]["id"]) if tracks else None,
    )


def _take_id(payload: dict[str, Any]) -> str | None:
    take = payload.get("take")
    return str(take.get("take_id")) if isinstance(take, dict) and take.get("take_id") else None


def _artifact_id(payload: dict[str, Any]) -> str | None:
    artifact = payload.get("artifact")
    if isinstance(artifact, dict) and artifact.get("artifact_id"):
        return str(artifact["artifact_id"])
    return None


def _artifact_id_from_error(payload: dict[str, Any]) -> str | None:
    detail = payload.get("detail")
    if isinstance(detail, dict) and detail.get("artifact_id"):
        return str(detail["artifact_id"])
    return None


def _collect_artifact_id(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    artifact_id = _artifact_id(payload)
    if artifact_id:
        summary["artifact_ids"].append(artifact_id)


def _api_storage_root(configured: str | None) -> Path:
    value = configured or os.environ.get("AI_ARRANGER_API_STORAGE")
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    return ROOT / "outputs/api"


def _find_artifact_root(
    configured: str | None,
    api_storage: Path,
    artifact_ids: list[Any],
) -> Path:
    candidates: list[Path] = []
    if configured:
        path = Path(configured).expanduser()
        candidates.append(path if path.is_absolute() else ROOT / path)
    candidates.extend(
        [
            api_storage / "model_artifacts",
            ROOT / "outputs/model_artifacts",
        ]
    )
    wanted = {str(item) for item in artifact_ids}
    for candidate in candidates:
        manifest = candidate / "artifact_manifest.json"
        if not manifest.exists():
            continue
        records = _artifact_records(candidate)
        if wanted <= {str(record.get("artifact_id")) for record in records}:
            return candidate
    return candidates[0]


def _artifact_records(artifact_root: Path) -> list[dict[str, Any]]:
    manifest = artifact_root / "artifact_manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"Artifact manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise RuntimeError(f"Artifact manifest has invalid artifacts list: {manifest}")
    return [item for item in artifacts if isinstance(item, dict)]


def _assert_final_artifact_statuses(
    artifact_root: Path,
    artifact_ids: list[str],
) -> dict[str, str]:
    records = {
        str(record.get("artifact_id")): record
        for record in _artifact_records(artifact_root)
        if record.get("artifact_id")
    }
    statuses: dict[str, str] = {}
    missing: list[str] = []
    unfinished: dict[str, str] = {}
    for artifact_id in artifact_ids:
        record = records.get(artifact_id)
        if record is None:
            missing.append(artifact_id)
            continue
        status = str(record.get("status") or "")
        statuses[artifact_id] = status
        if status not in FINAL_ARTIFACT_STATUSES:
            unfinished[artifact_id] = status
    if missing:
        raise RuntimeError(f"Artifact ids missing from manifest: {missing}")
    if unfinished:
        raise RuntimeError(f"Artifacts without final status: {unfinished}")
    return statuses


def _assert_no_pending_takes(takes_payload: dict[str, Any]) -> None:
    pending = [
        take.get("take_id")
        for take in takes_payload.get("takes", [])
        if isinstance(take, dict) and take.get("status") == "pending"
    ]
    if pending:
        raise RuntimeError(f"Pending takes remain after accept/reject smoke: {pending}")


def _assert_export_has_no_pending_takes(exported_takes_manifest: dict[str, Any]) -> None:
    pending = [
        take.get("take_id")
        for take in exported_takes_manifest.get("takes", [])
        if isinstance(take, dict) and take.get("status") == "pending"
    ]
    if pending:
        raise RuntimeError(f"Export includes pending takes: {pending}")


def _take_status_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for take in payload.get("takes", []):
        if isinstance(take, dict):
            status = str(take.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
    return counts


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_project_response(payload: dict[str, Any]) -> dict[str, Any]:
    project = payload.get("project", {})
    return {
        "status": payload.get("status"),
        "project_id": payload.get("project_id"),
        "bar_count": project.get("bar_count"),
        "track_count": len(project.get("tracks", [])) if isinstance(project, dict) else None,
        "validation": (payload.get("validation") or {}).get("status"),
    }


def _compact_models(models: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list):
        return []
    compact: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        metadata = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
        compact.append(
            {
                "id": model.get("id"),
                "backend_type": model.get("backend_type"),
                "enabled": model.get("enabled"),
                "status": model.get("status"),
                "commercial_use": model.get("commercial_use"),
                "role": metadata.get("role"),
                "error": model.get("error"),
            }
        )
    return compact


def _compact_plan_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "planner": payload.get("planner"),
        "plan_version": payload.get("plan_version"),
        "fallback_used": payload.get("fallback_used"),
        "validation": (payload.get("validation") or {}).get("status"),
        "attempts": [
            {
                "attempt": attempt.get("attempt"),
                "source": attempt.get("source"),
                "status": attempt.get("status"),
                "error": attempt.get("error"),
            }
            for attempt in payload.get("attempts", [])
            if isinstance(attempt, dict)
        ],
    }


def _compact_take_response(payload: dict[str, Any]) -> dict[str, Any]:
    take = payload.get("take", {})
    artifact = payload.get("artifact", {})
    return {
        "status": payload.get("status"),
        "backend": payload.get("backend"),
        "take_id": take.get("take_id") if isinstance(take, dict) else None,
        "take_status": take.get("status") if isinstance(take, dict) else None,
        "artifact_id": artifact.get("artifact_id") if isinstance(artifact, dict) else None,
        "artifact_status": artifact.get("status") if isinstance(artifact, dict) else None,
        "validation": (payload.get("validation") or {}).get("status"),
    }


def _compact_sketch_response(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = payload.get("artifact", {})
    sketch = payload.get("sketch", {})
    return {
        "status": payload.get("status"),
        "sketch_id": payload.get("sketch_id"),
        "artifact_id": artifact.get("artifact_id") if isinstance(artifact, dict) else None,
        "artifact_status": artifact.get("status") if isinstance(artifact, dict) else None,
        "track_count": len(sketch.get("tracks", [])) if isinstance(sketch, dict) else None,
        "validation": (payload.get("validation") or {}).get("status"),
    }


def _compact_export_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "project_id": payload.get("project_id"),
        "file_count": len(payload.get("files", [])),
        "validation": (payload.get("validation") or {}).get("status"),
    }


if __name__ == "__main__":
    main()
