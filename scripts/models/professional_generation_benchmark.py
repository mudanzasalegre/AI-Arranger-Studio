from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PyYAML. Run requirements.txt.") from exc

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
FINAL_ARTIFACT_STATUSES = {"validated", "rejected"}
REQUIRED_PACKAGE_FILES = {
    "arrangement_project.json",
    "export_manifest.json",
    "full_arrangement.mid",
    "full_score.musicxml",
    "model_trace.json",
    "session_readme.md",
    "takes_manifest.json",
    "validation_report.html",
    "validation_report.json",
}
TRACK_ID_ALIASES = {
    "drums": ("drum_kit",),
    "drum_kit": ("drums",),
    "trumpet": ("trumpet_bflat",),
    "trumpet_bflat": ("trumpet",),
}

for path in (
    ROOT / "apps" / "api",
    ROOT / "packages" / "arranger_core",
    ROOT / "packages" / "model_backends",
    SCRIPTS_ROOT,
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from arranger_core import load_project_json  # noqa: E402
from golden_generate import compute_music_metrics  # noqa: E402


class ApiError(RuntimeError):
    def __init__(self, method: str, url: str, status_code: int, data: Any) -> None:
        self.method = method
        self.url = url
        self.status_code = status_code
        self.data = data
        super().__init__(f"{method} {url} failed {status_code}: {data}")


def post(client: Any, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    data = _response_data(response)
    if response.status_code >= 400:
        raise ApiError("POST", url, response.status_code, data)
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-object JSON: {data!r}")
    return data


def get_json(client: Any, url: str) -> dict[str, Any]:
    response = client.get(url)
    data = _response_data(response)
    if response.status_code >= 400:
        raise ApiError("GET", url, response.status_code, data)
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {url} returned non-object JSON: {data!r}")
    return data


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/professional_benchmarks.yaml")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="midigpt")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--use-ai", dest="use_ai", action="store_true", default=True)
    parser.add_argument("--no-ai", dest="use_ai", action="store_false")
    parser.add_argument("--use-planner", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--skip-lifecycle-smoke", action="store_true")
    parser.add_argument("--lifecycle-backend", default="mock_symbolic")
    parser.add_argument("--export-mode", choices=["private", "commercial"], default="private")
    args = parser.parse_args(argv)

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing httpx. Run requirements.txt.") from exc

    load_dotenv(ROOT / ".env")

    config_path = _repo_path(args.config)
    config = load_config(config_path)
    output_root = _repo_path(config.get("output_root", "outputs/professional_benchmark"))
    output_root.mkdir(parents=True, exist_ok=True)

    base = args.api.rstrip("/")
    storage_root = _api_storage_root()
    artifact_root = _artifact_root(storage_root)
    projects_root = storage_root / "projects"
    summaries: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout) as client:
        health = get_json(client, base + "/health")
        models = _model_inventory(client, base)
        lifecycle = (
            {"status": "skipped", "reason": "disabled"}
            if args.skip_lifecycle_smoke
            else _run_take_lifecycle_smoke(
                client,
                base=base,
                project_id="professional_benchmark_take_lifecycle_smoke",
                backend=args.lifecycle_backend,
                storage_root=storage_root,
            )
        )

        for item in config.get("benchmarks", []):
            summary = _run_benchmark_case(
                client,
                base=base,
                item=item,
                quality_gates=config.get("quality_gates", {}),
                output_root=output_root,
                projects_root=projects_root,
                artifact_root=artifact_root,
                backend=args.backend,
                use_ai=args.use_ai,
                use_planner=args.use_planner,
                clean=not args.no_clean,
                export_mode=args.export_mode,
                models=models,
            )
            summaries.append(summary)

    aggregate = _aggregate_summary(
        config_path=config_path,
        output_root=output_root,
        health=health,
        model_inventory=models,
        lifecycle_smoke=lifecycle,
        summaries=summaries,
        use_ai=args.use_ai,
        backend=args.backend,
    )
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(summary_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "ok":
        raise SystemExit(1)


def _run_benchmark_case(
    client: Any,
    *,
    base: str,
    item: dict[str, Any],
    quality_gates: dict[str, Any],
    output_root: Path,
    projects_root: Path,
    artifact_root: Path,
    backend: str,
    use_ai: bool,
    use_planner: bool,
    clean: bool,
    export_mode: str,
    models: dict[str, Any],
) -> dict[str, Any]:
    project_id = str(item["id"])
    case_dir = output_root / project_id
    project_dir = projects_root / project_id
    if clean:
        _clean_child_dir(case_dir, output_root)
        _clean_child_dir(project_dir, projects_root)
    else:
        case_dir.mkdir(parents=True, exist_ok=True)
        project_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "id": project_id,
        "status": "fail",
        "output_dir": str(case_dir),
        "project_dir": str(project_dir),
        "seed": item.get("seed", 0),
        "required_tracks": item.get("required_tracks", []),
        "steps": [],
        "backends_used": [],
        "fallbacks": [],
        "artifact_statuses": {},
        "blocking_errors": [],
    }

    try:
        generated = post(
            client,
            base + "/v1/projects/generate",
            {
                "project_id": project_id,
                "prompt": item["prompt"],
                "seed": item.get("seed", 0),
                "options": {
                    "export": False,
                    "validate": True,
                    "include_pdf": False,
                    "validation_policy": "strict",
                },
            },
        )
        summary["generation"] = _compact_generation(generated)
        summary["steps"].append({"step": "generate", "status": "ok"})

        if use_planner:
            summary["steps"].append(
                _run_planner_step(
                    client,
                    base=base,
                    project_id=project_id,
                    prompt=item["prompt"],
                    seed=item.get("seed", 0),
                )
            )

        artifact_ids: list[str] = []
        if use_ai:
            track_ids = _track_ids(generated)
            backend_info = _backend_info(models, backend)
            for offset, target in enumerate(item.get("ai_infill_targets", []), start=1):
                step = _run_ai_infill_step(
                    client,
                    base=base,
                    project_id=project_id,
                    target=target,
                    seed=int(item.get("seed", 0)) + 100 + offset,
                    backend=backend,
                    track_ids=track_ids,
                    backend_info=backend_info,
                )
                summary["steps"].append(step)
                if step.get("status") == "accepted":
                    summary["backends_used"].append(step.get("backend", backend))
                    artifact_id = step.get("artifact_id")
                    if artifact_id:
                        artifact_ids.append(str(artifact_id))
                elif step.get("fallback"):
                    summary["fallbacks"].append(step)
                    artifact_id = step.get("artifact_id")
                    if artifact_id:
                        artifact_ids.append(str(artifact_id))
        else:
            summary["steps"].append({"step": "ai_infill", "status": "skipped", "reason": "no_ai"})

        exported = post(
            client,
            base + f"/v1/projects/{project_id}/export",
            {
                "include_pdf": False,
                "validation_policy": "strict",
                "export_mode": export_mode,
            },
        )
        summary["steps"].append({"step": "export", "status": "ok"})
        summary["export"] = _compact_export(exported)

        validation = get_json(client, base + f"/v1/projects/{project_id}/validation")
        summary["validation"] = _compact_validation(validation)

        package_path = case_dir / "package.zip"
        _download_zip(client, base + f"/v1/projects/{project_id}/zip", package_path)
        manifest = _copy_export_files(project_dir, case_dir, exported.get("manifest", {}))
        package_names = _zip_names(package_path)

        model_trace = _read_json(case_dir / "model_trace.json")
        takes_manifest = _read_json(case_dir / "takes_manifest.json")
        summary["package"] = {
            "path": str(package_path),
            "size_bytes": package_path.stat().st_size,
            "required_files_present": sorted(REQUIRED_PACKAGE_FILES & package_names),
            "missing_required_files": sorted(REQUIRED_PACKAGE_FILES - package_names),
        }
        summary["takes"] = {
            "statuses": dict(Counter(_take_statuses(takes_manifest))),
            "count": takes_manifest.get("count"),
            "active_take_id": takes_manifest.get("active_take_id"),
        }
        summary["model_trace"] = _compact_model_trace(model_trace)

        if artifact_ids:
            summary["artifact_statuses"] = _assert_final_artifact_statuses(
                artifact_root,
                artifact_ids,
            )

        metrics = _write_music_metrics(
            case_dir=case_dir,
            validation=validation,
            manifest=manifest,
            benchmark=item,
        )
        summary["music_metrics"] = _compact_music_metrics(metrics)

        gate = _case_quality_gates(
            case_dir=case_dir,
            generated=generated,
            validation=validation,
            quality_gates=quality_gates,
            required_tracks=item.get("required_tracks", []),
            model_trace=model_trace,
            takes_manifest=takes_manifest,
            package_names=package_names,
            accepted_artifact_ids=[
                artifact_id
                for artifact_id in artifact_ids
                if summary["artifact_statuses"].get(artifact_id) == "validated"
            ],
        )
        summary.update(gate)
        summary["status"] = "ok" if not gate["blocking_errors"] else "fail"
    except Exception as exc:
        summary["blocking_errors"].append(str(exc))
        summary["status"] = "fail"

    (case_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_planner_step(
    client: Any,
    *,
    base: str,
    project_id: str,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    try:
        payload = post(
            client,
            base + f"/v1/projects/{project_id}/ai/plan",
            {"prompt": prompt, "seed": seed},
        )
        return {
            "step": "ai_plan",
            "status": "ok",
            "planner": payload.get("planner"),
            "plan_version": payload.get("plan_version"),
            "fallback_used": payload.get("fallback_used"),
            "validation": payload.get("validation", {}).get("status"),
        }
    except Exception as exc:
        return {
            "step": "ai_plan",
            "status": "fallback",
            "fallback": "rule_based_song_plan",
            "error": str(exc),
        }


def _run_ai_infill_step(
    client: Any,
    *,
    base: str,
    project_id: str,
    target: dict[str, Any],
    seed: int,
    backend: str,
    track_ids: set[str],
    backend_info: dict[str, Any],
) -> dict[str, Any]:
    requested_track_id = str(target["track_id"])
    track_id = _resolve_track_id(requested_track_id, track_ids)
    bars = [int(bar) for bar in target["bars"]]
    step: dict[str, Any] = {
        "step": "ai_infill",
        "track_id": track_id,
        "requested_track_id": requested_track_id,
        "bars": bars,
        "requested_backend": backend,
        "backend_status": backend_info,
    }
    if track_id not in track_ids:
        return {
            **step,
            "status": "skipped",
            "fallback": "rule_based_existing_arrangement",
            "reason": f"track missing: {requested_track_id}",
        }

    try:
        take_payload = post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": backend,
                "track_id": track_id,
                "bars": bars,
                "instruction": target.get("instruction", "professional benchmark infill"),
                "density": target.get("density", "medium"),
                "temperature": target.get("temperature", 0.85),
                "seed": seed,
            },
        )
        take_id = take_payload.get("take", {}).get("take_id")
        artifact_id = take_payload.get("artifact", {}).get("artifact_id")
        if take_id:
            post(client, base + f"/v1/projects/{project_id}/takes/{take_id}/accept", {})
        return {
            **step,
            "status": "accepted",
            "backend": take_payload.get("backend", backend),
            "take_id": take_id,
            "artifact_id": artifact_id,
            "validation": take_payload.get("validation", {}).get("status"),
        }
    except ApiError as exc:
        return {
            **step,
            "status": "fallback",
            "fallback": "rule_based_existing_arrangement",
            "error": str(exc),
            "artifact_id": _artifact_id_from_error(exc.data),
        }
    except Exception as exc:
        return {
            **step,
            "status": "fallback",
            "fallback": "rule_based_existing_arrangement",
            "error": str(exc),
        }


def _run_take_lifecycle_smoke(
    client: Any,
    *,
    base: str,
    project_id: str,
    backend: str,
    storage_root: Path,
) -> dict[str, Any]:
    projects_root = storage_root / "projects"
    _clean_child_dir(projects_root / project_id, projects_root)
    summary: dict[str, Any] = {
        "project_id": project_id,
        "backend": backend,
        "status": "fail",
        "accepted_take_id": None,
        "rejected_take_id": None,
        "artifact_ids": [],
    }
    try:
        generated = post(
            client,
            base + "/v1/projects/generate",
            {
                "project_id": project_id,
                "prompt": "hard bop minor blues trio, piano bass drums, 96 bpm",
                "seed": 2799,
                "options": {"export": False, "validate": True},
            },
        )
        tracks = _track_ids(generated)
        target_track = "piano" if "piano" in tracks else sorted(tracks)[0]
        accepted = post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": backend,
                "track_id": target_track,
                "bars": [1],
                "instruction": "short lifecycle accept smoke",
                "seed": 27991,
            },
        )
        accepted_take_id = accepted.get("take", {}).get("take_id")
        accepted_artifact_id = accepted.get("artifact", {}).get("artifact_id")
        if accepted_take_id:
            post(client, base + f"/v1/projects/{project_id}/takes/{accepted_take_id}/accept", {})

        rejected = post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": backend,
                "track_id": target_track,
                "bars": [2],
                "instruction": "short lifecycle reject smoke",
                "seed": 27992,
            },
        )
        rejected_take_id = rejected.get("take", {}).get("take_id")
        rejected_artifact_id = rejected.get("artifact", {}).get("artifact_id")
        if rejected_take_id:
            post(client, base + f"/v1/projects/{project_id}/takes/{rejected_take_id}/reject", {})

        takes = get_json(client, base + f"/v1/projects/{project_id}/takes")
        _assert_no_pending_takes(takes)
        summary.update(
            {
                "status": "ok",
                "accepted_take_id": accepted_take_id,
                "rejected_take_id": rejected_take_id,
                "takes": dict(Counter(_take_statuses(takes))),
                "artifact_ids": [
                    artifact_id
                    for artifact_id in (accepted_artifact_id, rejected_artifact_id)
                    if artifact_id
                ],
            }
        )
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _copy_export_files(
    project_dir: Path,
    case_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not manifest:
        manifest = _read_json(project_dir / "export_manifest.json")
    case_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for record in manifest.get("files", []):
        if record.get("status", "created") == "skipped":
            continue
        raw_path = record.get("path")
        if not raw_path:
            continue
        source = _resolve_export_path(project_dir, raw_path)
        if not source.exists():
            continue
        relative = source.relative_to(project_dir)
        destination = case_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    manifest_path = project_dir / "export_manifest.json"
    if manifest_path.exists() and "export_manifest.json" not in copied:
        shutil.copy2(manifest_path, case_dir / "export_manifest.json")
        copied.append("export_manifest.json")
    manifest_copy = dict(manifest)
    manifest_copy["copied_files"] = sorted(copied)
    return manifest_copy


def _case_quality_gates(
    *,
    case_dir: Path,
    generated: dict[str, Any],
    validation: dict[str, Any],
    quality_gates: dict[str, Any],
    required_tracks: list[str],
    model_trace: dict[str, Any],
    takes_manifest: dict[str, Any],
    package_names: set[str],
    accepted_artifact_ids: list[str],
) -> dict[str, Any]:
    blocking: list[str] = []
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    max_errors = int(quality_gates.get("validation_errors_max", 0))
    max_warnings = int(quality_gates.get("validation_warnings_max", 999999))
    if len(errors) > max_errors:
        blocking.append(f"validation errors {len(errors)} > {max_errors}")
    if len(warnings) > max_warnings:
        blocking.append(f"validation warnings {len(warnings)} > {max_warnings}")

    generated_track_ids = _track_ids(generated)
    present_tracks, missing_tracks = _required_track_status(required_tracks, generated_track_ids)
    if missing_tracks:
        blocking.append(f"missing required tracks: {missing_tracks}")
    min_tracks = int(quality_gates.get("min_tracks", 0))
    if len(generated_track_ids) < min_tracks:
        blocking.append(f"track count {len(generated_track_ids)} < {min_tracks}")

    midi_path = case_dir / "full_arrangement.mid"
    musicxml_path = case_dir / "full_score.musicxml"
    if quality_gates.get("require_full_midi", True) and (
        not midi_path.exists() or midi_path.stat().st_size <= 0
    ):
        blocking.append("full_arrangement.mid missing or empty")
    if quality_gates.get("require_musicxml", True) and (
        not musicxml_path.exists() or musicxml_path.stat().st_size <= 0
    ):
        blocking.append("full_score.musicxml missing or empty")

    if quality_gates.get("require_model_trace_if_ai_used", True) and accepted_artifact_ids:
        if not model_trace:
            blocking.append("model_trace.json missing after accepted AI artifacts")
        elif not model_trace.get("model_artifacts"):
            blocking.append("model_trace.json has no model_artifacts for accepted AI")

    if quality_gates.get("require_no_pending_takes_in_export", True):
        try:
            _assert_export_has_no_pending_takes(takes_manifest)
        except RuntimeError as exc:
            blocking.append(str(exc))

    missing_package_files = sorted(REQUIRED_PACKAGE_FILES - package_names)
    if missing_package_files:
        blocking.append(f"package.zip missing required files: {missing_package_files}")

    return {
        "blocking_errors": blocking,
        "required_tracks_present": present_tracks,
        "missing_required_tracks": missing_tracks,
        "validation_error_count": len(errors),
        "validation_warning_count": len(warnings),
    }


def _write_music_metrics(
    *,
    case_dir: Path,
    validation: dict[str, Any],
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    project = load_project_json(case_dir / "arrangement_project.json")
    metrics = compute_music_metrics(
        project,
        validation_report=validation,
        export_manifest=manifest,
        preset_metadata={
            "benchmark_id": benchmark["id"],
            "prompt": benchmark["prompt"],
            "seed": benchmark.get("seed", 0),
        },
    )
    (case_dir / "music_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _aggregate_summary(
    *,
    config_path: Path,
    output_root: Path,
    health: dict[str, Any],
    model_inventory: dict[str, Any],
    lifecycle_smoke: dict[str, Any],
    summaries: list[dict[str, Any]],
    use_ai: bool,
    backend: str,
) -> dict[str, Any]:
    benchmark_status = "ok" if all(item.get("status") == "ok" for item in summaries) else "fail"
    lifecycle_status = lifecycle_smoke.get("status")
    status = "ok" if benchmark_status == "ok" and lifecycle_status in {"ok", "skipped"} else "fail"
    backends = sorted(
        {
            str(backend_id)
            for item in summaries
            for backend_id in item.get("backends_used", [])
            if backend_id
        }
    )
    fallback_count = sum(len(item.get("fallbacks", [])) for item in summaries)
    return {
        "schema_version": "0.1.0",
        "status": status,
        "config": str(config_path),
        "output_root": str(output_root),
        "health": health,
        "use_ai": use_ai,
        "requested_backend": backend,
        "backends_used": backends,
        "fallback_count": fallback_count,
        "model_inventory": _compact_model_inventory(model_inventory),
        "take_lifecycle_smoke": lifecycle_smoke,
        "count": len(summaries),
        "benchmarks": summaries,
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Professional generation benchmark",
        "",
        f"Status: **{summary['status']}**",
        f"Benchmarks: `{summary['count']}`",
        f"Requested backend: `{summary.get('requested_backend')}`",
        f"Backends used: `{', '.join(summary.get('backends_used', [])) or 'none'}`",
        f"Fallbacks: `{summary.get('fallback_count', 0)}`",
        f"Take lifecycle smoke: `{summary.get('take_lifecycle_smoke', {}).get('status')}`",
        "",
        "| Benchmark | Status | Errors | Warnings | Tracks | AI backend | Fallbacks | Package |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for item in summary["benchmarks"]:
        metrics = item.get("music_metrics", {}).get("project", {})
        package = item.get("package", {})
        lines.append(
            "| "
            f"`{item['id']}` | "
            f"{item.get('status')} | "
            f"{item.get('validation_error_count', 0)} | "
            f"{item.get('validation_warning_count', 0)} | "
            f"{metrics.get('tracks', '-')} | "
            f"`{', '.join(item.get('backends_used', [])) or 'rule_based'}` | "
            f"{len(item.get('fallbacks', []))} | "
            f"{package.get('path', '-')} |"
        )
    return "\n".join(lines) + "\n"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _response_data(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return {"text": response.text}


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _api_storage_root() -> Path:
    configured = os.environ.get("AI_ARRANGER_API_STORAGE")
    return _repo_path(configured or "outputs/api")


def _artifact_root(storage_root: Path) -> Path:
    if os.environ.get("AI_ARRANGER_API_STORAGE"):
        return storage_root / "model_artifacts"
    return _repo_path("outputs/model_artifacts")


def _clean_child_dir(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise RuntimeError(f"Refusing to clean path outside parent: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def _model_inventory(client: Any, base: str) -> dict[str, Any]:
    try:
        return get_json(
            client,
            base + "/v1/ai/models?include_disabled=true&include_unavailable=true",
        )
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "models": []}


def _compact_model_inventory(models: dict[str, Any]) -> dict[str, Any]:
    compact_models: list[dict[str, Any]] = []
    for model in models.get("models", []):
        if not isinstance(model, dict):
            continue
        metadata = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
        compact_models.append(
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
    return {
        "status": models.get("status"),
        "count": models.get("count"),
        "models": compact_models,
    }


def _backend_info(models: dict[str, Any], backend: str) -> dict[str, Any]:
    for model in models.get("models", []):
        if isinstance(model, dict) and model.get("id") == backend:
            return {
                "id": model.get("id"),
                "enabled": model.get("enabled"),
                "status": model.get("status"),
                "commercial_use": model.get("commercial_use"),
                "error": model.get("error"),
            }
    return {"id": backend, "status": "unknown"}


def _download_zip(client: Any, url: str, package_path: Path) -> None:
    response = client.get(url)
    if response.status_code >= 400:
        raise ApiError("GET", url, response.status_code, _response_data(response))
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_bytes(response.content)
    with zipfile.ZipFile(package_path) as archive:
        archive.testzip()


def _zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _resolve_export_path(project_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _track_ids(payload: dict[str, Any]) -> set[str]:
    project = payload.get("project", {})
    tracks = project.get("tracks", []) if isinstance(project, dict) else []
    return {str(track.get("id")) for track in tracks if isinstance(track, dict) and track.get("id")}


def _resolve_track_id(requested_track_id: str, available_track_ids: set[str]) -> str:
    if requested_track_id in available_track_ids:
        return requested_track_id
    for alias in TRACK_ID_ALIASES.get(requested_track_id, ()):
        if alias in available_track_ids:
            return alias
    return requested_track_id


def _required_track_status(
    required_tracks: list[str],
    available_track_ids: set[str],
) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for required_track in required_tracks:
        if _resolve_track_id(str(required_track), available_track_ids) in available_track_ids:
            present.append(str(required_track))
        else:
            missing.append(str(required_track))
    return sorted(present), sorted(missing)


def _take_statuses(payload: dict[str, Any]) -> list[str]:
    return [
        str(take.get("status") or "unknown")
        for take in payload.get("takes", [])
        if isinstance(take, dict)
    ]


def _assert_no_pending_takes(takes_payload: dict[str, Any]) -> None:
    pending = [
        take.get("take_id")
        for take in takes_payload.get("takes", [])
        if isinstance(take, dict) and take.get("status") == "pending"
    ]
    if pending:
        raise RuntimeError(f"Pending takes remain after lifecycle smoke: {pending}")


def _assert_export_has_no_pending_takes(exported_takes_manifest: dict[str, Any]) -> None:
    pending = [
        take.get("take_id")
        for take in exported_takes_manifest.get("takes", [])
        if isinstance(take, dict) and take.get("status") == "pending"
    ]
    if pending:
        raise RuntimeError(f"Export includes pending takes: {pending}")


def _artifact_records(artifact_root: Path) -> list[dict[str, Any]]:
    manifest = artifact_root / "artifact_manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
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


def _artifact_id_from_error(data: Any) -> str | None:
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, dict):
        artifact_id = detail.get("artifact_id")
        return str(artifact_id) if artifact_id else None
    return None


def _compact_generation(payload: dict[str, Any]) -> dict[str, Any]:
    project = payload.get("project", {})
    tracks = project.get("tracks", []) if isinstance(project, dict) else []
    return {
        "status": payload.get("status"),
        "project_id": payload.get("project_id"),
        "bars": project.get("bar_count") if isinstance(project, dict) else None,
        "tracks": len(tracks),
        "track_ids": sorted(_track_ids(payload)),
        "validation": payload.get("validation", {}).get("status"),
    }


def _compact_export(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest", {})
    return {
        "status": payload.get("status"),
        "files": len(payload.get("files", [])),
        "manifest_status": manifest.get("status") if isinstance(manifest, dict) else None,
        "pdf_status": manifest.get("pdf_status") if isinstance(manifest, dict) else None,
        "validation": payload.get("validation", {}).get("status"),
    }


def _compact_validation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "errors": len(payload.get("errors", [])),
        "warnings": len(payload.get("warnings", [])),
        "metrics": payload.get("metrics", {}),
    }


def _compact_model_trace(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload.get("model_artifacts", [])
    return {
        "status": payload.get("status"),
        "artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "backends": sorted(
            {
                str(item.get("backend_id"))
                for item in artifacts
                if isinstance(item, dict) and item.get("backend_id")
            }
        )
        if isinstance(artifacts, list)
        else [],
    }


def _compact_music_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    project = metrics.get("project", {})
    validation = metrics.get("validation", {})
    return {
        "project": {
            "style": project.get("style"),
            "form": project.get("form"),
            "bars": project.get("bars"),
            "tracks": project.get("tracks"),
            "note_events": project.get("note_events"),
            "notes_per_bar": project.get("notes_per_bar"),
        },
        "validation": validation,
        "quality_flags": len(metrics.get("quality_flags", [])),
        "estimated_score_1_to_5": metrics.get("baseline_rating", {}).get(
            "estimated_score_1_to_5"
        ),
        "tracks": [
            {
                "track_id": track.get("track_id"),
                "role": track.get("role"),
                "instrument": track.get("instrument"),
                "notes_per_bar": track.get("notes_per_bar"),
                "large_leaps": track.get("large_leaps"),
                "beat1_root_score": track.get("beat1_root_score"),
                "approach_to_next_root_score": track.get("approach_to_next_root_score"),
                "rootless_violations": track.get("rootless_violations"),
                "breath_rest_count": track.get("breath_rest_count"),
                "fill_bar_count": track.get("fill_bar_count"),
            }
            for track in metrics.get("tracks", [])
        ],
    }


if __name__ == "__main__":
    main()
