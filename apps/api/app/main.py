from __future__ import annotations

import json
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from arranger_core import (
    ArrangementProject,
    GenerationSpec,
    MusicValidationError,
    TakeManager,
    export_project,
    generate_arrangement,
    validate_project,
    write_full_midi,
    write_validation_html,
    write_validation_json,
)
from arranger_core import (
    compile_prompt as compile_generation_spec,
)
from dataset_tools import PatternIndex, create_manifest, import_dataset
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from model_backends import (
    ModelBackendConfigurationError,
    build_model_backend_registry,
    load_ai_models_config,
)
from pydantic import BaseModel, ConfigDict, Field

from app.routes.ai_generation import router as ai_generation_router
from app.routes.ai_planner import router as ai_planner_router

app = FastAPI(title="AI Arranger Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ai_planner_router)
app.include_router(ai_generation_router)

API_STORAGE_ENV = "AI_ARRANGER_API_STORAGE"
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
FILE_KIND_ALIASES = {
    "export_manifest": "export_manifest",
    "generation_spec": "generation_spec_json",
    "html": "validation_report_html",
    "manifest": "export_manifest",
    "midi": "midi_full",
    "model_trace": "model_trace_json",
    "musicxml": "musicxml_full",
    "project": "project_json",
    "readme": "session_readme",
    "score": "musicxml_full",
    "takes_manifest": "takes_manifest_json",
    "validation": "validation_report_json",
}


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PromptCompileRequest(ApiModel):
    prompt: str = Field(default="")
    seed: int = Field(default=0)


class GenerateOptions(ApiModel):
    export: bool = False
    run_validation: bool = Field(default=True, alias="validate")
    include_pdf: bool = False
    validation_policy: Literal["strict", "report_only"] = "strict"
    export_mode: Literal["private", "commercial"] = "private"


class ProjectGenerateRequest(ApiModel):
    prompt: str = Field(default="")
    seed: int | None = None
    spec: dict[str, Any] | None = None
    project_id: str | None = None
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class ProjectExportRequest(ApiModel):
    include_pdf: bool = False
    validation_policy: Literal["strict", "report_only"] = "strict"
    export_mode: Literal["private", "commercial"] = "private"


class RegenerateRequest(ApiModel):
    target: dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""
    seed: int | None = None
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class DatasetImportRequest(ApiModel):
    source_dir: str
    dataset_id: str | None = None
    manifest_path: str | None = None
    default_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_by_name: dict[str, dict[str, Any]] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "service": "ai-arranger-api",
        "status": "ok",
    }


@app.get("/v1/ai/models")
def list_ai_models(
    include_disabled: bool = Query(default=True),
    include_unavailable: bool = Query(default=True),
) -> dict[str, Any]:
    try:
        config = load_ai_models_config()
        registry = build_model_backend_registry(
            config=config,
            include_disabled=include_disabled,
            include_unavailable=include_unavailable,
        )
    except ModelBackendConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    models = registry.list(include_disabled=include_disabled)
    return {
        "status": "ok",
        "count": len(models),
        "models": models,
        "settings": config.settings,
    }


@app.post("/v1/prompts/compile")
def compile_prompt(payload: PromptCompileRequest) -> dict[str, Any]:
    spec = compile_generation_spec(payload.prompt, seed=payload.seed)
    return spec.model_dump(mode="json")


@app.post("/v1/projects/generate")
def generate_project(payload: ProjectGenerateRequest) -> dict[str, Any]:
    project_id = _safe_id(payload.project_id, prefix="project")
    project_dir = _project_dir(project_id)
    spec = _spec_from_generate_request(payload)

    project = generate_arrangement(spec, project_id=project_id)
    _save_project(project, project_dir)

    validation_report: dict[str, Any] = {}
    export_manifest: dict[str, Any] = {}
    if payload.options.export:
        export_manifest = _export_project(
            project,
            project_dir,
            include_pdf=payload.options.include_pdf,
            validation_policy=payload.options.validation_policy,
            export_mode=payload.options.export_mode,
        )
        project = _load_project(project_id)
        validation_report = project.validation_report
    elif payload.options.run_validation:
        validation_report = _validate_and_store(project, project_dir)

    return {
        "project_id": project_id,
        "status": "generated",
        "project": _project_metadata(project),
        "files": export_manifest.get("files", []),
        "validation": validation_report,
    }


@app.get("/v1/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = _load_project(project_id)
    project_dir = _project_dir(project_id)
    return {
        "project_id": project.project_id,
        "status": "available",
        "project": _project_metadata(project),
        "generation_spec": (
            project.generation_spec.model_dump(mode="json")
            if project.generation_spec
            else None
        ),
        "export_manifest": _read_json_file(project_dir / "export_manifest.json"),
        "validation": _read_json_file(project_dir / "validation_report.json"),
    }


@app.get("/v1/projects/{project_id}/file")
def get_project_file(
    project_id: str,
    kind: str = Query(default="musicxml"),
    track_id: str | None = None,
) -> FileResponse:
    path = _project_file_path(project_id, kind=kind, track_id=track_id)
    return FileResponse(
        path,
        filename=path.name,
        media_type=_media_type_for_path(path),
    )


@app.get("/v1/projects/{project_id}/zip")
def get_project_zip(project_id: str) -> StreamingResponse:
    project = _load_project(project_id)
    project_dir = _project_dir(project_id)
    manifest = _read_json_file(project_dir / "export_manifest.json")
    if not manifest:
        manifest = _export_project(
            project,
            project_dir,
            include_pdf=False,
            validation_policy="strict",
        )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _export_package_paths(project_dir, manifest):
            archive.write(path, path.relative_to(project_dir).as_posix())
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{project_id}.zip"',
        },
    )


@app.post("/v1/projects/{project_id}/export")
def export_existing_project(
    project_id: str,
    payload: ProjectExportRequest | None = None,
) -> dict[str, Any]:
    request = payload or ProjectExportRequest()
    project = _load_project(project_id)
    manifest = _export_project(
        project,
        _project_dir(project_id),
        include_pdf=request.include_pdf,
        validation_policy=request.validation_policy,
        export_mode=request.export_mode,
    )
    exported_project = _load_project(project_id)
    return {
        "project_id": project_id,
        "status": "exported",
        "files": manifest.get("files", []),
        "manifest": manifest,
        "validation": exported_project.validation_report,
    }


@app.get("/v1/projects/{project_id}/validation")
def get_project_validation(project_id: str) -> dict[str, Any]:
    project = _load_project(project_id)
    project_dir = _project_dir(project_id)
    validation_path = project_dir / "validation_report.json"
    if validation_path.exists():
        return _read_json_file(validation_path)
    return _validate_and_store(project, project_dir)


@app.get("/v1/projects/{project_id}/takes")
def list_project_takes(project_id: str) -> dict[str, Any]:
    project = _load_project(project_id)
    manager = TakeManager(_project_dir(project_id))
    return manager.list_takes(project=project)


@app.get("/v1/projects/{project_id}/takes/{take_id}/diff")
def diff_project_take(project_id: str, take_id: str) -> dict[str, Any]:
    take, snapshot_path, takes = _take_snapshot(project_id, take_id)
    active_project = _load_project(project_id)
    candidate_project = ArrangementProject.load_json(snapshot_path)
    validation_path = take.get("metadata", {}).get("validation_report_path")
    validation_report = (
        _read_json_file(Path(validation_path))
        if validation_path and Path(validation_path).exists()
        else candidate_project.validation_report
    )
    diff = _project_diff(active_project, candidate_project)
    return {
        "project_id": project_id,
        "take_id": take_id,
        "status": "diff_ready",
        "active_take_id": takes.get("active_take_id"),
        "take": take,
        "summary": diff["summary"],
        "tracks": diff["tracks"],
        "changed_bars": diff["changed_bars"],
        "validation": validation_report,
    }


@app.get("/v1/projects/{project_id}/takes/{take_id}/file")
def get_project_take_file(
    project_id: str,
    take_id: str,
    kind: str = Query(default="midi"),
) -> FileResponse:
    take, snapshot_path, _takes = _take_snapshot(project_id, take_id)
    normalized_kind = FILE_KIND_ALIASES.get(kind, kind)
    if normalized_kind == "midi_full":
        candidate_project = ArrangementProject.load_json(snapshot_path)
        preview_dir = snapshot_path.parent / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / "full_arrangement.mid"
        write_full_midi(candidate_project, preview_path)
        return FileResponse(
            preview_path,
            filename=f"{take_id}.mid",
            media_type=_media_type_for_path(preview_path),
        )
    if normalized_kind == "project_json":
        return FileResponse(
            snapshot_path,
            filename=f"{take_id}.arrangement_project.json",
            media_type=_media_type_for_path(snapshot_path),
        )
    if normalized_kind == "validation_report_json":
        validation_path = take.get("metadata", {}).get("validation_report_path")
        if validation_path:
            path = _existing_project_path(_project_dir(project_id), Path(validation_path))
            return FileResponse(
                path,
                filename=f"{take_id}.validation_report.json",
                media_type=_media_type_for_path(path),
            )
    raise HTTPException(status_code=404, detail=f"Take file not found for kind={kind!r}")


@app.post("/v1/projects/{project_id}/takes/{take_id}/accept")
def accept_project_take(project_id: str, take_id: str) -> dict[str, Any]:
    _load_project(project_id)
    manager = TakeManager(_project_dir(project_id))
    try:
        take, project = manager.accept_take(take_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    validation_report = _validate_and_store(project, _project_dir(project_id))
    return {
        "project_id": project_id,
        "status": "accepted",
        "take": take.model_dump(mode="json"),
        "project": _project_metadata(project),
        "validation": validation_report,
    }


@app.post("/v1/projects/{project_id}/takes/{take_id}/reject")
def reject_project_take(project_id: str, take_id: str) -> dict[str, Any]:
    _load_project(project_id)
    manager = TakeManager(_project_dir(project_id))
    try:
        take = manager.reject_take(take_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "project_id": project_id,
        "status": "rejected",
        "take": take.model_dump(mode="json"),
    }


@app.post("/v1/projects/{project_id}/regenerate")
def regenerate_project(project_id: str, payload: RegenerateRequest) -> dict[str, Any]:
    current = _load_project(project_id)
    if current.generation_spec is None:
        raise HTTPException(status_code=409, detail="Project has no generation_spec")

    seed = payload.seed if payload.seed is not None else current.generation_spec.seed + 1
    constraints = {
        **current.generation_spec.constraints,
        "regenerate_target": payload.target,
        "regenerate_instruction": payload.instruction,
    }
    spec = current.generation_spec.model_copy(
        update={"seed": seed, "constraints": constraints}
    )
    project = generate_arrangement(spec, project_id=project_id)
    project.metadata["regenerated_from"] = current.project_id
    project.metadata["regenerate_target"] = payload.target
    project.metadata["regenerate_instruction"] = payload.instruction

    project_dir = _project_dir(project_id)
    _save_project(project, project_dir)
    validation_report: dict[str, Any] = {}
    export_manifest: dict[str, Any] = {}
    if payload.options.export:
        export_manifest = _export_project(
            project,
            project_dir,
            include_pdf=payload.options.include_pdf,
            validation_policy=payload.options.validation_policy,
            export_mode=payload.options.export_mode,
        )
        project = _load_project(project_id)
        validation_report = project.validation_report
    elif payload.options.run_validation:
        validation_report = _validate_and_store(project, project_dir)

    return {
        "project_id": project_id,
        "status": "regenerated",
        "project": _project_metadata(project),
        "files": export_manifest.get("files", []),
        "validation": validation_report,
    }


@app.post("/v1/datasets/import")
def import_dataset_endpoint(payload: DatasetImportRequest) -> dict[str, Any]:
    dataset_id = _safe_id(payload.dataset_id, prefix="dataset")
    dataset_dir = _dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(payload.source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset source not found: {source_dir}")

    manifest_path = (
        Path(payload.manifest_path).expanduser().resolve()
        if payload.manifest_path
        else dataset_dir / "dataset_manifest.json"
    )
    if payload.manifest_path is None:
        create_manifest(
            source_dir,
            manifest_path,
            default_metadata=payload.default_metadata,
            metadata_by_name=payload.metadata_by_name,
        )
    elif not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset manifest not found: {manifest_path}")

    summary = import_dataset(source_dir, manifest_path, dataset_dir)
    metadata = {
        "dataset_id": dataset_id,
        "source_dir": str(source_dir),
        "manifest_path": str(manifest_path),
        "summary": summary.model_dump(mode="json"),
    }
    _write_json(dataset_dir / "dataset_api_metadata.json", metadata)
    return {
        "dataset_id": dataset_id,
        "status": "imported",
        "summary": summary.model_dump(mode="json"),
    }


@app.get("/v1/datasets")
def list_datasets() -> dict[str, Any]:
    datasets_root = _storage_root() / "datasets"
    datasets: list[dict[str, Any]] = []
    if datasets_root.exists():
        for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
            summary_path = dataset_dir / "import_summary.json"
            if not summary_path.exists():
                continue
            datasets.append(
                {
                    "dataset_id": dataset_dir.name,
                    "summary": _read_json_file(summary_path),
                    "pattern_index_path": str(dataset_dir / "pattern_index.json"),
                }
            )
    return {"count": len(datasets), "datasets": datasets}


@app.get("/v1/patterns/search")
def search_patterns(
    category: str | None = None,
    role: str | None = None,
    style: str | None = None,
    min_quality: int = Query(default=1, ge=1, le=5),
    tags: str | None = None,
    usable_for_training: bool | None = None,
    usable_for_pattern_extraction: bool | None = None,
    dataset_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    requested_tags = _split_tags(tags)
    matches = []
    for index_path in _pattern_index_paths(dataset_id):
        index = PatternIndex.load_json(index_path)
        for pattern in index.search(
            category=category,
            role=role,
            style=style,
            min_quality=min_quality,
            tags=requested_tags,
            usable_for_training=usable_for_training,
            usable_for_pattern_extraction=usable_for_pattern_extraction,
        ):
            matches.append(pattern.model_dump(mode="json"))

    matches.sort(
        key=lambda item: (
            -int(item.get("quality", 0)),
            -float(item.get("weight", 0.0)),
            str(item.get("id", "")),
        )
    )
    limited = matches[:limit]
    return {
        "count": len(limited),
        "total": len(matches),
        "truncated": len(matches) > limit,
        "patterns": limited,
    }


def _spec_from_generate_request(payload: ProjectGenerateRequest) -> GenerationSpec:
    if payload.spec is not None:
        spec = GenerationSpec.model_validate(payload.spec)
        if payload.seed is not None:
            spec = spec.model_copy(update={"seed": payload.seed})
        return spec
    return compile_generation_spec(payload.prompt, seed=payload.seed or 0)


def _export_project(
    project: ArrangementProject,
    output_dir: Path,
    *,
    include_pdf: bool,
    validation_policy: Literal["strict", "report_only"],
    export_mode: Literal["private", "commercial"] = "private",
) -> dict[str, Any]:
    try:
        return export_project(
            project,
            output_dir,
            include_pdf=include_pdf,
            validation_policy=validation_policy,
            export_mode=export_mode,
        )
    except MusicValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.report) from exc


def _validate_and_store(project: ArrangementProject, project_dir: Path) -> dict[str, Any]:
    report = validate_project(project)
    project.validation_report = report
    write_validation_json(report, project_dir / "validation_report.json")
    write_validation_html(report, project_dir / "validation_report.html")
    project.save_json(project_dir / "arrangement_project.json")
    return report


def _save_project(project: ArrangementProject, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    project.save_json(project_dir / "arrangement_project.json")
    if project.generation_spec is not None:
        _write_json(
            project_dir / "generation_spec.json",
            project.generation_spec.model_dump(mode="json"),
        )


def _load_project(project_id: str) -> ArrangementProject:
    project_path = _project_dir(project_id) / "arrangement_project.json"
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return ArrangementProject.load_json(project_path)


def _project_metadata(project: ArrangementProject) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "metadata": project.metadata,
        "bar_count": project.bar_count,
        "tracks": [
            {
                "id": track.id,
                "instrument": track.instrument,
                "role": track.role,
                "bars": track.bar_count,
                "metadata": track.metadata,
            }
            for track in project.tracks
        ],
    }


def _take_snapshot(
    project_id: str,
    take_id: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    active_project = _load_project(project_id)
    project_dir = _project_dir(project_id)
    manager = TakeManager(project_dir)
    takes = manager.list_takes(project=active_project)
    take = next((item for item in takes["takes"] if item["take_id"] == take_id), None)
    if take is None:
        raise HTTPException(status_code=404, detail=f"Take not found: {take_id}")
    snapshot_path = take.get("project_snapshot_path")
    if not snapshot_path:
        raise HTTPException(status_code=409, detail=f"Take has no snapshot: {take_id}")
    return take, _existing_project_path(project_dir, Path(str(snapshot_path))), takes


def _project_diff(
    active_project: ArrangementProject,
    candidate_project: ArrangementProject,
) -> dict[str, Any]:
    active_tracks = {track.id: track for track in active_project.tracks}
    candidate_tracks = {track.id: track for track in candidate_project.tracks}
    track_diffs: list[dict[str, Any]] = []
    changed_bars: list[dict[str, Any]] = []
    for track_id in sorted(set(active_tracks) | set(candidate_tracks)):
        active_track = active_tracks.get(track_id)
        candidate_track = candidate_tracks.get(track_id)
        if active_track is None:
            bars = [bar.number for bar in candidate_track.bars] if candidate_track else []
            track_diffs.append(
                {
                    "track_id": track_id,
                    "status": "added",
                    "changed_bars": bars,
                    "active_note_count": 0,
                    "candidate_note_count": _track_note_count(candidate_track),
                    "note_delta": _track_note_count(candidate_track),
                }
            )
            changed_bars.extend(
                {
                    "track_id": track_id,
                    "bar": bar_number,
                    "active_note_count": 0,
                    "candidate_note_count": _bar_note_count(
                        next(bar for bar in candidate_track.bars if bar.number == bar_number)
                    )
                    if candidate_track
                    else 0,
                    "note_delta": _bar_note_count(
                        next(bar for bar in candidate_track.bars if bar.number == bar_number)
                    )
                    if candidate_track
                    else 0,
                }
                for bar_number in bars
            )
            continue
        if candidate_track is None:
            bars = [bar.number for bar in active_track.bars]
            track_diffs.append(
                {
                    "track_id": track_id,
                    "status": "removed",
                    "changed_bars": bars,
                    "active_note_count": _track_note_count(active_track),
                    "candidate_note_count": 0,
                    "note_delta": -_track_note_count(active_track),
                }
            )
            changed_bars.extend(
                {
                    "track_id": track_id,
                    "bar": bar.number,
                    "active_note_count": _bar_note_count(bar),
                    "candidate_note_count": 0,
                    "note_delta": -_bar_note_count(bar),
                }
                for bar in active_track.bars
            )
            continue

        active_bars = {bar.number: bar for bar in active_track.bars}
        candidate_bars = {bar.number: bar for bar in candidate_track.bars}
        track_changed_bars: list[int] = []
        for bar_number in sorted(set(active_bars) | set(candidate_bars)):
            before_bar = active_bars.get(bar_number)
            after_bar = candidate_bars.get(bar_number)
            if _bar_payload(before_bar) == _bar_payload(after_bar):
                continue
            before_notes = _bar_note_count(before_bar)
            after_notes = _bar_note_count(after_bar)
            track_changed_bars.append(bar_number)
            changed_bars.append(
                {
                    "track_id": track_id,
                    "bar": bar_number,
                    "active_note_count": before_notes,
                    "candidate_note_count": after_notes,
                    "note_delta": after_notes - before_notes,
                }
            )
        track_diffs.append(
            {
                "track_id": track_id,
                "status": "modified" if track_changed_bars else "unchanged",
                "changed_bars": track_changed_bars,
                "active_note_count": _track_note_count(active_track),
                "candidate_note_count": _track_note_count(candidate_track),
                "note_delta": _track_note_count(candidate_track)
                - _track_note_count(active_track),
            }
        )

    modified_tracks = [track for track in track_diffs if track["status"] != "unchanged"]
    return {
        "summary": {
            "changed_tracks": len(modified_tracks),
            "changed_bars": len(changed_bars),
            "active_note_count": sum(_track_note_count(track) for track in active_project.tracks),
            "candidate_note_count": sum(
                _track_note_count(track) for track in candidate_project.tracks
            ),
        },
        "tracks": track_diffs,
        "changed_bars": changed_bars,
    }


def _track_note_count(track: Any) -> int:
    if track is None:
        return 0
    return sum(_bar_note_count(bar) for bar in track.bars)


def _bar_note_count(bar: Any) -> int:
    if bar is None:
        return 0
    return sum(1 for event in bar.events if getattr(event, "type", None) == "note")


def _bar_payload(bar: Any) -> dict[str, Any] | None:
    return bar.model_dump(mode="json") if bar is not None else None


def _export_package_paths(project_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for record in manifest.get("files", []):
        if record.get("status", "created") == "skipped":
            continue
        raw_path = record.get("path")
        if not raw_path:
            continue
        path = _existing_project_path(project_dir, Path(str(raw_path)))
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(project_dir).as_posix())


def _project_file_path(project_id: str, *, kind: str, track_id: str | None) -> Path:
    project_dir = _project_dir(project_id)
    _load_project(project_id)
    normalized_kind = FILE_KIND_ALIASES.get(kind, kind)
    manifest = _read_json_file(project_dir / "export_manifest.json")
    for record in manifest.get("files", []):
        if record.get("kind") != normalized_kind:
            continue
        if track_id is not None and record.get("track_id") != track_id:
            continue
        return _existing_project_path(project_dir, Path(str(record.get("path", ""))))

    direct_files = {
        "export_manifest": project_dir / "export_manifest.json",
        "generation_spec_json": project_dir / "generation_spec.json",
        "project_json": project_dir / "arrangement_project.json",
        "validation_report_html": project_dir / "validation_report.html",
        "validation_report_json": project_dir / "validation_report.json",
    }
    direct_path = direct_files.get(normalized_kind)
    if direct_path is not None and direct_path.exists():
        return _existing_project_path(project_dir, direct_path)
    raise HTTPException(
        status_code=404,
        detail=f"Project file not found for kind={kind!r}",
    )


def _existing_project_path(project_dir: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_dir / path).resolve()
    if project_dir.resolve() not in resolved.parents and resolved != project_dir.resolve():
        raise HTTPException(status_code=400, detail="Requested file is outside project storage")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Project file not found: {resolved.name}")
    return resolved


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mid":
        return "audio/midi"
    if suffix in {".musicxml", ".xml"}:
        return "application/vnd.recordare.musicxml+xml"
    if suffix == ".json":
        return "application/json"
    if suffix == ".html":
        return "text/html"
    if suffix == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _pattern_index_paths(dataset_id: str | None) -> list[Path]:
    datasets_root = _storage_root() / "datasets"
    if dataset_id is not None:
        dataset_dir = _dataset_dir(dataset_id)
        index_path = dataset_dir / "pattern_index.json"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail=f"Pattern index not found: {dataset_id}")
        return [index_path]

    if not datasets_root.exists():
        return []
    return sorted(datasets_root.glob("*/pattern_index.json"))


def _storage_root() -> Path:
    configured = os.environ.get(API_STORAGE_ENV)
    root = Path(configured).expanduser() if configured else Path.cwd() / "outputs" / "api"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _project_dir(project_id: str) -> Path:
    return _storage_root() / "projects" / _validate_id(project_id)


def _dataset_dir(dataset_id: str) -> Path:
    return _storage_root() / "datasets" / _validate_id(dataset_id)


def _safe_id(value: str | None, *, prefix: str) -> str:
    if value is None or value == "":
        return f"{prefix}-{uuid4().hex[:12]}"
    return _validate_id(value)


def _validate_id(value: str) -> str:
    if not ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid id: {value!r}")
    return value


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _split_tags(tags: str | None) -> list[str] | None:
    if tags is None:
        return None
    values = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return values or None
