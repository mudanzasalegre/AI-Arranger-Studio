from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from arranger_core import (
    ArrangementProject,
    ArtifactImporter,
    ArtifactStore,
    ProjectMerger,
    SketchImportError,
    TakeManager,
    Text2MidiSketchImporter,
    ValidationGate,
    write_full_midi,
    write_validation_html,
    write_validation_json,
)
from arranger_core.ai.artifact_importer import ArtifactImportError
from fastapi import APIRouter, HTTPException
from model_backends import (
    ModelBackendConfigurationError,
    ModelBackendUnavailableError,
    ModelGenerationError,
    ModelGenerationRequest,
    UnsupportedModelTaskError,
    build_model_backend_registry,
    load_ai_models_config,
)
from pydantic import BaseModel, ConfigDict, Field

API_STORAGE_ENV = "AI_ARRANGER_API_STORAGE"
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

router = APIRouter()


class AiGenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiInfillRequest(AiGenerationModel):
    backend: str = "midigpt"
    track_id: str
    bars: list[int] = Field(min_length=1)
    instruction: str = ""
    density: str = "medium"
    temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    seed: int | None = None
    locked_tracks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextToMidiSketchRequest(AiGenerationModel):
    backend: str = "text2midi"
    prompt: str = Field(min_length=1)
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/v1/projects/{project_id}/ai/infill")
def infill_project_bars(project_id: str, payload: AiInfillRequest) -> dict[str, Any]:
    project_dir = _project_dir(project_id)
    project_path = project_dir / "arrangement_project.json"
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    project = ArrangementProject.load_json(project_path)
    target_track = _find_track(project, payload.track_id)
    if target_track is None:
        raise HTTPException(status_code=404, detail=f"Track not found: {payload.track_id}")
    bars = _validated_bars(payload.bars, project=project)
    if payload.track_id in set(payload.locked_tracks):
        raise HTTPException(status_code=409, detail=f"Target track is locked: {payload.track_id}")

    request_id = f"infill_{uuid4().hex[:12]}"
    context_midi_path = _write_context_midi(project, project_dir=project_dir, request_id=request_id)
    artifact_store = ArtifactStore(_artifact_store_root())
    backend = _load_backend(payload.backend, artifact_raw_dir=artifact_store.root / "raw")
    model_request = _model_request(
        payload,
        project=project,
        request_id=request_id,
        context_midi_path=context_midi_path,
        bars=bars,
        target_role=target_track.role,
        target_instrument=target_track.instrument,
    )

    try:
        generation_result = backend.generate(model_request)
    except UnsupportedModelTaskError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelBackendUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    records = artifact_store.store_generation_result(
        generation_result,
        project_id=project.project_id,
    )
    if not records:
        raise HTTPException(status_code=502, detail="Backend returned no artifacts")
    record = records[0]

    try:
        imported = ArtifactImporter(artifact_store=artifact_store).import_record(
            record,
            project=project,
            target_track_id=payload.track_id,
            target_bars=bars,
        )
        candidate = ProjectMerger().merge(
            project,
            imported,
            target_track_id=payload.track_id,
            target_bars=bars,
            locked_tracks=payload.locked_tracks,
        )
        validation_report = ValidationGate().validate_candidate(
            base_project=project,
            candidate_project=candidate,
            target_track_id=payload.track_id,
            target_bars=bars,
            locked_tracks=payload.locked_tracks,
        )
    except (ArtifactImportError, ValueError) as exc:
        rejected = artifact_store.mark_rejected(
            artifact_store.get(record.artifact_id),
            reason=str(exc),
        )
        raise HTTPException(
            status_code=422,
            detail={
                "status": "rejected",
                "artifact_id": rejected.artifact_id,
                "reason": str(exc),
            },
        ) from exc

    if validation_report["status"] == "fail":
        rejected = artifact_store.mark_rejected(
            artifact_store.get(record.artifact_id),
            reason="validation_failed",
            metadata={"validation": validation_report},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "status": "rejected",
                "artifact_id": rejected.artifact_id,
                "validation": validation_report,
            },
        )

    validated_record = artifact_store.mark_validated(
        artifact_store.get(record.artifact_id),
        metadata={"validation": validation_report},
    )
    take = TakeManager(project_dir).create_pending_take(
        base_project=project,
        candidate_project=candidate,
        artifact_records=[validated_record],
        validation_report=validation_report,
        track_id=payload.track_id,
        bars=bars,
        instruction=payload.instruction,
        seed=payload.seed,
        metadata={
            "model_trace": {
                "backend": generation_result.backend_id,
                "task": generation_result.task,
                "track_id": payload.track_id,
                "bars": bars,
                "instruction": payload.instruction,
                "density": payload.density,
                "temperature": payload.temperature,
                "seed": payload.seed,
                "context_midi_path": str(context_midi_path),
                "validation_status": validation_report["status"],
                "commercial_use": _commercial_use(payload.backend),
            }
        },
    )

    return {
        "project_id": project.project_id,
        "status": "pending_take",
        "backend": generation_result.backend_id,
        "take": take.model_dump(mode="json"),
        "artifact": artifact_store.get(validated_record.artifact_id).model_dump(mode="json"),
        "validation": validation_report,
        "context_midi_path": str(context_midi_path),
    }


@router.post("/v1/ai/text-to-midi-sketch")
def create_text_to_midi_sketch(payload: TextToMidiSketchRequest) -> dict[str, Any]:
    request_id = f"sketch_{uuid4().hex[:12]}"
    sketch_id = f"sketch_{uuid4().hex[:12]}"
    artifact_store = ArtifactStore(_artifact_store_root())
    backend = _load_backend(payload.backend, artifact_raw_dir=artifact_store.root / "raw")
    model_request = ModelGenerationRequest(
        request_id=request_id,
        task="generate_full_sketch",
        prompt=payload.prompt,
        instruction=payload.prompt,
        seed=payload.seed,
        metadata={
            **payload.metadata,
            "sketch_id": sketch_id,
            "sketch_only": True,
        },
    )

    try:
        generation_result = backend.generate(model_request)
    except UnsupportedModelTaskError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelBackendUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    records = artifact_store.store_generation_result(
        generation_result,
        project_id=sketch_id,
    )
    if not records:
        raise HTTPException(status_code=502, detail="Backend returned no artifacts")
    record = records[0]

    try:
        imported = Text2MidiSketchImporter().import_record(
            record,
            prompt=payload.prompt,
            seed=payload.seed,
            sketch_id=sketch_id,
        )
    except SketchImportError as exc:
        rejected = artifact_store.mark_rejected(record, reason=str(exc))
        raise HTTPException(
            status_code=422,
            detail={
                "status": "sketch_rejected",
                "artifact_id": rejected.artifact_id,
                "reason": str(exc),
            },
        ) from exc

    sketch_dir = _sketch_dir(sketch_id)
    project_path = imported.project.save_json(sketch_dir / "arrangement_project.json")
    validation_json_path = write_validation_json(
        imported.validation_report,
        sketch_dir / "validation_report.json",
    )
    validation_html_path = write_validation_html(
        imported.validation_report,
        sketch_dir / "validation_report.html",
    )
    metadata_path = _write_json(
        sketch_dir / "sketch_metadata.json",
        {
            "sketch_id": sketch_id,
            "status": imported.status,
            "backend": generation_result.backend_id,
            "prompt": payload.prompt,
            "seed": payload.seed,
            "artifact_id": record.artifact_id,
            "classifications": [
                classification.model_dump(mode="json")
                for classification in imported.classifications
            ],
            "uncertainty_reasons": imported.uncertainty_reasons,
            "limitations": imported.limitations,
            "commercial_use": _commercial_use(payload.backend),
        },
    )
    imported_record = artifact_store.mark_imported(
        artifact_store.get(record.artifact_id),
        imported_path=project_path,
        metadata={
            "sketch_id": sketch_id,
            "sketch_status": imported.status,
            "role_confidence": imported.role_confidence,
            "uncertainty_reasons": imported.uncertainty_reasons,
        },
    )

    if imported.status == "sketch_rejected":
        rejected = artifact_store.mark_rejected(
            imported_record,
            reason="validation_failed",
            metadata={
                "validation": imported.validation_report,
                "validation_report_path": str(validation_json_path),
            },
        )
        raise HTTPException(
            status_code=422,
            detail={
                "status": "sketch_rejected",
                "sketch_id": sketch_id,
                "artifact_id": rejected.artifact_id,
                "validation": imported.validation_report,
            },
        )

    validated = artifact_store.mark_validated(
        imported_record,
        validated_path=validation_json_path,
        metadata={
            "validation": imported.validation_report,
            "validation_report_path": str(validation_json_path),
        },
    )

    return {
        "status": imported.status,
        "sketch_id": sketch_id,
        "project_id": imported.project.project_id,
        "backend": generation_result.backend_id,
        "artifact": artifact_store.get(validated.artifact_id).model_dump(mode="json"),
        "sketch": {
            "bar_count": imported.project.bar_count,
            "tracks": [
                {
                    "id": track.id,
                    "instrument": track.instrument,
                    "role": track.role,
                    "bars": track.bar_count,
                    "metadata": track.metadata,
                }
                for track in imported.project.tracks
            ],
            "classifications": [
                classification.model_dump(mode="json")
                for classification in imported.classifications
            ],
            "uncertainty_reasons": imported.uncertainty_reasons,
            "limitations": imported.limitations,
        },
        "validation": imported.validation_report,
        "files": {
            "project": str(project_path),
            "validation_json": str(validation_json_path),
            "validation_html": str(validation_html_path),
            "metadata": str(metadata_path),
        },
    }


def _load_backend(backend_id: str, *, artifact_raw_dir: Path):
    try:
        config = load_ai_models_config()
        config = config.model_copy(
            update={
                "settings": {
                    **config.settings,
                    "artifact_raw_dir": str(artifact_raw_dir),
                }
            }
        )
        registry = build_model_backend_registry(
            config=config,
            include_disabled=True,
            include_unavailable=True,
        )
        return registry.get(backend_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelBackendUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelBackendConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _model_request(
    payload: AiInfillRequest,
    *,
    project: ArrangementProject,
    request_id: str,
    context_midi_path: Path,
    bars: list[int],
    target_role: str,
    target_instrument: str,
) -> ModelGenerationRequest:
    song_plan = project.metadata.get("song_plan")
    if not isinstance(song_plan, dict):
        song_plan = None
    return ModelGenerationRequest(
        request_id=request_id,
        task="infill_bars",
        project=project.model_dump(mode="json"),
        song_plan=song_plan,
        section_plan=_section_context(song_plan, bars),
        phrase_plan=_phrase_context(song_plan, bars),
        groove_map=song_plan.get("groove_map") if isinstance(song_plan, dict) else None,
        role_intent={
            "role": target_role,
            "instrument": target_instrument,
            "track_id": payload.track_id,
            "bars": bars,
            "density": payload.density,
            "instruction": payload.instruction,
        },
        track_id=payload.track_id,
        bars=bars,
        locked_tracks=payload.locked_tracks,
        instruction=payload.instruction,
        density=payload.density,  # type: ignore[arg-type]
        temperature=payload.temperature,
        seed=payload.seed,
        metadata={
            **payload.metadata,
            "context_midi_path": str(context_midi_path),
            "context_track_map": _context_track_map(project),
            "target_role": target_role,
            "target_instrument": target_instrument,
        },
    )


def _write_context_midi(
    project: ArrangementProject,
    *,
    project_dir: Path,
    request_id: str,
) -> Path:
    path = project_dir / "model_contexts" / f"{request_id}_context.mid"
    context_project = project.model_copy(deep=True)
    for track in context_project.tracks:
        track.name = track.id
        track.metadata = {
            **track.metadata,
            "model_context_track_name": track.id,
        }
    write_full_midi(context_project, path)
    _write_json(
        path.with_suffix(".track_map.json"),
        {
            "project_id": project.project_id,
            "tracks": _context_track_map(project),
        },
    )
    return path


def _context_track_map(project: ArrangementProject) -> dict[str, str]:
    return {track.id: track.id for track in project.tracks}


def _section_context(song_plan: dict[str, Any] | None, bars: list[int]) -> dict[str, Any] | None:
    if not song_plan:
        return None
    sections = song_plan.get("sections")
    if not isinstance(sections, list):
        return None
    target = set(bars)
    matches = [
        section
        for section in sections
        if target & set(range(int(section.get("start_bar", 0)), int(section.get("end_bar", 0)) + 1))
    ]
    return matches[0] if matches else None


def _phrase_context(song_plan: dict[str, Any] | None, bars: list[int]) -> dict[str, Any] | None:
    if not song_plan:
        return None
    phrases = song_plan.get("phrases")
    if not isinstance(phrases, list):
        return None
    target = set(bars)
    matches = [
        phrase
        for phrase in phrases
        if target & set(range(int(phrase.get("start_bar", 0)), int(phrase.get("end_bar", 0)) + 1))
    ]
    return matches[0] if matches else None


def _find_track(project: ArrangementProject, track_id: str):
    return next((track for track in project.tracks if track.id == track_id), None)


def _validated_bars(bars: list[int], *, project: ArrangementProject) -> list[int]:
    unique = sorted({int(bar) for bar in bars})
    invalid = [bar for bar in unique if bar < 1 or bar > project.bar_count]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "rejected",
                "reason": "bars_outside_project",
                "bars": invalid,
                "project_bar_count": project.bar_count,
            },
        )
    return unique


def _commercial_use(backend_id: str) -> str:
    try:
        config = load_ai_models_config()
    except ModelBackendConfigurationError:
        return "unknown"
    backend = config.backends.get(backend_id)
    if backend is None:
        return "unknown"
    return backend.commercial_use


def _storage_root() -> Path:
    configured = os.environ.get(API_STORAGE_ENV)
    root = Path(configured).expanduser() if configured else Path.cwd() / "outputs" / "api"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _artifact_store_root() -> Path:
    if os.environ.get(API_STORAGE_ENV):
        return _storage_root() / "model_artifacts"
    return Path.cwd() / "outputs" / "model_artifacts"


def _project_dir(project_id: str) -> Path:
    return _storage_root() / "projects" / _validate_id(project_id)


def _sketch_dir(sketch_id: str) -> Path:
    return _storage_root() / "sketches" / _validate_id(sketch_id)


def _validate_id(value: str) -> str:
    if not ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid id: {value!r}")
    return value


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
