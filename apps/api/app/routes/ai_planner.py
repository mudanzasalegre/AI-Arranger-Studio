from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from arranger_core import ArrangementProject, LlmPlanner
from fastapi import APIRouter, HTTPException
from model_backends import (
    ModelBackendConfigurationError,
    ModelBackendUnavailableError,
    build_model_backend_registry,
    load_ai_models_config,
)
from pydantic import BaseModel, ConfigDict, Field

API_STORAGE_ENV = "AI_ARRANGER_API_STORAGE"
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

router = APIRouter()


class AiPlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiPlanRequest(AiPlannerModel):
    prompt: str = Field(default="")
    mode: Literal["create_or_patch_plan"] = "create_or_patch_plan"
    locked_tracks: list[str] = Field(default_factory=list)
    locked_sections: list[str] = Field(default_factory=list)
    seed: int | None = None


@router.post("/v1/projects/{project_id}/ai/plan")
def plan_project_with_ai(project_id: str, payload: AiPlanRequest) -> dict[str, Any]:
    project_dir = _project_dir(project_id)
    project_path = project_dir / "arrangement_project.json"
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    project = ArrangementProject.load_json(project_path)
    track_snapshot = [track.model_dump(mode="json") for track in project.tracks]
    result = LlmPlanner(provider=_configured_planner_provider()).plan(
        prompt=payload.prompt,
        project=project,
        mode=payload.mode,
        locked_tracks=payload.locked_tracks,
        locked_sections=payload.locked_sections,
        seed=payload.seed,
    )
    if result.status != "ok":
        raise HTTPException(status_code=422, detail=result.validation)

    project.metadata = _updated_metadata(
        project.metadata,
        prompt=payload.prompt,
        mode=payload.mode,
        locked_tracks=payload.locked_tracks,
        locked_sections=payload.locked_sections,
        result=result.model_dump(mode="json"),
    )
    if [track.model_dump(mode="json") for track in project.tracks] != track_snapshot:
        raise HTTPException(status_code=500, detail="AI planner attempted to modify project tracks")

    plan_dir = project_dir / "plan_versions"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{result.plan_version}.json"
    _write_json(plan_path, result.model_dump(mode="json"))
    _write_json(project_dir / "song_plan.json", result.song_plan.model_dump(mode="json"))
    project.save_json(project_path)

    return {
        "project_id": project.project_id,
        "status": "ok",
        "planner": result.planner,
        "plan_version": result.plan_version,
        "song_plan_patch": result.song_plan_patch.model_dump(mode="json"),
        "song_plan": result.song_plan.model_dump(mode="json"),
        "validation": result.validation,
        "attempts": [attempt.model_dump(mode="json") for attempt in result.attempts],
        "fallback_used": result.fallback_used,
        "files": [
            {"kind": "plan_version_json", "path": str(plan_path.relative_to(project_dir))},
            {"kind": "song_plan_json", "path": "song_plan.json"},
        ],
    }


def _updated_metadata(
    metadata: dict[str, Any],
    *,
    prompt: str,
    mode: str,
    locked_tracks: list[str],
    locked_sections: list[str],
    result: dict[str, Any],
) -> dict[str, Any]:
    plan_versions = list(metadata.get("plan_versions", []))
    plan_versions.append(
        {
            "plan_version": result["plan_version"],
            "planner": result["planner"],
            "created_at": result["created_at"],
            "validation_status": result["validation"]["status"],
        }
    )
    return {
        **metadata,
        "song_plan": result["song_plan"],
        "active_plan_version": result["plan_version"],
        "plan_versions": plan_versions,
        "ai_planner": {
            "prompt": prompt,
            "mode": mode,
            "locked_tracks": locked_tracks,
            "locked_sections": locked_sections,
            "planner": result["planner"],
            "fallback_used": result["fallback_used"],
            "validation": result["validation"],
            "song_plan_patch": result["song_plan_patch"],
        },
    }


def _storage_root() -> Path:
    configured = os.environ.get(API_STORAGE_ENV)
    root = Path(configured).expanduser() if configured else Path.cwd() / "outputs" / "api"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _project_dir(project_id: str) -> Path:
    return _storage_root() / "projects" / _validate_id(project_id)


def _validate_id(value: str) -> str:
    if not ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid id: {value!r}")
    return value


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _configured_planner_provider():
    try:
        config = load_ai_models_config()
    except ModelBackendConfigurationError as exc:
        return _UnavailablePlannerProvider(str(exc))

    backend_config = config.backends.get("local_llm_planner")
    if backend_config is None or not backend_config.enabled:
        return None

    try:
        registry = build_model_backend_registry(
            config=config,
            include_disabled=False,
            include_unavailable=True,
        )
        return registry.get("local_llm_planner")
    except (ModelBackendConfigurationError, ModelBackendUnavailableError, KeyError) as exc:
        return _UnavailablePlannerProvider(str(exc))


class _UnavailablePlannerProvider:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def generate_plan_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        previous_error: str | None = None,
    ) -> str:
        raise ModelBackendUnavailableError(self.reason)
