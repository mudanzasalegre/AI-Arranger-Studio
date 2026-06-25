from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactStatus = Literal["raw", "imported", "validated", "rejected"]
TakeStatus = Literal["pending", "accepted", "rejected"]
TakeSource = Literal["rule_based", "model", "manual", "retrieval"]


class TakeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelArtifactRecord(TakeModel):
    artifact_id: str
    project_id: str | None
    backend_id: str
    task: str
    artifact_type: str
    raw_path: str
    imported_path: str | None = None
    validated_path: str | None = None
    rejected_path: str | None = None
    status: ArtifactStatus
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArrangementTake(TakeModel):
    take_id: str
    project_id: str
    parent_take_id: str | None = None
    source: TakeSource
    backend_id: str | None = None
    task: str | None = None
    track_id: str | None = None
    bars: list[int] = Field(default_factory=list)
    instruction: str | None = None
    seed: int | None = None
    status: TakeStatus = "pending"
    validation_report_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    project_snapshot_path: str | None = None
    created_at: str
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
