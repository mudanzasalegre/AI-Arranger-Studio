from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerStatus(WorkerModel):
    status: str = "ok"
    worker: str = "ai-arranger-model-worker"
    models: list[dict[str, object]] = Field(default_factory=list)
