from __future__ import annotations

from app.runtime import model_runtime_status
from app.schemas import WorkerStatus
from fastapi import FastAPI

app = FastAPI(title="AI Arranger Studio Model Worker", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "ai-arranger-model-worker", "status": "ok"}


@app.get("/v1/models/status")
def models_status() -> WorkerStatus:
    return WorkerStatus(models=model_runtime_status())
