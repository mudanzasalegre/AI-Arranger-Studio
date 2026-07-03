from __future__ import annotations

import shutil
from pathlib import Path

from app.runtime import model_runtime_status
from app.schemas import Text2MidiSketchRequest, Text2MidiSketchResponse, WorkerStatus
from fastapi import FastAPI, HTTPException
from model_backends import ModelGenerationRequest
from model_backends.errors import ModelBackendUnavailableError, ModelGenerationError
from model_backends.symbolic.text2midi_backend import Text2MidiBackend

app = FastAPI(title="AI Arranger Studio Model Worker", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"service": "ai-arranger-model-worker", "status": "ok"}


@app.get("/v1/models/status")
def models_status() -> WorkerStatus:
    return WorkerStatus(models=model_runtime_status())


@app.post("/v1/text2midi/sketch")
def generate_text2midi_sketch(payload: Text2MidiSketchRequest) -> Text2MidiSketchResponse:
    output_path = Path(payload.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend = Text2MidiBackend(
        repo_dir=payload.repo_dir,
        checkpoint_dir=payload.checkpoint_dir,
        model_file=payload.model_file,
        tokenizer_file=payload.tokenizer_file,
        flan_tokenizer=payload.flan_tokenizer,
        output_dir=output_path.parent,
        execution_mode="subprocess",
        device=payload.device,
        max_len=payload.max_len,
        temperature=payload.temperature,
    )
    try:
        result = backend.generate(
            ModelGenerationRequest(
                request_id=output_path.stem,
                task="generate_full_sketch",
                prompt=payload.prompt,
                seed=payload.seed,
                temperature=payload.temperature,
            )
        )
    except ModelBackendUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    artifact_path = Path(result.artifacts[0].path)
    if artifact_path.resolve() != output_path:
        shutil.copy2(artifact_path, output_path)
    return Text2MidiSketchResponse(
        status="ok",
        output_path=str(output_path),
        metadata=result.artifacts[0].metadata,
    )
