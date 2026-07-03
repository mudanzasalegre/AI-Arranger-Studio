from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerStatus(WorkerModel):
    status: str = "ok"
    worker: str = "ai-arranger-model-worker"
    models: list[dict[str, object]] = Field(default_factory=list)


class Text2MidiSketchRequest(WorkerModel):
    prompt: str
    output_path: str
    seed: int | None = None
    temperature: float = 0.8
    max_len: int = 2000
    device: str = "auto"
    repo_dir: str = "models/external_repos/text2midi"
    checkpoint_dir: str = "models/checkpoints/text2midi"
    model_file: str = "pytorch_model.bin"
    tokenizer_file: str = "vocab_remi.pkl"
    flan_tokenizer: str = "google/flan-t5-base"


class Text2MidiSketchResponse(WorkerModel):
    status: str
    output_path: str
    metadata: dict[str, object] = Field(default_factory=dict)
