from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

MODEL_BACKENDS_CONTRACT_VERSION = "0.1.0"

ModelTask = Literal[
    "plan_song",
    "generate_full_sketch",
    "generate_track",
    "infill_bars",
    "continue_section",
    "reharmonize",
    "generate_variation",
]
ArtifactType = Literal["midi", "json", "tokens", "log"]
CommercialUse = Literal["allowed", "non_commercial", "review_required", "unknown"]
Density = Literal["low", "medium", "medium_high", "high"]


class ModelBackendModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCapabilities(ModelBackendModel):
    symbolic_midi: bool = False
    multitrack: bool = False
    bar_infill: bool = False
    track_generation: bool = False
    text_prompt: bool = False
    json_planning: bool = False
    token_output: bool = False
    supports_training: bool = False
    commercial_use: CommercialUse = "unknown"


class ModelGenerationRequest(ModelBackendModel):
    schema_version: str = MODEL_BACKENDS_CONTRACT_VERSION
    request_id: str = ""
    task: ModelTask
    project: dict[str, Any] | None = None
    song_plan: dict[str, Any] | None = None
    section_plan: dict[str, Any] | None = None
    phrase_plan: dict[str, Any] | None = None
    groove_map: dict[str, Any] | None = None
    role_intent: dict[str, Any] | None = None

    track_id: str | None = None
    bars: list[int] | None = None
    locked_tracks: list[str] = Field(default_factory=list)
    locked_bars: list[int] = Field(default_factory=list)

    instruction: str | None = None
    prompt: str | None = None
    style: str | None = None
    density: Density | None = None
    complexity: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelArtifact(ModelBackendModel):
    artifact_type: ArtifactType
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelGenerationResult(ModelBackendModel):
    schema_version: str = MODEL_BACKENDS_CONTRACT_VERSION
    backend_id: str
    task: ModelTask
    artifacts: list[ModelArtifact]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MusicModelBackend(Protocol):
    backend_id: str
    capabilities: ModelCapabilities

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        ...
