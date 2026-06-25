# 04 — Contrato de backends de modelos

## Objetivo

Crear un contrato estable para integrar modelos simbólicos sin acoplar la app a MIDI-GPT, Text2MIDI u otro backend concreto.

## Estructura de archivos

```text
packages/model_backends/
  pyproject.toml
  model_backends/
    __init__.py
    base.py
    registry.py
    config.py
    errors.py
    artifact.py
    symbolic/
      __init__.py
      mock_backend.py
      midigpt_backend.py
      text2midi_backend.py
```

## base.py

```python
from typing import Literal, Protocol
from pydantic import BaseModel, Field

ModelTask = Literal[
    "plan_song",
    "generate_full_sketch",
    "generate_track",
    "infill_bars",
    "continue_section",
    "reharmonize",
    "generate_variation",
]

class ModelCapabilities(BaseModel):
    symbolic_midi: bool = False
    multitrack: bool = False
    bar_infill: bool = False
    track_generation: bool = False
    text_prompt: bool = False
    json_planning: bool = False
    token_output: bool = False
    supports_training: bool = False
    commercial_use: Literal[
        "allowed",
        "non_commercial",
        "review_required",
        "unknown",
    ] = "unknown"

class ModelGenerationRequest(BaseModel):
    task: ModelTask
    project: dict | None = None
    song_plan: dict | None = None
    section_plan: dict | None = None
    phrase_plan: dict | None = None
    groove_map: dict | None = None
    role_intent: dict | None = None

    track_id: str | None = None
    bars: list[int] | None = None
    locked_tracks: list[str] = Field(default_factory=list)
    locked_bars: list[int] = Field(default_factory=list)

    instruction: str | None = None
    prompt: str | None = None
    style: str | None = None
    density: Literal["low", "medium", "medium_high", "high"] | None = None
    complexity: float = 0.7
    temperature: float = 0.8
    seed: int | None = None

class ModelArtifact(BaseModel):
    artifact_type: Literal["midi", "json", "tokens", "log"]
    path: str
    metadata: dict = Field(default_factory=dict)

class ModelGenerationResult(BaseModel):
    backend_id: str
    task: ModelTask
    artifacts: list[ModelArtifact]
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_metadata: dict = Field(default_factory=dict)

class MusicModelBackend(Protocol):
    backend_id: str
    capabilities: ModelCapabilities

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        ...
```

## registry.py

```python
class ModelBackendRegistry:
    def __init__(self):
        self._backends = {}

    def register(self, backend):
        self._backends[backend.backend_id] = backend

    def get(self, backend_id: str):
        if backend_id not in self._backends:
            raise KeyError(f"Model backend not registered: {backend_id}")
        return self._backends[backend_id]

    def list(self):
        return [
            {
                "id": backend.backend_id,
                "capabilities": backend.capabilities.model_dump(),
            }
            for backend in self._backends.values()
        ]
```

## config.py

Debe cargar `configs/ai_models.yaml`.

Reglas:

- Backends desactivados no se registran salvo en modo `include_disabled=True`.
- Si una dependencia opcional falta, el backend puede registrarse como `unavailable` o devolver error controlado.
- El API debe arrancar aunque MIDI-GPT/Text2MIDI no estén instalados.

## Mock backend obligatorio

El `MockSymbolicBackend` debe existir desde el primer commit para tests.

Debe poder generar:

- artifact MIDI fixture válido;
- artifact MIDI inválido;
- artifact JSON válido;
- error simulado.

## Acceptance criteria

- `GET /v1/ai/models` lista backends.
- Los backends opcionales no rompen import-time.
- Tests cubren registry, config, mock y dependencia ausente.
- El contrato acepta `SongPlan`, `SectionPlan`, `PhrasePlan`, `GrooveMap` y `RoleIntent`.
