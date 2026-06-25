# 11 — API y model worker

## Objetivo

Añadir endpoints de IA simbólica sin cargar modelos pesados dentro del proceso principal si no es necesario.

## Endpoints API

```text
GET  /v1/ai/models
POST /v1/projects/{id}/ai/plan
POST /v1/projects/{id}/ai/infill
POST /v1/projects/{id}/ai/generate-track
POST /v1/ai/text-to-midi-sketch
GET  /v1/projects/{id}/takes
POST /v1/projects/{id}/takes/{take_id}/accept
POST /v1/projects/{id}/takes/{take_id}/reject
GET  /v1/projects/{id}/artifacts
```

## Modelo de respuesta común

```json
{
  "status": "ok|pending|rejected|error",
  "project_id": "...",
  "take_id": "...",
  "artifact_ids": [],
  "validation_report": {},
  "warnings": []
}
```

## Worker recomendado

```text
apps/model_worker/
  pyproject.toml
  app/
    main.py
    routes.py
    runtime.py
```

Endpoints internos:

```text
GET  /internal/models/status
POST /internal/models/midigpt/infill
POST /internal/models/text2midi/sketch
```

## Por qué worker separado

- Aísla `torch`, `transformers`, `midigpt`.
- Evita que la API caiga por memoria.
- Permite activar GPU solo en profile `ai`.
- Permite colas y jobs largos.
- Facilita mock en tests.

## Docker profile

```yaml
services:
  api:
    build: ./apps/api
    volumes:
      - ./outputs:/app/outputs
      - ./configs:/app/configs

  model-worker:
    build: ./apps/model_worker
    profiles: ["ai"]
    volumes:
      - ./outputs:/app/outputs
      - ./models:/app/models
      - ./configs:/app/configs
    environment:
      - AI_DEVICE=auto
      - HF_HOME=/app/models/hf_cache
```

## Modo sin IA real

Debe funcionar:

```bash
docker compose up api web
```

## Modo con IA

Debe funcionar:

```bash
docker compose --profile ai up api web model-worker
```

## Acceptance criteria

- API funciona sin worker.
- Worker puede estar desactivado.
- `GET /v1/ai/models` indica unavailable/disabled correctamente.
- Todos los endpoints funcionan con mock.
- Ningún endpoint IA modifica proyecto activo sin take.
