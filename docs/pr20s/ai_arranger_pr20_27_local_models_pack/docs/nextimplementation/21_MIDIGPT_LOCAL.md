# PR-21 — MIDI-GPT local

## Objetivo

Instalar y activar MIDI-GPT como backend simbólico local para `infill_bars`, `generate_track`, `continue_section` y `generate_variation`.

## Uso dentro de AI Arranger Studio

Permitido:

```text
- Regenerar compases concretos.
- Crear una pista nueva sobre un proyecto existente.
- Continuar una sección.
- Crear variaciones de melodía/piano/vientos.
```

Prohibido:

```text
- Generar export final directo.
- Saltarse artifact quarantine.
- Reemplazar Harmony Engine o validadores.
```

## Instalación

```bash
python -m pip install -r requirements-ai.txt
# o solo:
python -m pip install "midigpt[inference]"
```

Con `HF_HOME` y `HF_HUB_CACHE` configurados, el modelo se cacheará bajo `models/hf_cache/`.

## Smoke test de carga

```bash
python scripts/models/download_midigpt.py --model-name yellow
python scripts/models/smoke_midigpt.py --model-name yellow --output outputs/model_artifacts/raw/midigpt_smoke.mid
```

## Adaptación obligatoria del backend

El backend actual probablemente necesita ajustarse a la API real de MIDI-GPT.

Implementar en `packages/model_backends/model_backends/symbolic/midigpt_backend.py`:

1. Import real:

```python
from midigpt import Score
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt
```

2. Cargar contexto:

```python
score = Score.from_midi(str(context_midi_path))
```

3. Resolver track index:

```python
track_index = resolve_track_index(score, request.track_id)
```

4. Crear request:

```python
generation_request = GenerationRequest(
    tracks=[TrackPrompt(id=track_index, bars=[bar - 1 for bar in request.bars])],
    config=InferenceConfig(
        temperature=request.temperature,
        top_p=0.95,
        model_dim=_model_dim_for_bars(request.bars),
        mask_mode="attention",
    ),
)
```

5. Ejecutar:

```python
result = engine.session(score, generation_request).run()
result.to_midi(str(output_path))
```

Nota: internamente tu app usa compases 1-based; MIDI-GPT usa índices de barras 0-based en sus ejemplos. Convertir siempre con `bar - 1` al enviar, y volver a 1-based al validar.

## Mapeo de pistas

Añadir metadata estable al export de contexto:

```text
track.id -> track_name MIDI
alto_sax -> alto_sax
trumpet -> trumpet
piano -> piano
```

Si no puede resolver pista por nombre, rechazar con error claro.

## Activación

Cuando smoke test pase:

```yaml
# configs/ai_models.local.yaml
midigpt:
  enabled: true
```

y `.env`:

```env
AI_MODELS_CONFIG=./configs/ai_models.local.yaml
AI_ENABLE_MIDIGPT=true
```

## API test

1. Generar proyecto rule-based.
2. Llamar:

```bash
curl -X POST http://127.0.0.1:8000/v1/projects/PROJECT_ID/ai/infill \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "midigpt",
    "track_id": "alto_sax",
    "bars": [5,6,7,8],
    "instruction": "bebop phrase, medium density, clear cadence",
    "temperature": 0.9,
    "seed": 2101
  }'
```

Debe devolver `status: pending_take`.

## Acceptance criteria

```text
- MIDI-GPT carga localmente.
- Produce un MIDI smoke.
- `/v1/projects/{id}/ai/infill` crea una take pendiente.
- El proyecto activo no cambia hasta accept.
- Artifact queda en raw -> imported/validated o rejected.
- Si el modelo genera fuera de rango, ValidationGate rechaza.
- model_trace registra backend, seed, track, bars, instruction y commercial_use.
```
