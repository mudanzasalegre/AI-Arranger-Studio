# PR-23 — Local LLM Planner

## Objetivo

Conectar un LLM local para convertir prompts de usuario en `SongPlan` JSON válido, sin generar notas.

## Backend recomendado para empezar

Ollama.

Razones:

- Instalación simple.
- API local HTTP.
- No requiere claves externas.
- Suficiente para JSON planning.

## Instalación manual

1. Instalar Ollama desde su web oficial.
2. Descargar modelo:

```bash
ollama pull qwen3:8b
```

Alternativas si el equipo no soporta 8B:

```bash
ollama pull qwen3:4b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## Smoke test

```bash
python scripts/models/smoke_ollama_planner.py \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/api
```

Debe devolver JSON parseable con:

```text
style
key
tempo
meter
form
ensemble
sections
instruments
generation_strategy
```

## Implementación

Crear:

```text
packages/model_backends/model_backends/planner/
  __init__.py
  ollama_planner_backend.py
```

El backend debe:

1. Construir prompt de sistema estricto.
2. Pedir JSON, no texto libre.
3. Validar con `LlmSongPlanPatch`.
4. Reintentar una vez con el error de Pydantic.
5. Si falla, caer a fallback rule-based.

## Conexión con `LlmPlanner`

Actualmente `LlmPlanner(provider=None)` cae a fallback. Cambiar la ruta `/v1/projects/{id}/ai/plan` para construir provider desde config si `local_llm_planner.enabled=true`.

Pseudocódigo:

```python
provider = None
if local_llm_planner_enabled:
    provider = OllamaPlannerJsonProvider(...)
planner = LlmPlanner(provider=provider)
```

## Reglas duras

El LLM planner no puede:

```text
- escribir notas;
- modificar tracks directamente;
- exportar;
- generar audio;
- cambiar locked_tracks;
- pedir backends de audio.
```

## Acceptance criteria

```text
- Ollama responde localmente.
- `/v1/projects/{id}/ai/plan` usa LLM local cuando está habilitado.
- JSON inválido reintenta una vez.
- Si falla, fallback rule-based.
- El endpoint verifica que las pistas no cambiaron.
- Plan version queda guardado.
```
