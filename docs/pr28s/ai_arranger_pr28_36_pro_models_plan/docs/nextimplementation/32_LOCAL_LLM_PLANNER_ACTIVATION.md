# PR-32 — Local LLM Planner activo por defecto en perfil pro

## Objetivo

Conectar `OllamaPlannerBackend` al endpoint `/v1/projects/{id}/ai/plan` de forma real.

## Problema actual

`LlmPlanner` acepta un provider, pero el endpoint instancia:

```python
LlmPlanner()
```

Eso usa fallback rule-based si no se inyecta provider.

## Cambios obligatorios

### 1. Provider factory

Crear:

```text
packages/arranger_core/arranger_core/planning/provider_factory.py
```

Función:

```python
def build_planner_provider_from_registry() -> PlannerJsonProvider | None:
    ...
```

Debe:

1. leer `AI_MODELS_CONFIG`;
2. construir registry;
3. buscar `local_llm_planner`;
4. si está available, devolver backend;
5. si no, devolver `None`.

### 2. Endpoint

Cambiar:

```python
result = LlmPlanner().plan(...)
```

por:

```python
provider = build_planner_provider_from_registry()
result = LlmPlanner(provider=provider).plan(...)
```

### 3. Config pro

En `configs/ai_models.pro.yaml`:

```yaml
local_llm_planner:
  enabled: true
```

### 4. JSON schema

Mantener `format` como JSON schema cuando Ollama lo soporte. Si falla, caer a `"json"` y validar con Pydantic.

### 5. Modelo recomendado

Por defecto:

```text
qwen3:8b
```

Alternativas:

```text
qwen3:4b  → máquinas pequeñas
qwen3:14b → más calidad si hay VRAM/RAM
```

## Acceptance

```bash
python scripts/models/smoke_ollama_planner.py
```

y:

```bash
curl -X POST http://127.0.0.1:8000/v1/projects/<id>/ai/plan ...
```

deben mostrar:

```text
planner: llm
fallback_used: false
```

cuando Ollama esté disponible.
