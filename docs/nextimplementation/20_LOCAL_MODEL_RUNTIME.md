# PR-20 — Local Model Runtime

## Objetivo

Preparar la infraestructura local para ejecutar modelos pesados sin contaminar el core ni el proceso principal de FastAPI.

## Tareas

### 1. Copiar configs y crear runtime local

```bash
cp configs/local_model_runtime.example.yaml configs/local_model_runtime.yaml
cp configs/ai_models.local.example.yaml configs/ai_models.local.yaml
cp configs/model_registry.example.yaml configs/model_registry.yaml
```

### 2. Fusionar `.env.local-models.example`

Actualizar `.env` con variables locales.

No versionar `.env`.

### 3. Crear carpetas locales

```bash
python scripts/models/ensure_local_model_dirs.py
```

Debe crear:

```text
models/hf_cache/hub
models/hf_cache/assets
models/external_repos
models/checkpoints/text2midi
models/checkpoints/custom/{melody,bass,piano_comping,horns,drums}
models/manifests
outputs/model_artifacts/{raw,imported,rejected,validated}
```

### 4. Añadir scripts de diagnóstico

Ejecutar:

```bash
python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.yaml
```

Debe comprobar:

- Python version.
- Directorios escribibles.
- Variables `HF_HOME`, `HF_HUB_CACHE`, `AI_MODELS_ROOT`.
- `configs/ai_models.local.yaml` existe.
- Ningún modelo pesado es obligatorio todavía.

### 5. Añadir targets Makefile

Aplicar `PATCHES/Makefile.additions.txt` manualmente.

Targets mínimos:

```make
models-bootstrap:
	python scripts/models/ensure_local_model_dirs.py

models-check:
	python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.yaml

ai-local-smoke:
	python scripts/models/ai_local_smoke.py
```

### 6. Worker de modelos

Crear `apps/model_worker` con el esqueleto incluido en este pack.

En PR-20 no debe cargar modelos reales. Solo debe exponer:

```text
GET /health
GET /v1/models/status
```

## Acceptance criteria

```text
- La app arranca sin modelos reales.
- `models/` y `outputs/model_artifacts/` existen localmente.
- `models/` no queda versionado.
- `configs/ai_models.local.yaml` existe pero mantiene midigpt/text2midi/local_llm_planner desactivados.
- `python scripts/models/check_local_model_runtime.py` pasa.
- `GET /v1/ai/models` sigue funcionando.
```

## Prompt para Codex

```text
Implementa PR-20 Local Model Runtime. No instales modelos. Copia configs locales, crea scripts/models, crea apps/model_worker con health/status, actualiza Makefile con targets de modelos, actualiza .env.example con variables locales y asegúrate de que la API funciona con mock_symbolic aunque no haya torch, midigpt ni transformers instalados.
```
