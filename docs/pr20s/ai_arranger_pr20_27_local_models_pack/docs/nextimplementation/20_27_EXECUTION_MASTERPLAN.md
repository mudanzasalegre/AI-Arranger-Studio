# PR-20 a PR-27 — Masterplan de modelos locales

## Punto de partida

Este plan parte de un repo que ya tiene PR-00 a PR-19 implementados:

- `ArrangementProject`, `SongPlan`, `SectionPlan`, `PhrasePlan`, `GrooveMap`.
- `ModelBackendContract` y `ModelBackendRegistry`.
- `ArtifactStore`, `ArtifactImporter`, `ProjectMerger`, `ValidationGate`, `TakeManager`.
- Backends opcionales para `midigpt` y `text2midi` aún desactivados.
- Dataset profiler, retrieval, tokenization bridge y custom role model interface.

## Principio rector

Los modelos locales no sustituyen al motor musical. Entran como generadores controlados.

```text
modelo local -> artifact raw -> importación -> fusión controlada -> validación -> take pendiente
```

El proyecto activo solo cambia cuando el usuario acepta la take.

## Orden estricto

```text
PR-20 — Local Model Runtime
PR-21 — MIDI-GPT local
PR-22 — Text2MIDI local
PR-23 — Local LLM Planner
PR-24 — MidiTok / training stack
PR-25 — Custom Role Model Bootstrap
PR-26 — Local Model Smoke Tests
PR-27 — Professional Generation Benchmark
```

## Gates globales antes de empezar

Ejecutar:

```bash
python -m pip install -r requirements.txt
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web install
npm --prefix apps/web run lint
python scripts/package_smoke.py
python scripts/golden_generate.py
python scripts/ai_contract_smoke.py
python scripts/tokenization_dataset_smoke.py
python scripts/statistical_baselines_smoke.py
python scripts/custom_role_model_smoke.py
```

Si un comando falla, no empezar PR-20.

## Modelo de carpetas local

```text
models/
  hf_cache/
    hub/
    assets/
  external_repos/
    text2midi/
  checkpoints/
    text2midi/
    custom/
      melody/
      bass/
      piano_comping/
      horns/
      drums/
  manifests/
    model_registry.yaml
    install_report.json
outputs/
  model_artifacts/
    raw/
    imported/
    rejected/
    validated/
```

`models/` y `outputs/` no se versionan.

## Variables base

Copiar `.env.local-models.example` a `.env` o fusionarlo con el `.env` existente.

Claves necesarias:

```env
AI_ARRANGER_API_STORAGE=./outputs/api
AI_MODELS_CONFIG=./configs/ai_models.local.yaml
LOCAL_MODEL_RUNTIME_CONFIG=./configs/local_model_runtime.yaml
MODEL_REGISTRY_CONFIG=./configs/model_registry.yaml
AI_MODELS_ROOT=./models
HF_HOME=./models/hf_cache
HF_HUB_CACHE=./models/hf_cache/hub
HF_HUB_DISABLE_TELEMETRY=1
AI_DEVICE=auto
```

## Cómo se activan modelos

1. Instalar dependencia.
2. Descargar/cachear pesos.
3. Pasar smoke test específico.
4. Cambiar `enabled: true` en `configs/ai_models.local.yaml`.
5. Pasar `make ai-local-smoke` o script equivalente.
6. Probar benchmark profesional.

Nunca activar un backend sin smoke test.

## Resultado final esperado

Al terminar PR-27:

```text
- LLM planner local produce SongPlan JSON válido.
- MIDI-GPT regenera compases/pistas mediante artifact quarantine.
- Text2MIDI genera sketches importados, nunca export final directo.
- MidiTok/training stack prepara datasets por rol.
- Custom role model interface puede cargar modelos propios futuros.
- Professional benchmark genera y exporta 5 demos jazz validadas.
```
