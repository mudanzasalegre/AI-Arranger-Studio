# AI Arranger Studio — PR-20 a PR-27 Local Models Pack

Este paquete se copia sobre la raíz del repositorio `AI-Arranger-Studio` después de haber terminado PR-19.

No instala modelos por sí solo. Añade la receta completa, configuraciones, scripts y esqueletos para implementar e instalar los modelos locales en orden controlado.

## Objetivo de esta fase

Convertir el proyecto en una aplicación local capaz de generar y mejorar arreglos MIDI profesionales usando modelos simbólicos, sin modelos de audio todavía.

Flujo obligatorio:

```text
Prompt / proyecto
  -> SongPlan / SectionPlan / PhrasePlan / GrooveMap
  -> base rule-based / retrieval
  -> modelo simbólico local
  -> artifact raw
  -> importación a ArrangementProject
  -> fusión controlada
  -> validación musical
  -> take pendiente
  -> aceptación manual
  -> export DAW-ready
```

Nunca:

```text
modelo -> MIDI final directo
```

## Qué añade este pack

```text
docs/nextimplementation/
  20_27_EXECUTION_MASTERPLAN.md
  20_LOCAL_MODEL_RUNTIME.md
  21_MIDIGPT_LOCAL.md
  22_TEXT2MIDI_LOCAL.md
  23_LOCAL_LLM_PLANNER.md
  24_MIDITOK_TRAINING_STACK.md
  25_CUSTOM_ROLE_MODEL_BOOTSTRAP.md
  26_LOCAL_MODEL_SMOKE_TESTS.md
  27_PROFESSIONAL_GENERATION_BENCHMARK.md
  20_27_CODEX_PROMPTS.md
  20_27_ACCEPTANCE_CHECKLIST.md
  LOCAL_MODEL_SECURITY_AND_LICENSE.md

configs/
  ai_models.local.example.yaml
  local_model_runtime.example.yaml
  model_registry.example.yaml
  professional_benchmarks.yaml

scripts/models/
  ensure_local_model_dirs.py
  check_local_model_runtime.py
  download_midigpt.py
  smoke_midigpt.py
  download_text2midi.py
  smoke_text2midi.py
  smoke_ollama_planner.py
  ai_local_smoke.py
  professional_generation_benchmark.py
  write_install_report.py
  README.md

apps/model_worker/
  README.md
  pyproject.toml
  app/main.py
  app/runtime.py
  app/schemas.py

PATCHES/
  Makefile.additions.txt
  env_example.additions.txt
  ai_models_yaml_update_notes.md
  midigpt_backend_real_api_notes.md
```

## Instalación del pack

Desde la raíz del repo:

```bash
unzip ai_arranger_pr20_27_local_models_pack.zip -d .
```

Después lee:

```text
docs/nextimplementation/20_27_EXECUTION_MASTERPLAN.md
```

## Orden estricto de implementación

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

## Primer comando después de copiar

```bash
python scripts/models/ensure_local_model_dirs.py
python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.example.yaml
```

## Importante

`models/`, `outputs/`, datasets privados y checkpoints siguen sin versionarse en Git. Este pack solo añade recetas, scripts y configs reproducibles.
