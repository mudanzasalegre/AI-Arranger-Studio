# Prompts para Codex — PR-28 a PR-36

## Prompt base

```text
Lee README.md, docs/nextimplementation/*.md, configs/*.yaml, scripts/models/*.py y este archivo.

Estamos después de PR-27. El objetivo ahora es cerrar la fase de instalación automática y activación real de modelos locales para generación MIDI profesional.

No añadas modelos de audio.
No permitas que un modelo escriba el proyecto activo directamente.
No exportes pending takes.
No entrenes con datasets de licencia bloqueada.
```

## PR-28

```text
Implementa PR-28. Crea scripts/models_pro/pro_readiness_audit.py.
Debe ejecutar o comprobar los comandos base, detectar configs faltantes, variables .env, .gitignore, smoke scripts y escribir outputs/pro_audit/pr28_repo_health.json/md.
No instales modelos todavía.
```

## PR-29

```text
Implementa PR-29. Crea el instalador automático scripts/models_pro/install_all_local_models.py.
Debe instalar dependencias opcionales, crear carpetas, setear entorno, descargar/cachear MIDI-GPT, clonar/descargar Text2MIDI, comprobar Ollama, ejecutar smoke tests y escribir install_report.json.
Debe ser idempotente.
```

## PR-30

```text
Implementa PR-30. Reescribe MidiGptBackend para usar la API real de midigpt: Score.from_midi, InferenceEngine.from_pretrained/from_checkpoint, GenerationRequest, InferenceConfig, TrackPrompt, session(...).run(), result.to_midi().
Debe mapear track_id interno a track index y convertir barras 1-based a 0-based.
```

## PR-31

```text
Implementa PR-31. Fortalece Text2MIDI: instalación, wrapper subprocess, error reports, prompt templates y sketch import.
Text2MIDI sigue siendo sketch-only.
```

## PR-32

```text
Implementa PR-32. Conecta OllamaPlannerBackend a LlmPlanner en el endpoint /v1/projects/{id}/ai/plan mediante provider factory. Cuando local_llm_planner esté available debe usarse; si falla, fallback rule-based.
```

## PR-33

```text
Implementa PR-33. Endurece MidiTok pipeline: entrada desde manifests reales, splits train/val/test, filtros por licencia, filtros por rol/calidad, reportes de pérdida y smoke desde manifest.
```

## PR-34

```text
Implementa PR-34. Añade StatisticalCustomRoleBackend no dummy y scripts de entrenamiento n-gram/Markov por rol. Debe generar checkpoints locales y artifacts útiles que pasen importación/validación.
```

## PR-35

```text
Implementa PR-35. Crea ProfessionalGenerationOrchestrator y script generate_professional_midi.py. Debe usar planner, base rule-based/retrieval, custom role models, MIDI-GPT selective infill, Text2MIDI sketch opcional, validation gate y quality gate.
```

## PR-36

```text
Implementa PR-36. Crea ProQualityGate, thresholds, professional benchmark gate y release candidate report.
El benchmark debe bloquear outputs flojos y producir ratings A/B/C/D.
```
