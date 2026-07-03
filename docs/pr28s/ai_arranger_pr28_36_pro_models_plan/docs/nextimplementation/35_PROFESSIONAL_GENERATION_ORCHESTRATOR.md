# PR-35 — Professional generation orchestrator

## Objetivo

Crear un flujo que use los modelos en el orden correcto para generar MIDI profesional.

## Nuevo comando

```bash
python scripts/models_pro/generate_professional_midi.py \
  --prompt "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto..." \
  --profile pro \
  --seed 1234
```

## Pipeline

```text
1. Prompt
2. LLM Planner JSON
3. SongPlan/SectionPlan/PhrasePlan/GrooveMap
4. Base rule-based/retrieval
5. Custom role models si disponibles
6. MIDI-GPT solo en zonas concretas:
   - melodía débil
   - horn responses
   - piano comping
7. Text2MIDI solo como sketch comparativo/opcional
8. ValidationGate
9. QualityGate
10. pending take
11. export si pasa
```

## Decisiones importantes

### Text2MIDI

No debe alimentar el resultado final salvo que:

- se importe correctamente;
- se clasifiquen roles;
- pase validación;
- el usuario lo seleccione explícitamente.

Por defecto: `sketch_reference`.

### MIDI-GPT

Usar en:

```text
melody bars con baja calidad
horn_response huecos
piano comping variations
```

No usar en:

```text
walking_bass completo
drums completo
estructura armónica
```

### Custom role models

Si existen checkpoints reales, usarlos antes de MIDI-GPT para cada rol.

### Retry loop

```text
intento 1: creativo
intento 2: más conservador
intento 3: fallback rule-based/retrieval
```

## Output

```text
outputs/pro_benchmarks/<run_id>/
  arrangement_project.json
  full_arrangement.mid
  midi_tracks/
  full_score.musicxml
  validation_report.json
  quality_report.json
  model_trace.json
  takes_manifest.json
  generation_summary.md
```

## Acceptance

El benchmark profesional debe producir al menos 5 presets sin errores bloqueantes.
