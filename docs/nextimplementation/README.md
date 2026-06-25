# Next Implementation Plan v2.1 - PR roadmap LLM simbolico MIDI

Este directorio sustituye al plan anterior de `docs/nextimplementation`.

## Punto de partida

El proyecto parte de este estado:

```text
Epic 15 - Musical QA baseline
Epic 16 - SongPlan / SectionPlan / PhrasePlan / GrooveMap
```

No se debe continuar directamente con el antiguo Epic 17. Antes hay que insertar
una capa segura de LLM/modelos simbolicos que proponga material MIDI sin tomar el
control del proyecto activo.

## Decision principal

La jerarquia obligatoria es:

```text
SongPlan manda.
SectionPlan organiza.
PhrasePlan frasea.
GrooveMap coordina acentos y espacio.
RoleIntent limita cada instrumento.
El modelo propone.
El validador decide.
El usuario acepta una take.
El exportador produce el MIDI/MusicXML/PDF final.
```

No se acepta este flujo:

```text
prompt -> LLM -> MIDI final directo
```

El flujo correcto es:

```text
prompt
  -> LLM Planner JSON
  -> SongPlan / SectionPlan / PhrasePlan / GrooveMap
  -> base rule-based solida
  -> backend simbolico para infill / generate-track / sketch
  -> artifact quarantine
  -> importacion a ArrangementProject
  -> fusion controlada
  -> validacion musical
  -> take aceptada o rechazada
  -> exportacion profesional
```

## Cola estricta de PRs

```text
PR-00 - Sustituir docs/nextimplementation
PR-01 - Congelar Epic 15/16
PR-02 - Epic 16.5 AI Model Backend Contract
PR-03 - Epic 16.6 Artifact Quarantine + Takes
PR-04 - Epic 16.7 LLM Planner JSON
PR-05 - Epic 16.8 MIDI-GPT Backend
PR-06 - Epic 16.9 Text2MIDI Sketch Backend
PR-07 - Epic 17 GrooveMap / PerformanceMap 2.0
PR-08 - Epic 18 Melody Engine 2.0
PR-09 - Epic 19 Bass Engine 2.0
PR-10 - Epic 20 Drums Engine 2.0
PR-11 - Epic 21 Piano / Comping Engine 2.0
PR-12 - Epic 22 Dataset Profiler / Role Classifier
PR-13 - Epic 23 Retrieval Model
PR-14 - Epic 24 App Workflow 2.0
PR-15 - Epic 25 DAW-ready Export
PR-16 - Epic 26 Release Quality Gate
PR-17 - Epic 27 Tokenization Dataset Export
PR-18 - Epic 28 Baseline Statistical Role Models
PR-19 - Epic 29 Custom Role Model Interface
```

La parte nueva critica es `PR-02` a `PR-06`: blindar la entrada de modelos
simbolicos/LLM antes de continuar mejorando motores musicales.

## Orden de lectura recomendado

1. `00_CONTEXT_AND_DECISIONS.md`
2. `01_DIAGNOSIS.md`
3. `02_TARGET_ARCHITECTURE.md`
4. `03_LLM_SYMBOLIC_MIDI_STRATEGY.md`
5. `04_MODEL_BACKENDS_CONTRACT.md`
6. `05_ARTIFACT_QUARANTINE_TAKES.md`
7. `06_LLM_PLANNER_JSON.md`
8. `07_MIDIGPT_BACKEND.md`
9. `08_TEXT2MIDI_SKETCH_BACKEND.md`
10. `09_MUSICAL_ENGINE_ROADMAP.md`
11. `10_DATA_AND_LEARNING_ROADMAP.md`
12. `11_API_WORKER_ROADMAP.md`
13. `12_FRONTEND_WORKFLOW.md`
14. `13_MIDI_RENDERING_DAW_EXPORT.md`
15. `14_QA_AND_ACCEPTANCE.md`
16. `15_IMPLEMENTATION_BACKLOG.md`
17. `16_EXECUTION_INSTRUCTIONS.md`
18. `17_CODEX_PROMPTS.md`
19. `18_FILE_MAP_AND_STUBS.md`
20. `19_LICENSE_AND_MODEL_RISK.md`

## Regla de oro

No avanzar de PR si no pasan:

```bash
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
python scripts/golden_generate.py
```

Si un comando no existe, crear el minimo equivalente o documentar claramente el
equivalente actual en `docs/progress/`.
