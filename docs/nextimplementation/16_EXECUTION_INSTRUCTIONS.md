# 16 - Instrucciones de ejecucion

## Regla general

Ejecutar un solo PR por iteracion. No avanzar al siguiente PR dentro del mismo
input salvo que el usuario lo pida explicitamente.

Cada PR debe terminar con:

- implementacion o cambio documental correspondiente;
- tests;
- smoke musical si aplica;
- documento en `docs/progress/`;
- comandos de verificacion;
- lista de riesgos pendientes.

## Secuencia obligatoria

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

## Gates obligatorios

```powershell
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
python scripts/golden_generate.py
```

Si frontend:

```powershell
npm --prefix apps/web run build
```

Si packaging:

```powershell
python scripts/package_smoke.py
```

Si un comando no existe:

1. crear el minimo equivalente si corresponde;
2. o documentar el equivalente actual en `docs/progress/<PR>.md`;
3. no continuar sin dejar test/smoke reproducible.

## Formato de progreso

Usar `templates/EPIC_PROGRESS_TEMPLATE.md` como base, aunque el identificador
sea `PR-XX`.

## Criterio de avance

No empezar el siguiente PR si:

- tests fallan;
- no hay doc de progreso;
- artifact IA puede modificar proyecto activo sin take;
- endpoints aceptan resultado IA sin validacion;
- hay backend de audio habilitado por error;
- se rompio export MIDI/MusicXML;
- se empeoraron golden metrics sin justificacion.
