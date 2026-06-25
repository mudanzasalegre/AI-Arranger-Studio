# 17 - Prompts listos para Codex

## Prompt PR-00

```text
Implementa PR-00 - Sustituir docs/nextimplementation end-to-end.
Lee la carpeta docs/nextimplementation completa y usa el plan PR-00 a PR-19
como fuente de verdad. No implementes PR-01 ni PR-02 todavia.
Al terminar, verifica que README.md reconoce Epic 15/16 como punto de partida,
que la carpeta contiene documentos 00 a 19 sin duplicados antiguos y que la
siguiente unidad de trabajo es PR-01.
Ejecuta los gates indicados y documenta en docs/progress/PR_00.md.
```

## Prompt PR-01

```text
Implementa PR-01 - Congelar Epic 15/16 end-to-end.
Confirma que SongPlan serializa/deserializa, que SectionPlan/PhrasePlan/GrooveMap
se guardan con el proyecto y que los generadores reciben el plan.
Ejecuta golden_generate.py, tests y lint. Actualiza docs/progress/MUSIC_BASELINE.md
y documenta el cierre en docs/progress/PR_01.md.
No implementes todavia la capa de model_backends.
```

## Prompt PR-02

```text
Implementa PR-02 - Epic 16.5 AI Model Backend Contract.

Crea packages/model_backends con:
- base.py
- registry.py
- config.py
- errors.py
- artifact.py
- symbolic/mock_backend.py
- symbolic/midigpt_backend.py
- symbolic/text2midi_backend.py

Crea configs/ai_models.yaml.
Anade GET /v1/ai/models.
La app debe arrancar aunque MIDI-GPT/Text2MIDI no esten instalados.
Incluye tests de registry, config y mock backend.
No implementes generacion real de MIDI-GPT ni Text2MIDI.
```

## Prompt PR-03

```text
Implementa PR-03 - Epic 16.6 Artifact Quarantine + Takes.

Crea ArtifactStore, ArtifactImporter, ProjectMerger, ValidationGate y TakeManager.
Todo artifact generado por modelo debe guardarse primero en outputs/model_artifacts/raw.
Si se importa y valida, crear take pending.
Si falla, mover a rejected.
El proyecto activo no puede modificarse hasta aceptar una take.
Anade endpoints de listado/accept/reject de takes.
Incluye tests con MockSymbolicBackend.
```

## Prompt PR-04

```text
Implementa PR-04 - Epic 16.7 LLM Planner JSON.

El planner solo puede generar JSON validado por Pydantic para
SongPlan/SectionPlan/PhrasePlan/GrooveMap/RoleIntent/GenerationStrategy.
No puede generar notas, MIDI, MusicXML ni audio.
Si el JSON es invalido, reintenta una vez. Si vuelve a fallar, fallback rule-based.
Anade endpoint POST /v1/projects/{id}/ai/plan.
Incluye tests de JSON valido, JSON invalido y fallback.
```

## Prompt PR-05

```text
Implementa PR-05 - Epic 16.8 MIDI-GPT backend opcional.

Crea MidiGptBackend en packages/model_backends/model_backends/symbolic/midigpt_backend.py.
No importes midigpt en import-time.
Si la dependencia falta, devolver error controlado.
Soporta inicialmente infill_bars.
Exporta contexto temporal desde ArrangementProject.
Ejecuta backend o mock segun config.
Guarda artifact raw.
Importa, fusiona solo track/bars target, valida y crea take pending.
Anade endpoint POST /v1/projects/{id}/ai/infill.
```

## Prompt PR-06

```text
Implementa PR-06 - Epic 16.9 Text2MIDI sketch backend opcional.

Crea Text2MidiBackend como backend experimental.
Usalo solo para generate_full_sketch.
El resultado se importa como sketch y no como proyecto final.
Anade endpoint POST /v1/ai/text-to-midi-sketch.
Clasifica pistas si existe TrackRoleClassifier.
Si no hay confianza, marcar sketch_uncertain.
Incluye tests con mock.
```

## Prompt PR-07

```text
Cuando PR-02 a PR-06 esten completados, implementa PR-07 - Epic 17
GrooveMap / PerformanceMap 2.0.
Asegurate de que PerformanceMap tambien normaliza material aceptado desde modelos
IA. No apliques humanizacion doble si el artifact ya viene expresivo.
Anade performance_source = rule_based | imported_model | normalized_model.
```
