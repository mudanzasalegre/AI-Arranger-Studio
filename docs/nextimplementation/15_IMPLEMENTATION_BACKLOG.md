# 15 - Implementation backlog PR-00 a PR-19

## Estado actual

```text
Epic 15 - Musical QA baseline
Epic 16 - SongPlan / SectionPlan / PhrasePlan / GrooveMap
```

No continuar con el antiguo Epic 17 hasta completar primero la capa LLM
simbolica segura.

## PR-00 - Sustituir docs/nextimplementation

Objetivo: reemplazar la carpeta antigua por esta documentacion.

Acciones:

- Descomprimir `nextimplementation_llm_integration_v2_1.zip` si existe.
- Sustituir `docs/nextimplementation` por la carpeta nueva.
- Revisar `README.md`.
- Confirmar que el plan actual reconoce Epic 16 como punto de partida.

Acceptance:

- `docs/nextimplementation` contiene `README.md` y documentos `00` a `19`.
- No quedan documentos antiguos duplicados que apunten al backlog viejo.
- La siguiente unidad de trabajo es `PR-01`.

## PR-01 - Congelar Epic 15/16

Objetivo: verificar que la base actual sigue sana antes de meter IA.

Revisar:

- Epic 15 - Musical QA baseline.
- Epic 16 - SongPlan / SectionPlan / PhrasePlan / GrooveMap.

Tareas:

- Confirmar que SongPlan serializa/deserializa correctamente.
- Confirmar que SectionPlan y PhrasePlan estan guardados junto al proyecto.
- Confirmar que GrooveMap existe o esta preparado para continuar.
- Confirmar que los generadores actuales reciben el plan.
- Ejecutar golden tests actuales.
- Guardar `docs/progress/MUSIC_BASELINE.md`.

Acceptance:

- La app genera lo mismo que antes.
- Tests y smoke pasan.
- Tenemos una foto clara del estado previo a IA.

## PR-02 - Epic 16.5: AI Model Backend Contract

Objetivo: crear una capa comun para conectar modelos simbolicos sin acoplarlos
al core.

Crear:

```text
packages/model_backends/
  model_backends/
    base.py
    registry.py
    config.py
    errors.py
    artifact.py
    symbolic/
      mock_backend.py
      midigpt_backend.py
      text2midi_backend.py
configs/ai_models.yaml
```

Debe incluir:

- `ModelCapabilities`
- `ModelGenerationRequest`
- `ModelGenerationResult`
- `ModelArtifact`
- `MusicModelBackend`
- `ModelBackendRegistry`
- `MockSymbolicBackend`

Endpoint:

```text
GET /v1/ai/models
```

Acceptance:

- El backend mock aparece en `/v1/ai/models`.
- Las dependencias pesadas son opcionales.
- No se importa `torch`, `midigpt` ni `transformers` en import-time.
- Hay tests del registry y del mock backend.

## PR-03 - Epic 16.6: Artifact Quarantine + Takes

Objetivo: impedir que una salida IA modifique directamente el proyecto principal.

Crear:

```text
outputs/model_artifacts/
  raw/
  imported/
  rejected/
  validated/
```

Servicios:

- `ArtifactStore`
- `ArtifactImporter`
- `ProjectMerger`
- `ValidationGate`
- `TakeManager`

Endpoints:

```text
GET  /v1/projects/{id}/takes
POST /v1/projects/{id}/takes/{take_id}/accept
POST /v1/projects/{id}/takes/{take_id}/reject
```

Acceptance:

- Artifact valido crea una take pendiente.
- Artifact invalido se rechaza.
- El proyecto activo no cambia automaticamente.
- Hay tests de aceptacion, rechazo y rollback.

## PR-04 - Epic 16.7: LLM Planner JSON

Objetivo: permitir que un LLM convierta prompts en planes musicales, sin generar
notas.

Puede generar/modificar:

- `SongPlan`
- `SectionPlan`
- `PhrasePlan`
- `GrooveMap`
- `RoleIntent`
- `GenerationStrategy`

No puede generar:

- MIDI final.
- Notas definitivas.
- Audio.
- Exports.

Crear:

```text
packages/arranger_core/arranger_core/planning/llm_planner.py
packages/arranger_core/arranger_core/planning/plan_schema.py
packages/arranger_core/arranger_core/planning/plan_validator.py
apps/api/app/routes/ai_planner.py
```

Endpoint:

```text
POST /v1/projects/{id}/ai/plan
```

Acceptance:

- Prompt libre produce plan valido.
- JSON invalido no rompe la app.
- Hay fallback rule-based.
- El planner no genera notas.

## PR-05 - Epic 16.8: MIDI-GPT Backend opcional

Objetivo: integrar MIDI-GPT para generacion simbolica controlada.

Usos permitidos:

- `infill_bars`
- `generate_track`
- `continue_section`
- `generate_variation`

No usar todavia para:

- Generar cancion final completa.
- Sustituir Harmony Engine.
- Sustituir validadores.
- Exportar directamente.

Crear:

```text
packages/model_backends/model_backends/symbolic/midigpt_backend.py
```

Endpoint:

```text
POST /v1/projects/{id}/ai/infill
```

Acceptance:

- Si MIDI-GPT no esta instalado, error controlado.
- Con mock/fixture se crea take.
- Solo modifica la pista y compases solicitados.
- Si rompe rango/armonia/duracion, se rechaza.

## PR-06 - Epic 16.9: Text2MIDI Sketch Backend opcional

Objetivo: usar Text2MIDI solo para bocetos iniciales desde prompt.

Crear:

```text
packages/model_backends/model_backends/symbolic/text2midi_backend.py
```

Endpoint:

```text
POST /v1/ai/text-to-midi-sketch
```

Acceptance:

- Sketch importado como proyecto/take experimental.
- Si no detecta roles, queda marcado como `sketch_uncertain`.
- No contamina proyectos profesionales.

## PR-07 - Epic 17: GrooveMap / PerformanceMap 2.0

Objetivo: aplicar performance/humanizacion tanto a material rule-based como a
material IA aceptado.

Acceptance:

- El MIDI final suena mas humano.
- No rompe MusicXML ni validadores.
- No se humaniza dos veces material importado.

## PR-08 - Epic 18: Melody Engine 2.0

Objetivo: mejorar generacion melodica profesional.

Modos:

```text
rule_based
retrieval
ai_infill
```

Acceptance: la melodia respeta acorde, forma, registro, respiracion y fraseo.

## PR-09 - Epic 19: Bass Engine 2.0

Objetivo: bajo caminante solido.

Prioridad:

```text
rule-based
retrieval
estadistico
IA solo con validacion fuerte
```

Acceptance: el bajo sostiene la armonia y no genera lineas erraticas.

## PR-10 - Epic 20: Drums Engine 2.0

Objetivo: bateria mas musical desde `GrooveMap`, retrieval y `PerformanceMap`.

Acceptance: la bateria acompana la forma y no parece un patron plano repetido.

## PR-11 - Epic 21: Piano / Comping Engine 2.0

Objetivo: comping profesional con voicings, registro y densidad controlada.

Acceptance: el piano acompana sin embarrar el arreglo.

## PR-12 - Epic 22: Dataset Profiler / Role Classifier

Objetivo: preparar el modo aprendizaje.

Acceptance: una libreria MIDI importada queda clasificada y lista para retrieval
o futuros modelos.

## PR-13 - Epic 23: Retrieval Model

Objetivo: puente entre rule-based e IA neural.

Acceptance: los generadores pueden pedir patrones similares y variarlos sin
copiar literal.

## PR-14 - Epic 24: App Workflow 2.0

Objetivo: adaptar frontend/API a generacion profesional con IA controlada.

Acceptance: el usuario puede regenerar pista/compases concretos, revisar el
resultado y aceptarlo o rechazarlo.

## PR-15 - Epic 25: DAW-ready Export

Objetivo: exportar solo resultados aceptados y trazables.

Acceptance: el ZIP final es util en DAW/MuseScore y contiene `model_trace.json`.

## PR-16 - Epic 26: Release Quality Gate

Objetivo: impedir releases con material invalido.

Acceptance: no se crea release/export final si hay errores bloqueantes.

## PR-17 - Epic 27: Tokenization Dataset Export

Objetivo: preparar datasets para entrenar modelos propios.

Acceptance: dataset tokenizado y reproducible por rol.

## PR-18 - Epic 28: Baseline Statistical Role Models

Objetivo: crear modelos simples antes de neural grande.

Acceptance: hay modelos estadisticos comparables contra rule-based y retrieval.

## PR-19 - Epic 29: Custom Role Model Interface

Objetivo: preparar modelos propios por rol.

Acceptance: la app puede alternar `rule_based`, `retrieval`, `external_model` y
`custom_model` sin tocar exportadores ni API principal.
