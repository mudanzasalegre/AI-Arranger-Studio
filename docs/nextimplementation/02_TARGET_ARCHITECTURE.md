# 02 — Arquitectura objetivo con LLM simbólicos

## Diagrama general

```text
User Prompt
  ↓
PromptCompiler / LLM Planner JSON
  ↓
SongPlan
  ↓
SectionPlan + PhrasePlan + GrooveMap
  ↓
RuleBasedArrangementEngine
  ↓
RoleIntentBuilder
  ↓
SymbolicModelBackends
  ├─ MIDI-GPT: infill_bars / generate_track / continue_section
  ├─ Text2MIDI: generate_full_sketch
  ├─ MockSymbolicBackend: tests
  └─ FutureCustomRoleModels
  ↓
ArtifactQuarantine
  ↓
MIDI/Token Importer
  ↓
ProjectMerger
  ↓
ValidationGate
  ↓
TakeManager
  ↓
ExportEngine
  ├─ MIDI multitrack
  ├─ MIDI por pistas
  ├─ MusicXML
  ├─ PDF / particellas
  └─ ZIP DAW-ready
```

## Capas

### 1. Planning layer

Responsable de convertir intención en estructura musical:

- estilo;
- forma;
- tonalidad;
- tempo;
- secciones;
- densidad;
- energía;
- instrumentos;
- estrategia de generación.

El LLM aquí solo devuelve JSON.

### 2. Rule-based base arrangement

Genera una versión base musicalmente consistente:

- armonía;
- melodía base;
- bajo;
- batería;
- piano;
- vientos;
- performance map.

Esta capa debe seguir funcionando aunque no haya modelos instalados.

### 3. Symbolic model layer

Modelos externos opcionales:

- regeneran una pista;
- rellenan compases;
- continúan secciones;
- generan bocetos completos;
- crean variaciones.

No exportan resultado final.

### 4. Quarantine/import/merge layer

Protege el proyecto:

- guarda artifact raw;
- importa MIDI/tokens;
- extrae solo material autorizado;
- fusiona sobre copia temporal;
- valida.

### 5. Take layer

Cada candidato queda guardado como take:

```text
project_id
base_take_id
candidate_take_id
backend_id
task
track_id
bars
instruction
seed
validation_report
status: pending|accepted|rejected
```

### 6. Export layer

Solo exporta desde proyectos/takes aceptados.

## Packages nuevos

```text
packages/model_backends/
packages/arranger_core/arranger_core/ai/
packages/arranger_core/arranger_core/takes/
packages/arranger_core/arranger_core/merge/
packages/arranger_core/arranger_core/planning/llm/
apps/model_worker/                 # recomendado, puede ir después
```

## Dependencias obligatorias vs opcionales

Obligatorias:

```text
pydantic
pyyaml
pretty_midi / mido / miditoolkit según lo ya usado
music21 si ya está integrado
```

Opcionales:

```text
midigpt[inference]
transformers
torch
huggingface_hub
miditok
```

Regla: el API principal debe arrancar sin dependencias opcionales.
