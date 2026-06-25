# 02 — Arquitectura técnica

## Diagrama lógico

```text
User prompt
   ↓
Prompt Compiler
   ↓
GenerationSpec
   ↓
Music Planner
   ↓
Harmony/Form Engine
   ↓
Role Generators
   ├─ DrumsGenerator
   ├─ BassGenerator
   ├─ PianoCompingGenerator
   ├─ MelodyGenerator
   ├─ HornSectionGenerator
   └─ Humanizer
   ↓
ArrangementProject
   ↓
Validators
   ↓
Exporters
   ├─ MIDI
   ├─ MusicXML
   ├─ PDF/parts via MuseScore CLI
   ├─ audio preview
   └─ ZIP package
```

## Monorepo

```text
apps/
  api/              FastAPI
  web/              Next.js
packages/
  arranger_core/    dominio musical puro
  dataset_tools/    ingesta, normalización, extracción de patrones
  render_engine/    MuseScore/FluidSynth/preview
  midi_models/      vacío al inicio; contrato de IA futura
configs/            conocimiento musical editable
examples/           requests y proyectos ejemplo
data/               raw/processed/gold/manifests
outputs/            salidas generadas
scripts/            utilidades de CLI
```

## Backend

FastAPI se usa solo como capa de entrada/salida: endpoints, validación request/response, jobs, persistencia y archivos. La lógica musical vive en `arranger_core`.

## Frontend

Next.js se usa para:

- crear proyecto
- ver score
- escuchar MIDI/audio preview
- editar prompt, acordes y forma
- exportar paquete

OpenSheetMusicDisplay renderiza MusicXML en navegador.

## Núcleo musical

`arranger_core` debe poder usarse sin servidor:

```python
from arranger_core import generate_arrangement, export_project

project = generate_arrangement(spec)
export_project(project, output_dir="outputs/demo")
```

## Contratos principales

### GenerationSpec

Entrada normalizada del usuario:

- prompt original
- style
- substyle
- tempo
- key
- meter
- form
- ensemble
- duration_bars
- density
- complexity
- seed
- constraints

### ArrangementProject

Proyecto musical completo:

- metadata
- form
- harmony
- tracks
- notation events
- performance data
- validation report
- export manifest

### PatternLibrary

Patrones extraídos de configs y datasets:

- chord progressions
- voicings
- bass cells
- drum grooves
- comping rhythms
- melodic motifs
- horn responses

## Interfaces de generador

Todo generador debe seguir este contrato:

```python
class RoleGenerator(Protocol):
    role: str
    def generate(self, context: GenerationContext) -> TrackData: ...
```

Luego, un modelo de IA podrá sustituir un generador sin cambiar API ni frontend.

## Persistencia

Versión local simple:

- metadata en SQLite/PostgreSQL
- archivos en `outputs/`

Versión futura:

- PostgreSQL
- object storage compatible S3
- job queue para render/export largo

## Configuración musical

Todo lo editable por músico debe ir en YAML:

- `configs/instruments.yaml`
- `configs/chord_dictionary.yaml`
- `configs/jazz_progressions.yaml`
- `configs/style_profiles/jazz/*.yaml`
- `configs/patterns/*.yaml`

## Seguridad musical/legal

- No entrenar con archivos sin permiso explícito.
- No incluir canciones comerciales en configs.
- No crear plantillas que reproduzcan standards identificables compás por compás.
- Usar progresiones genéricas, variaciones y gramáticas.
