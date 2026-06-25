# 04 — Objetivos de implementación

Este documento sustituye al plan PR por PR. Cada objetivo es un bloque completo que Codex debe implementar y cerrar con tests.

---

## Objetivo 0 — Inicializar repo ejecutable

### Meta

Crear monorepo mínimo con backend, frontend, paquete core, tests, lint y comandos comunes.

### Entregables

- estructura `apps/api`, `apps/web`, `packages/arranger_core`, `packages/dataset_tools`, `configs`, `examples`, `outputs`
- `Makefile`
- `docker-compose.yml`
- `pyproject.toml` o configuración equivalente
- API `/health`
- página web inicial
- CI opcional

### Acceptance criteria

- `make setup` instala dependencias Python básicas.
- `make test` pasa.
- `make lint` pasa.
- `make api` arranca FastAPI.
- `make web` arranca Next.js o deja instrucciones claras.

---

## Objetivo 1 — Modelo simbólico `ArrangementProject`

### Meta

Implementar el formato maestro interno.

### Entregables

- `GenerationSpec`
- `ArrangementProject`
- `Track`
- `Bar`
- `NoteEvent`
- `RestEvent`
- `ChordSymbol`
- `Section`
- serialización JSON
- carga JSON
- versionado de schema

### Acceptance criteria

- crear proyecto vacío
- crear proyecto con 4 compases y 2 pistas
- serializar/cargar sin pérdida crítica
- validar duración de compás

---

## Objetivo 2 — Conocimiento musical base

### Meta

Cargar catálogos YAML de instrumentos, acordes, escalas, progresiones, estilos y patrones.

### Entregables

- loader de configs
- `InstrumentCatalog`
- `ChordParser`
- `ScaleCatalog`
- `ProgressionLibrary`
- `StyleProfile`
- `PatternLibrary`

### Acceptance criteria

- parsear acordes jazz complejos: `F#m7b5`, `B7alt`, `Ebmaj7#11`, `G13b9`, `CmMaj9`, `D7#5#9/Ab`
- calcular notas base y tensiones
- cargar estilos jazz
- cargar plantillas de progresión
- tests de transposición de instrumentos

---

## Objetivo 3 — Exportación MIDI/MusicXML/PDF

### Meta

Exportar proyectos reales a formatos útiles.

### Entregables

- MIDI full arrangement
- MIDI por pista
- MusicXML full score
- PDF full score vía MuseScore CLI
- PDF de particellas
- export manifest

### Acceptance criteria

- generar `outputs/demo/full_arrangement.mid`
- generar `outputs/demo/full_score.musicxml`
- si MuseScore está instalado, generar PDFs
- abrir MusicXML en MuseScore sin errores graves
- tests de pistas MIDI separadas

---

## Objetivo 4 — Text prompt → GenerationSpec

### Meta

Crear un text-to-MIDI realista sin IA pesada mediante parser determinista enriquecido.

### Entregables

- prompt compiler con diccionarios bilingües ES/EN
- extracción de estilo, tonalidad, tempo, forma, ensemble, densidad, mood
- fallback inteligente a defaults
- API/CLI para compilar prompt

### Acceptance criteria

Prompt:

```text
hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombón, piano, contrabajo y batería
```

Debe producir `GenerationSpec` con:

- style hard_bop
- key C minor
- tempo 132
- form minor_blues_12
- ensemble jazz_sextet
- instruments correctos

---

## Objetivo 5 — Harmony/Form Engine

### Meta

Generar formas y progresiones ricas tipo jazz sin copiar standards.

### Entregables

- generador blues 12
- generador minor blues
- generador AABA
- generador rhythm-changes-like original
- generador modal vamp
- generador ballad
- variaciones: tritone subs, backdoor, secondary dominants, passing diminished, turnarounds
- control de complejidad

### Acceptance criteria

- crear chord grid de 12/16/32 compases
- reproducible por seed
- cifrado exportado a MusicXML
- no usar nombres ni melodías de standards

---

## Objetivo 6 — Lead sheet generator

### Meta

Generar melodía principal + acordes + forma.

### Entregables

- motivo inicial
- variación motívica
- frases 2/4 compases
- cadencias
- respiración
- articulaciones básicas

### Acceptance criteria

- lead sheet de 12-bar blues
- lead sheet AABA 32 compases
- melodía dentro de rango configurable
- MusicXML/PDF correcto

---

## Objetivo 7 — Generadores rule-based por rol

### Meta

Crear arreglo completo jazz sin IA.

### Entregables

- `DrumsGenerator`
- `WalkingBassGenerator`
- `PianoCompingGenerator`
- `MelodyGenerator`
- `HornResponseGenerator`
- `ShoutChorusGenerator`
- `Humanizer`

### Acceptance criteria

- arreglo jazz trio
- arreglo jazz quartet
- arreglo jazz quintet/sextet
- MIDI por pistas
- vientos con respuestas
- batería con fills
- piano rootless básico
- bajo caminante coherente

---

## Objetivo 8 — Validadores musicales

### Meta

Evitar outputs malos.

### Entregables

- validador de compases
- validador de rango instrumental
- validador de transposición
- validador armónico nota-acorde
- validador de respiración/fraseo
- validador de voicings
- validador de exportación
- reporte JSON/HTML

### Acceptance criteria

- errores graves bloquean export profesional
- warnings se reportan
- reporte por pista y compás
- tests con casos rotos

---

## Objetivo 9 — Dataset learning mode sin IA

### Meta

Permitir que el usuario añada librerías MIDI/MusicXML y el sistema aprenda patrones reutilizables sin entrenar todavía una red neuronal.

### Entregables

- manifiesto de datasets/licencias
- ingesta de carpetas
- normalización
- deduplicación
- etiquetado estilo/calidad/rol
- extracción de progresiones
- extracción de grooves
- extracción de voicings
- extracción de bass cells
- extracción de motivos
- pattern index consultable por generadores

### Acceptance criteria

- añadir 10 MIDIs de prueba
- extraer patrones
- usar patrones en generación rule-based
- respetar `usable_for_training` y `usable_for_pattern_extraction`

---

## Objetivo 10 — API completa

### Meta

Exponer generación, exportación, validación y dataset mode.

### Entregables

Endpoints:

- `GET /health`
- `POST /v1/prompts/compile`
- `POST /v1/projects/generate`
- `GET /v1/projects/{id}`
- `POST /v1/projects/{id}/export`
- `GET /v1/projects/{id}/validation`
- `POST /v1/projects/{id}/regenerate`
- `POST /v1/datasets/import`
- `GET /v1/datasets`
- `GET /v1/patterns/search`

### Acceptance criteria

- API OpenAPI generada
- tests de endpoints
- generación completa desde endpoint

---

## Objetivo 11 — Interfaz web usable

### Meta

Crear aplicación tipo estudio simbólico.

### Pantallas

- home
- new project
- project detail
- score viewer
- track mixer simple
- chord/form editor
- validation report
- dataset library
- export panel

### Acceptance criteria

- escribir prompt y generar proyecto
- ver partitura
- escuchar preview/MIDI
- descargar ZIP
- ver errores/warnings

---

## Objetivo 12 — Presets potentes de primera pasada

### Meta

Que la app no parezca demo pobre.

### Presets obligatorios

- hard bop minor blues sextet
- bebop blues quintet
- swing AABA quartet
- jazz ballad trio/quartet
- modal jazz quintet
- bossa nova quartet
- jazz waltz trio
- funk jazz straight-eighth

### Acceptance criteria

- cada preset genera proyecto completo
- cada preset tiene estilo reconocible
- cada preset exporta MIDI/MusicXML/PDF
- evaluation pack con 20 prompts

---

## Objetivo 13 — Packaging y smoke tests

### Meta

Que el proyecto sea fácil de arrancar y verificar.

### Entregables

- `make demo-jazz`
- `make export-demo`
- `make validate-demo`
- `make zip-demo`
- Docker opcional
- documentación de instalación

### Acceptance criteria

- una persona nueva puede generar un arreglo demo en menos de 10 minutos si tiene dependencias instaladas

---

## Objetivo 14 — Contrato para IA futura

### Meta

Dejar preparado el cambio a modelos.

### Entregables

- interfaces `ModelBackend`
- interfaces `RoleModelGenerator`
- dataset splits train/val/test
- tokenization placeholder
- feature store
- evaluación de similitud/memorización
- adapters vacíos para MidiTok/modelos externos

### Acceptance criteria

- se puede sustituir `WalkingBassGenerator` por `AIWalkingBassGenerator` sin tocar API/web
- todo dataset tiene licencia/uso controlado
- documentación de entrenamiento futuro
