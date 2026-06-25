# 05 — Dataset learning mode sin IA

## Objetivo

Añadir una “librería de aprendizaje” desde el principio. En la primera versión no se entrena una red neuronal; se extraen patrones, estadísticas y ejemplos reutilizables para enriquecer la generación rule-based.

## Flujo

```text
MIDI/MusicXML library
  ↓
manifest + license check
  ↓
import
  ↓
normalize
  ↓
deduplicate
  ↓
segment by bars/tracks
  ↓
annotate roles/styles
  ↓
extract patterns
  ↓
PatternLibrary
  ↓
rule-based generators use learned patterns
```

## Manifiesto obligatorio

Cada archivo debe tener:

- path
- source
- license
- copyright_notes
- usable_for_training
- usable_for_pattern_extraction
- style
- quality
- tags
- imported_at
- hash

Nunca entrenar ni extraer patrones de archivos marcados como no permitidos.

## Extracciones útiles

### Progresiones

Extraer secuencias de acordes por 2, 4, 8, 12, 16, 32 compases.

### Walking bass cells

Guardar células por:

- acorde actual
- acorde siguiente
- grado inicial
- grado final
- patrón rítmico
- contorno

### Piano voicings

Guardar:

- familia de acorde
- notas relativas
- inversión
- registro
- densidad
- mano izquierda/derecha si se puede inferir

### Drum grooves

Guardar:

- estilo
- compás
- ride pattern
- hihat
- snare comping
- kick
- fills

### Melodic motifs

Guardar:

- grados relativos
- ritmo
- relación con acorde
- contorno
- posición en frase

### Horn responses

Guardar:

- número de voces
- ritmo
- notas relativas al acorde
- spacing
- rango

## Calidad de dataset

Usar calificación 1-5:

- 1: corrupto/pobre/no usar
- 2: usable solo para análisis general
- 3: usable para patrones simples
- 4: buen material
- 5: dataset oro/manualmente revisado

## No IA todavía

Este modo “aprende” de forma estadística/retrieval:

- cuenta patrones frecuentes
- extrae variaciones
- indexa por estilo/rol/contexto
- elige patrones por seed, peso, contexto armónico y calidad

## Preparación para IA futura

Todo segmento debe poder exportarse luego como ejemplo de entrenamiento:

```json
{
  "style": "hard_bop",
  "role": "walking_bass",
  "instrument": "double_bass",
  "key": "C minor",
  "meter": "4/4",
  "tempo": 132,
  "chord_context": ["Cm7", "F7", "Bbmaj7"],
  "previous_tokens": [],
  "target_tokens": [],
  "source_file_id": "...",
  "license": "..."
}
```
