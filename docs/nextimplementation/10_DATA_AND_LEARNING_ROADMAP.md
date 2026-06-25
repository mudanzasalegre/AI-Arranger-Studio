# 10 — Datos, aprendizaje y preparación para modelos propios

## Objetivo

Alinear el modo aprendizaje actual con la futura integración de modelos propios, sin depender de modelos externos para todo.

## Capas de datos

```text
raw datasets
  -> manifest/licencia
  -> profiler
  -> role classifier
  -> segmentation
  -> pattern extraction
  -> retrieval index
  -> tokenization datasets
  -> model training future
```

## Uso de datasets tipo Real Book

Separar siempre:

```text
- corpus armónico estilo fakebook / realbook-like;
- lead sheets con licencia clara;
- solos jazz legales;
- librería privada local del usuario;
- corpus sintético propio.
```

Regla:

```text
si no sabemos licencia/procedencia -> no entra en training comercial
```

Pero puede entrar en:

```text
private_local_learning: true
not_redistributable: true
```

## PatternLibrary

El modo aprendizaje inicial debe extraer:

- progresiones;
- células de walking bass;
- voicings de piano;
- patrones de comping;
- grooves/fills de batería;
- motivos melódicos;
- respuestas de viento;
- densidad por estilo;
- rangos por instrumento.

## Tokenización futura

Añadir preparación para:

```text
packages/training/
  tokenizers/
  datasets/
  train_melody.py
  train_bass.py
  train_piano.py
  train_horns.py
  train_drums.py
```

Primero no entrenar. Solo generar datasets limpios.

## MidiTok

Uso futuro:

- convertir MIDI/ABC a tokens;
- probar REMI/TSD/otros tokenizers;
- entrenar BPE/Unigram/WordPiece;
- evaluar reconstrucción.

## Dataset manifests

Cada fuente debe tener:

```yaml
dataset_id: string
source: string
license: string
license_confidence: high|medium|low
commercial_training: allowed|forbidden|review_required
local_learning_only: true|false
contains_melody: true|false
contains_chords: true|false
contains_arrangement: true|false
roles:
  - melody
  - harmony
  - bass
  - drums
  - piano
  - horns
```

## No-memorization

Cada patrón usado en generación debe guardar:

```text
pattern_id
source_dataset
source_file_hash
transformations_applied
similarity_to_source
```

Si la similitud supera umbral, bloquear o transformar más.

## Acceptance criteria

- Dataset profiler explica qué roles cubre cada corpus.
- Role classifier asigna confianza por pista.
- Pattern extraction alimenta RetrievalModel.
- Token export produce dataset listo para modelos futuros.
- Licencias se respetan en gates.
