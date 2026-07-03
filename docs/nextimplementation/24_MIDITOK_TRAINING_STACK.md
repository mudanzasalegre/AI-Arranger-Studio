# PR-24 — MidiTok / training stack

## Objetivo

Preparar la tokenización real y el pipeline de entrenamiento futuro para modelos propios por rol.

## Instalación

```bash
python -m pip install -r requirements-training-ai.txt
```

## Roles objetivo

```text
melody
walking_bass
piano_comping
horn_responses
drums
```

## Dataset correcto

El dataset de entrenamiento debe salir de:

1. Librerías privadas/locales importadas por el usuario.
2. Datasets con licencia clara.
3. Corpus sintético generado por el motor propio.
4. Patrones extraídos y validados por `DatasetProfiler`.

No entrenar modelos comerciales con:

```text
- Real Book MIDI descargado sin licencia clara;
- transcripciones privadas redistribuidas;
- outputs de modelos externos sin permiso;
- datasets marcados research_only/non_commercial.
```

## Estructura de salida

```text
data/processed/tokenized/
  melody/
  walking_bass/
  piano_comping/
  horn_responses/
  drums/
  manifests/
    tokenization_summary.json
    license_report.json
```

## Implementación

El repo ya trae `MidiTokBridgeTokenizer` como puente. En PR-24 se añade dependencia real `miditok` y se crea un adaptador que no rompa si MidiTok no está instalado.

Crear:

```text
packages/training/training/tokenizers/miditok_real.py
scripts/models/smoke_miditok.py
```

`miditok_real.py` debe:

- Importar `miditok` solo dentro de funciones.
- Leer MIDI/MusicXML convertido a MIDI.
- Tokenizar por rol.
- Guardar `tokenizer.json` y config.
- Reconstruir un MIDI de prueba.

## Smoke test

```bash
python scripts/models/smoke_miditok.py
```

Debe:

1. Crear/usar un MIDI fixture.
2. Tokenizar.
3. Reconstruir.
4. Escribir resumen JSON.
5. Reportar pérdida de información aceptable.

## Acceptance criteria

```text
- MidiTok instalado de forma opcional.
- Tokenización por rol funciona.
- `tokenization_summary.json` se genera.
- Cada segmento conserva metadata: role, style, chord_context, source_file_id, license.
- Los datasets sin permiso no entran a train split.
```
