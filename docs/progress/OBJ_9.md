# Objetivo 9 - Dataset learning mode sin IA

Fecha: 2026-06-24

## Alcance implementado

- Paquete `dataset_tools` con API publica para:
  - `create_manifest(...)`,
  - `import_dataset(...)`,
  - `extract_patterns(...)`,
  - `load_pattern_index(...)`,
  - `sha256_file(...)`.
- Modelos Pydantic para:
  - `DatasetManifest` y `DatasetManifestEntry`,
  - `NormalizedFile`,
  - `ExtractedPattern`,
  - `PatternIndex`,
  - `ImportSummary`.
- Manifiesto JSON obligatorio con:
  - path,
  - source,
  - license,
  - copyright_notes,
  - usable_for_training,
  - usable_for_pattern_extraction,
  - style,
  - quality,
  - tags,
  - imported_at,
  - hash.
- Ingesta de carpetas para `.mid`, `.midi`, `.musicxml` y `.xml`.
- Normalizacion por copia a `normalized/` con IDs estables por hash.
- Deduplicacion por SHA-256.
- Etiquetado de estilo, calidad, uso legal, tags y roles inferidos.
- Extraccion de patrones:
  - progresiones desde MusicXML,
  - grooves de bateria desde MIDI,
  - voicings de piano,
  - walking bass cells,
  - motivos melodicos,
  - respuestas de vientos.
- `PatternIndex` consultable por categoria, rol, estilo, calidad, tags, permiso de training y permiso de extraccion.
- Integracion opt-in con los generadores rule-based mediante:
  - `GenerationSpec.constraints["pattern_index_path"]`,
  - `GenerationSpec.constraints["pattern_index"]`,
  - `GenerationSpec.constraints["pattern_min_quality"]`.

## Integracion con generadores

- `RuleBasedArranger` carga el indice aprendido al crear el `GenerationContext`.
- `DrumsGenerator` puede reutilizar grooves aprendidos si cubren todo el compas.
- `WalkingBassGenerator` puede transformar bass cells aprendidas al acorde actual.
- `PianoCompingGenerator` puede adaptar voicings aprendidos como rootless voicings.
- La generacion por defecto no cambia si no se proporciona un indice.
- El proyecto generado marca `metadata["pattern_index_used"]`.
- Las pistas/eventos que usan patrones incluyen `learned_pattern_id`.

## Caso de aceptacion

- Se generan 10 MIDIs sinteticos de prueba en el smoke OBJ9.
- La importacion extrae patrones de todas las categorias previstas.
- La generacion rule-based usa el `pattern_index.json` importado.
- `usable_for_pattern_extraction=False` evita extraer patrones.
- `quality < 3` evita extraer patrones.
- `usable_for_training=False` se conserva en patrones extraidos, pero no impide extraccion si esta permitida.
- La deduplicacion por hash queda cubierta por tests.

## Verificacion ejecutada

- `python -m pytest -q`: OK, 61 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python -m ruff check apps packages scripts tests`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/bootstrap_check.py`: OK.
- `python scripts/dataset_learning_smoke.py`: OK.

## Smoke de dataset/generacion/exportacion

`scripts/dataset_learning_smoke.py` genera:

- `outputs/obj9_dataset_demo/source/`
- `outputs/obj9_dataset_demo/dataset_manifest.json`
- `outputs/obj9_dataset_demo/imported/dataset_manifest.json`
- `outputs/obj9_dataset_demo/imported/normalized_files.json`
- `outputs/obj9_dataset_demo/imported/pattern_index.json`
- `outputs/obj9_dataset_demo/imported/import_summary.json`
- `outputs/obj9_dataset_demo/generated/full_arrangement.mid`
- `outputs/obj9_dataset_demo/generated/full_score.musicxml`
- `outputs/obj9_dataset_demo/generated/export_manifest.json`
- `outputs/obj9_dataset_demo/smoke_summary.json`

Resultado observado:

```json
{
  "source_midis": 10,
  "imported_files": 11,
  "skipped_for_license": 1,
  "skipped_for_quality": 1,
  "extracted_patterns": 123,
  "pattern_counts": {
    "drum_grooves": 8,
    "horn_responses": 12,
    "melodic_motifs": 16,
    "piano_voicings": 64,
    "progressions": 7,
    "walking_bass_cells": 16
  },
  "pattern_index_used": true,
  "exported_files": 10
}
```

## Notas

- Este modo no entrena modelos: solo extrae, normaliza, indexa y reutiliza patrones.
- El uso de patrones aprendidos es deliberadamente conservador: si un groove aprendido no cubre el compas, el generador vuelve al patron rule-based interno.
- `PatternIndex.search(...)` permite filtrar por `usable_for_training` y `usable_for_pattern_extraction` para preparar flujos futuros de IA sin mezclar permisos.
