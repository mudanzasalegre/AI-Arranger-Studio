# Objetivo 5 - Harmony/Form Engine

Fecha: 2026-06-24

## Alcance implementado

- Motor `HarmonyFormEngine` en `arranger_core.harmony_engine`.
- API publica:
  - `generate_harmony_plan(spec)`,
  - `generate_harmony_project(spec, ...)`,
  - `degree_to_chord_symbol(...)`,
  - `parse_key(...)`.
- Generadores de forma:
  - `jazz_blues_12`,
  - `minor_blues_12`,
  - `aaba_32`,
  - `rhythm_changes_like_32`,
  - `modal_vamp` y `modal_vamp_16`,
  - `ballad_aaba_32`,
  - `sixteen_bar`.
- Variaciones controladas por `complexity` y `seed`:
  - turnarounds,
  - passing diminished,
  - secondary dominants,
  - tritone substitutions,
  - backdoor cadence.
- Generacion de `ArrangementProject` con pista `lead_sheet` de silencios para transportar el cifrado al exportador.
- Exportador MusicXML ajustado para conservar `track.name` como nombre de parte cuando existe.
- Smoke script `scripts/harmony_smoke.py` para compilar prompt, generar armonia y exportar MusicXML.

## Caso de aceptacion

- `minor_blues_12` genera 12 compases.
- `modal_vamp_16` y `sixteen_bar` generan 16 compases.
- `aaba_32`, `ballad_aaba_32` y `rhythm_changes_like_32` generan 32 compases.
- La misma seed produce el mismo chord grid y otra seed puede alterar variaciones.
- El MusicXML exportado contiene elementos `<harmony>` y tipos de acorde como `minor-seventh`.
- No se generan melodias ni nombres de standards; el objetivo queda limitado a forma y cifrado.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 42 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/harmony_smoke.py`: OK.

## Smoke de generacion/exportacion

`scripts/harmony_smoke.py` genera:

- `outputs/obj5_harmony_demo/arrangement_project.json`
- `outputs/obj5_harmony_demo/generation_spec.json`
- `outputs/obj5_harmony_demo/full_arrangement.mid`
- `outputs/obj5_harmony_demo/full_score.musicxml`
- `outputs/obj5_harmony_demo/export_manifest.json`

Resultado observado:

```text
Generated obj5 harmony smoke: 12 bars, 17 chord symbols, outputs/obj5_harmony_demo/full_score.musicxml
```

## Notas

- El motor usa plantillas originales por grados y transposicion determinista, no melodias.
- `complexity < 0.35` no aplica variaciones; niveles superiores van activando mas sustituciones.
- El control de aleatoriedad esta encapsulado en `random.Random(spec.seed)`.
