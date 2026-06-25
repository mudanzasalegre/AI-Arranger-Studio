# Objetivo 6 - Lead sheet generator

Fecha: 2026-06-24

## Alcance implementado

- Generador `LeadSheetGenerator` en `arranger_core.lead_sheet`.
- Funcion publica `generate_lead_sheet_project(spec, ...)`.
- Integracion con el `HarmonyFormEngine` del objetivo 5 para conservar forma y chord grid.
- Pista unica `Lead Sheet` con rol `melody`, acordes globales y compases completos con notas/silencios.
- Motivo inicial documentado en metadata del proyecto.
- Variacion motivica por:
  - desplazamiento ritmico,
  - secuencia de notas guia,
  - respuesta cadencial.
- Fraseo de 4 compases para blues/AABA, con respiraciones mediante silencios al final de frases y subfrases.
- Cadencias en barras de cierre de frase.
- Articulaciones basicas en `NoteEvent.articulations`: `tenuto`, `staccato`, `accent`.
- Dinamicas basicas en `NoteEvent.dynamic`.
- Rango melodico configurable desde `GenerationSpec.constraints`:
  - `melody_range`,
  - `lead_sheet_range`,
  - `lead_range`.
- Exportador MusicXML actualizado para escribir articulaciones y dinamicas desde `NoteEvent`.

## Caso de aceptacion

- `minor_blues_12` genera lead sheet de 12 compases.
- `aaba_32` genera lead sheet de 32 compases.
- La melodia respeta el rango configurable en tests (`C4-Bb5`, `D4-A4`, `C4-C5`).
- El proyecto valida duraciones de compas sin huecos ni sobrecargas.
- MusicXML contiene:
  - cifrado armonico `<harmony>`,
  - nombre de parte `Lead Sheet`,
  - articulaciones,
  - dinamicas.
- Smoke exporta MusicXML y PDF cuando MuseScore CLI esta disponible.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 45 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/lead_sheet_smoke.py`: OK.

## Smoke de generacion/exportacion

`scripts/lead_sheet_smoke.py` genero dos paquetes:

- `outputs/obj6_lead_sheet_demo/minor_blues_12`
- `outputs/obj6_lead_sheet_demo/aaba_32`

Resultado observado:

```text
Generated minor_blues_12: 12 bars, 16 chord symbols, pdf_status=created
Generated aaba_32: 32 bars, 55 chord symbols, pdf_status=created
```

Cada paquete contiene `arrangement_project.json`, `generation_spec.json`, `full_arrangement.mid`,
`full_score.musicxml`, `full_score.pdf`, `export_manifest.json`, `validation_report.json`,
`midi_tracks/` y `parts_pdf/`.

## Notas

- No se han generado nombres ni melodias de standards.
- El generador produce una melodia monofonica de lead sheet; los arreglos por rol quedan para el objetivo 7.
- Las aproximaciones cromaticas se limitan a notas de paso hacia objetivos de cadencia o notas guia.
