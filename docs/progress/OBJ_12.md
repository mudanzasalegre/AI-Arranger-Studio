# Objetivo 12 - Presets potentes de primera pasada

Fecha: 2026-06-24

## Alcance implementado

- Ocho presets obligatorios en `configs/generation_presets/`.
- Evaluation pack de 20 prompts en `configs/evaluation_pack.yaml`.
- Modulo `arranger_core.presets` con:
  - `GenerationPreset`,
  - `EvaluationPrompt`,
  - `PresetLibrary`.
- Export publico desde `arranger_core`.
- Perfiles de estilo nuevos:
  - `bebop`,
  - `swing`,
  - `jazz_waltz`,
  - `funk_jazz`.
- Harmony engine ampliado con:
  - `bossa_32`,
  - `jazz_waltz_32`.
- Generadores por rol ampliados para estilos:
  - bossa,
  - funk straight-eighth,
  - jazz waltz 3/4,
  - ballad ligera.
- Interfaz web ampliada con botones de presets rapidos en `New project`.
- Smoke end-to-end en `scripts/preset_smoke.py`.

## Presets implementados

- `jazz_hard_bop_minor_blues_sextet`
- `jazz_bebop_blues_quintet`
- `jazz_swing_aaba_quartet`
- `jazz_ballad_quartet`
- `jazz_modal_quintet`
- `jazz_bossa_nova_quartet`
- `jazz_waltz_trio`
- `jazz_funk_straight_eighth_quintet`

## Caso de aceptacion

- Cada preset genera un `ArrangementProject` completo.
- Cada preset tiene estilo reconocible mediante:
  - `style`,
  - `form`,
  - `meter`,
  - `constraints.required_features`,
  - groove de bateria por estilo cuando aplica.
- Cada preset exporta:
  - MIDI full,
  - MIDI por pista,
  - MusicXML,
  - PDF full score,
  - PDF por partes.
- Evaluation pack contiene 20 prompts.

## Verificacion ejecutada

- `python -m pytest -q`: OK, 69 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python -m ruff check apps packages scripts tests`: OK.
- `npm --prefix apps/web run lint`: OK.
- `npm --prefix apps/web run build`: OK.
- `python scripts/bootstrap_check.py`: OK.
- `python scripts/api_smoke.py`: OK. Muestra el mismo warning externo de Starlette/FastAPI.
- `python scripts/preset_smoke.py`: OK.

## Smoke de presets

`scripts/preset_smoke.py` genera:

- `outputs/obj12_preset_pack/<preset_id>/arrangement_project.json`
- `outputs/obj12_preset_pack/<preset_id>/generation_spec.json`
- `outputs/obj12_preset_pack/<preset_id>/full_arrangement.mid`
- `outputs/obj12_preset_pack/<preset_id>/full_score.musicxml`
- `outputs/obj12_preset_pack/<preset_id>/full_score.pdf`
- `outputs/obj12_preset_pack/<preset_id>/midi_tracks/`
- `outputs/obj12_preset_pack/<preset_id>/parts_pdf/`
- `outputs/obj12_preset_pack/preset_smoke_summary.json`
- `outputs/obj12_preset_pack/evaluation_pack.json`

Resultado observado:

```json
{
  "preset_count": 8,
  "evaluation_prompt_count": 20,
  "musescore_cli": "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.EXE",
  "pdf_status": "created for every preset",
  "validation_status": "pass for every preset"
}
```

## Notas

- El smoke detecto MuseScore 4 y creo PDFs reales.
- Los presets evitan standards concretos: usan formas y gramaticas genericas.
- El editor web muestra presets rapidos, pero la fuente canonica esta en YAML y se valida desde Python.
