# Objetivo 8 - Validadores musicales

Fecha: 2026-06-24

## Alcance implementado

- Modulo `arranger_core.validators`.
- API publica:
  - `validate_project(project, ...)`,
  - `validate_export_package(project, manifest, output_dir)`,
  - `merge_validation_reports(...)`,
  - `write_validation_json(report, path)`,
  - `write_validation_html(report, path)`,
  - `MusicValidationError`.
- Reporte estructurado con:
  - `status`,
  - `errors`,
  - `warnings`,
  - `by_track`,
  - `by_bar`,
  - `metrics`.
- Reporte HTML simple para lectura humana.
- Integracion con `export_project`.
- Modo estricto por defecto en exportacion profesional:
  - errores bloquean MIDI/MusicXML/PDF,
  - JSON/HTML de validacion se escriben antes de bloquear.
- Modo `validation_policy="report_only"` para exportar aun con errores y conservar diagnostico.

## Validadores implementados

- `BarDurationValidator`:
  - detecta huecos y sobrecargas por voz, pista y compas.
- `InstrumentRangeValidator`:
  - error fuera de rango absoluto,
  - warning fuera de rango comodo.
- `TranspositionValidator`:
  - verifica que instrumentos transpositores no produzcan pitch sonoro fuera de MIDI.
- `HarmonyValidator`:
  - calcula proporcion de chord tones/tensions por pista,
  - avisa si la relacion armonica cae bajo umbral.
- `VoiceLeadingValidator`:
  - avisa saltos melodicos excesivos.
- `BreathValidator`:
  - avisa ausencia de respiraciones y frases demasiado largas en vientos.
- `PianoPlayabilityValidator`:
  - avisa voicings demasiado densos,
  - avisa voicings demasiado abiertos,
  - avisa si un voicing marcado rootless duplica la raiz.
- `DrumValidator`:
  - error si bateria no usa canal 10 cuando se fuerza canal,
  - error con pitches de bateria no soportados,
  - warning si no hay fills.
- `ExportValidator`:
  - comprueba archivos requeridos,
  - parsea MusicXML,
  - parsea MIDI,
  - comprueba que exista un MIDI por pista.

## Caso de aceptacion

- Los errores graves bloquean `export_project` en modo estricto.
- Los warnings quedan reportados sin bloquear.
- El reporte agrupa por pista y compas.
- Hay tests con casos rotos para:
  - duracion de compas,
  - rango instrumental,
  - transposicion,
  - armonia,
  - respiracion,
  - voicings de piano,
  - bateria,
  - export bloqueado.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 58 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/validation_smoke.py`: OK.

## Smoke de generacion/exportacion

`scripts/validation_smoke.py` genera:

- `outputs/obj8_validation_demo/arrangement_project.json`
- `outputs/obj8_validation_demo/generation_spec.json`
- `outputs/obj8_validation_demo/full_arrangement.mid`
- `outputs/obj8_validation_demo/full_score.musicxml`
- `outputs/obj8_validation_demo/export_manifest.json`
- `outputs/obj8_validation_demo/validation_report.json`
- `outputs/obj8_validation_demo/validation_report.html`
- `outputs/obj8_validation_demo/midi_tracks/`

Resultado observado:

```text
Validation smoke OK: 12 bars, 6 tracks, status=pass, files=13
```

## Notas

- Los validadores avanzados reportan warnings cuando el problema no impide exportacion.
- La validacion de rango usa los rangos configurados en `configs/instruments.yaml`.
- El export profesional mantiene compatibilidad: `validation_policy="strict"` es el modo por defecto.
