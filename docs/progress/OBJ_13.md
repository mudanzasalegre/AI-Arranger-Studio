# Objetivo 13 - Packaging y smoke tests

## Estado

Completado.

## Implementado

- Targets `make demo-jazz`, `make export-demo`, `make validate-demo`, `make zip-demo`
  y `make package-smoke`.
- `make demo-check` ahora ejecuta el smoke real de packaging.
- Script `scripts/demo_jazz.py` para generar el paquete demo canonico desde el preset
  `jazz_hard_bop_minor_blues_sextet`.
- Script `scripts/validate_demo.py` para validar el proyecto y el paquete exportado.
- Script `scripts/zip_demo.py` para crear y verificar `outputs/obj13_demo_jazz.zip`.
- Script `scripts/package_smoke.py` para ejecutar generacion, validacion y ZIP end to end.
- Tests de regresion para los targets y el flujo demo/validacion/ZIP sin PDF.
- Documentacion de instalacion y smoke tests en `docs/10_PACKAGING_AND_SMOKE.md`.

## Salidas esperadas

- `outputs/obj13_demo_jazz/arrangement_project.json`
- `outputs/obj13_demo_jazz/full_arrangement.mid`
- `outputs/obj13_demo_jazz/full_score.musicxml`
- `outputs/obj13_demo_jazz/validation_report.json`
- `outputs/obj13_demo_jazz/validation_report.html`
- `outputs/obj13_demo_jazz/export_manifest.json`
- `outputs/obj13_demo_jazz.zip`

Si MuseScore CLI esta disponible, el smoke tambien genera `full_score.pdf` y
PDFs de particellas.

## Verificacion

Ejecutado:

- `python -m ruff check apps packages scripts tests` - OK
- `python -m pytest -q` - OK, 71 tests
- `npm --prefix apps/web run lint` - OK
- `npm --prefix apps/web run build` - OK
- `python scripts/bootstrap_check.py` - OK
- `python scripts/package_smoke.py` - OK

Resultado del smoke:

- preset: `jazz_hard_bop_minor_blues_sextet`
- barras: 12
- pistas: 6
- validacion: `pass`, 0 errores, 0 warnings
- PDF: `created`
- MuseScore CLI: `C:\Program Files\MuseScore 4\bin\MuseScore4.EXE`
- ZIP: `outputs/obj13_demo_jazz.zip`

No se pudo ejecutar `make package-smoke` en este entorno porque `make` no esta
instalado en PATH. El target queda cubierto por tests de Makefile y por el
comando equivalente `python scripts/package_smoke.py`.
