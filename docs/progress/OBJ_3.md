# Objetivo 3 - Exportacion MIDI/MusicXML/PDF

Fecha: 2026-06-24

## Alcance implementado

- Exportador `export_project(project, output_dir)` en `arranger_core`.
- MIDI full arrangement en `full_arrangement.mid` con:
  - pista conductor,
  - tempo inicial,
  - compas,
  - key signature,
  - markers de seccion,
  - una pista MIDI por instrumento,
  - nombres de pista,
  - program changes General MIDI,
  - canal 10 para bateria/percussion.
- MIDI separado por pista en `midi_tracks/<track>.mid`.
- MusicXML full score en `full_score.musicxml` con:
  - partes,
  - compases,
  - tempo,
  - compas,
  - armadura,
  - cifrado armonico,
  - marcadores de seccion,
  - notas, acordes y silencios.
- PDF full score y PDFs de partes via MuseScore CLI cuando esta instalado.
- `export_manifest.json`, `generation_spec.json`, `arrangement_project.json` y `validation_report.json`.
- Script reproducible `scripts/export_demo.py`.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 27 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/export_demo.py`: OK.
- Smoke explicito:
  - `outputs/demo/full_arrangement.mid`: creado.
  - `outputs/demo/midi_tracks/double_bass.mid`: creado.
  - `outputs/demo/midi_tracks/piano.mid`: creado.
  - `outputs/demo/full_score.musicxml`: creado y parseable con `music21.converter.parse`.
  - `outputs/demo/full_score.pdf`: creado con MuseScore 4 instalado.
  - `outputs/demo/parts_pdf/double_bass.pdf`: creado.
  - `outputs/demo/parts_pdf/piano.pdf`: creado.

## Notas

- El exporter no genera musica nueva; exporta proyectos `ArrangementProject` existentes.
- La validacion incluida en este objetivo es basica: duracion de compases y comprobacion de archivos exportados. Los validadores musicales avanzados quedan para el Objetivo 8.
- Si MuseScore CLI no existe, el manifest marca PDF como `skipped` y conserva MIDI/MusicXML.
