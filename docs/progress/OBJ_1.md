# Objetivo 1 - Modelo simbolico ArrangementProject

Fecha: 2026-06-24

## Alcance implementado

- `GenerationSpec` con version de schema, prompt normalizado, estilo, tempo, tonalidad, compas, forma, ensemble, densidad, complejidad, instrumentos, constraints y seed.
- `ArrangementProject` como formato maestro con metadata, spec, tempo/key/meter maps, forma, chord grid, tracks, validation report y export manifest.
- Modelos `Track`, `Bar`, `NoteEvent`, `RestEvent`, `ChordSymbol` y `Section`.
- Versionado inicial con `SCHEMA_VERSION = "0.1.0"` y rechazo de versiones no soportadas.
- Serializacion/carga JSON con `to_json`, `from_json`, `save_json`, `load_json`, `load_project_json` y `save_project_json`.
- Validacion de duracion de compas con deteccion de huecos y exceso por voz.
- Export publico desde `arranger_core.__init__`.
- Ejemplo `examples/projects/arrangement_project.example.json` actualizado al formato con bars/eventos.

## Decisiones de contrato

- `start` y `duration` de eventos se expresan en beats de negra dentro del compas.
- Un compas sin eventos representa silencio y no falla validacion.
- Si hay eventos, se valida la cobertura temporal por voz contra el compas efectivo.
- Las notas simultaneas con el mismo intervalo se aceptan como acorde y no cuentan como exceso.
- La duracion esperada del compas se deriva de `Bar.meter`, o del `meter_map`/`GenerationSpec.meter` del proyecto.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron los comandos equivalentes a los targets.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 12 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- Smoke simbolico: crear `ArrangementProject`, validar compas, guardar en `outputs/obj1_smoke/arrangement_project.json`, cargar y comparar round-trip: OK.

## Smoke de generacion/exportacion

No aplica en Objetivo 1. No se implementaron generadores ni exportadores MIDI/MusicXML/PDF; esos entregables empiezan en objetivos posteriores.
