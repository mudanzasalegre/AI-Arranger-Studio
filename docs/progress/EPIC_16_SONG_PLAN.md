# Epic 16 - SongPlan y SectionPlan

Fecha: 2026-06-25
Estado: completado

## Alcance implementado

- Se anadio `arranger_core.song_planner` con los modelos `SongPlan`, `SectionPlan`, `PhrasePlan`, `GrooveMap` y `EnergyPoint`.
- Se incluyo serializacion JSON estable con `to_json`, `from_json`, `save_json` y `load_json`.
- Se implemento `SongPlanner`, determinista por `GenerationSpec.seed`, para construir planes globales a partir de `GenerationSpec` y `ArrangementProject`.
- El planner genera energia global, energia por seccion, mapas de groove, frases, densidad, funcion estructural, motivo principal, estrategia de cierre y perfil de mezcla.
- `RuleBasedArranger.generate` crea un `SongPlan` por arreglo y lo entrega a todos los generadores mediante `GenerationContext.song_plan`.
- El proyecto generado guarda el plan en `project.metadata["song_plan"]`.
- El exportador de paquetes escribe `song_plan.json` cuando el proyecto contiene plan global y registra el archivo en el manifiesto como `song_plan_json`.
- La API publica de `arranger_core` exporta los nuevos modelos y helpers.

## Criterios de aceptacion

- Generadores reciben plan global: cubierto por `GenerationContext.song_plan` y test con generador espia.
- Energia cambia por seccion: cubierto con tests de blues, AABA y ballad.
- Plan se exporta junto al proyecto: cubierto por export package y smoke `golden_generate.py`.
- Determinismo por seed: cubierto con comparacion de dos planes generados con la misma especificacion.
- Serializacion JSON: cubierta con round trip `save_json` / `load_json`.

## Tests anadidos

- `tests/test_song_planner.py::test_song_plan_serializes_and_is_seed_deterministic`
- `tests/test_song_planner.py::test_aaba_bridge_has_distinct_energy`
- `tests/test_song_planner.py::test_ballad_plan_keeps_head_and_ending_lower_density`
- `tests/test_song_planner.py::test_generators_receive_global_song_plan_and_project_metadata_exports_it`

## Verificacion ejecutada

- `python -m pytest tests/test_song_planner.py -q` - 4 passed.
- `python -m pytest -q` - 82 passed, 1 warning.
- `python -m ruff check apps packages scripts tests` - OK.
- `npm --prefix apps/web run lint` - OK.
- `python scripts/golden_generate.py` - OK, 8 presets generados en `outputs/golden`, con `song_plan.json` exportado.

## Archivos principales

- `packages/arranger_core/arranger_core/song_planner.py`
- `packages/arranger_core/arranger_core/role_generators.py`
- `packages/arranger_core/arranger_core/exporters.py`
- `packages/arranger_core/arranger_core/__init__.py`
- `tests/test_song_planner.py`
