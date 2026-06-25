# Objetivo 10 - API completa

Fecha: 2026-06-24

## Alcance implementado

- API FastAPI completa en `apps/api/app/main.py`.
- OpenAPI generado automaticamente por FastAPI en `/openapi.json`.
- Persistencia local simple en:
  - `outputs/api/` por defecto,
  - o la ruta configurada con `AI_ARRANGER_API_STORAGE`.
- Orquestacion de:
  - prompt compiler,
  - generacion rule-based,
  - exportacion MIDI/MusicXML/PDF opcional,
  - validacion musical,
  - regeneracion,
  - importacion de datasets,
  - busqueda de patrones aprendidos.
- Tests de endpoints en `tests/test_api.py`.
- Smoke end-to-end en `scripts/api_smoke.py`.

## Endpoints implementados

- `GET /health`
- `POST /v1/prompts/compile`
- `POST /v1/projects/generate`
- `GET /v1/projects/{project_id}`
- `POST /v1/projects/{project_id}/export`
- `GET /v1/projects/{project_id}/validation`
- `POST /v1/projects/{project_id}/regenerate`
- `POST /v1/datasets/import`
- `GET /v1/datasets`
- `GET /v1/patterns/search`
- `GET /v1/projects/{project_id}/file`
- `GET /v1/projects/{project_id}/zip`

## Contratos principales

- `POST /v1/projects/generate` acepta:
  - `prompt`,
  - `seed`,
  - `project_id` opcional,
  - `spec` opcional como `GenerationSpec`,
  - `options.export`,
  - `options.validate`,
  - `options.include_pdf`,
  - `options.validation_policy`.
- `GET /v1/projects/{project_id}` devuelve:
  - metadata del proyecto,
  - `GenerationSpec`,
  - export manifest,
  - validation report si existe.
- `POST /v1/projects/{project_id}/export` exporta un proyecto persistido.
- `GET /v1/projects/{project_id}/validation` devuelve o recalcula el reporte.
- `POST /v1/projects/{project_id}/regenerate` regenera el proyecto con nueva seed y conserva target/instruction en metadata.
- `POST /v1/datasets/import` importa una carpeta local con manifiesto existente o genera uno desde `default_metadata`.
- `GET /v1/patterns/search` filtra por:
  - dataset,
  - categoria,
  - rol,
  - estilo,
  - calidad minima,
  - tags,
  - `usable_for_training`,
  - `usable_for_pattern_extraction`.

## Caso de aceptacion

- API OpenAPI generada y cubierta por tests.
- Todos los endpoints pedidos en `docs/04_OBJECTIVES.md` existen en OpenAPI.
- La descarga de archivos y ZIP queda expuesta para la interfaz web de OBJ11.
- La generacion completa funciona desde endpoint.
- La exportacion desde endpoint genera manifest y archivos.
- La validacion se expone por endpoint.
- Dataset mode queda expuesto por endpoint.
- La busqueda de patrones aprendidos queda expuesta por endpoint.

## Verificacion ejecutada

- `python -m pytest -q`: OK, 64 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python -m ruff check apps packages scripts tests`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/bootstrap_check.py`: OK.
- `python scripts/api_smoke.py`: OK. Muestra el mismo warning externo de Starlette/FastAPI.

## Smoke de API

`scripts/api_smoke.py` genera:

- `outputs/obj10_api_demo/api_storage/projects/obj10-api-smoke/`
- `outputs/obj10_api_demo/api_storage/datasets/obj10-dataset/`
- `outputs/obj10_api_demo/dataset_source/`
- `outputs/obj10_api_demo/api_smoke_summary.json`

Resultado observado:

```json
{
  "openapi_paths": 12,
  "project_id": "obj10-api-smoke",
  "generated_status": "generated",
  "project_bar_count": 12,
  "validation_status": "pass",
  "regenerated_status": "regenerated",
  "exported_files": 10,
  "dataset_status": "imported",
  "datasets_count": 1,
  "patterns_found": 1
}
```

## Notas

- La API no guarda estado en memoria; cada request lee/escribe archivos bajo el storage configurado.
- La opcion JSON publica sigue siendo `options.validate`; internamente usa un alias para evitar conflicto con metodos de Pydantic.
- `include_pdf` queda desactivado por defecto para endpoints y smoke; puede activarse si MuseScore CLI esta instalado.
