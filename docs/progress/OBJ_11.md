# Objetivo 11 - Interfaz web usable

Fecha: 2026-06-24

## Alcance implementado

- Aplicacion Next.js convertida de mock estatico a estudio simbolico usable.
- `apps/web/app/page.tsx` ahora es cliente React con:
  - estado de API,
  - compilacion de prompt,
  - generacion de proyecto,
  - carga de proyecto por ID,
  - exportacion,
  - validacion,
  - regeneracion,
  - importacion de datasets,
  - busqueda de patrones,
  - descarga de archivos y ZIP.
- `apps/web/app/globals.css` renovado con layout operativo, denso y responsivo.
- API ampliada para soportar la interfaz:
  - `GET /v1/projects/{project_id}/file?kind=...`,
  - `GET /v1/projects/{project_id}/zip`,
  - CORS para `localhost:3000` y `127.0.0.1:3000`.

## Pantallas implementadas

- Home:
  - resumen del proyecto activo,
  - estado de validacion,
  - numero de tracks y archivos.
- New project:
  - prompt,
  - seed,
  - toggle PDF,
  - compile,
  - generate.
- Project detail:
  - metadata,
  - `GenerationSpec`,
  - tracks,
  - regeneracion dirigida.
- Score viewer:
  - render MusicXML con OpenSheetMusicDisplay,
  - preview audible guia via Web Audio,
  - enlace al MIDI completo.
- Track mixer simple:
  - mute,
  - solo,
  - volumen local por pista.
- Chord/form editor:
  - grid de acordes editable localmente,
  - vista de secciones/form,
  - regeneracion.
- Validation report:
  - errores,
  - warnings,
  - agrupacion visible por pista/compas cuando existe.
- Dataset library:
  - importacion de carpeta local,
  - lista de datasets,
  - busqueda de patrones.
- Export panel:
  - exportar,
  - toggle PDF,
  - abrir archivos,
  - descargar ZIP.

## Caso de aceptacion

- Se puede escribir un prompt y generar un proyecto desde la web.
- La partitura se renderiza desde `full_score.musicxml` usando OSMD.
- Hay preview audible guia y enlace al MIDI exportado.
- Se puede descargar ZIP del proyecto exportado.
- Se pueden ver errores/warnings de validacion.

## Verificacion ejecutada

- `python -m pytest -q`: OK, 65 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python -m ruff check apps packages scripts tests`: OK.
- `npm --prefix apps/web run lint`: OK.
- `npm --prefix apps/web run build`: OK.
- `python scripts/bootstrap_check.py`: OK.
- `python scripts/api_smoke.py`: OK. Muestra el mismo warning externo de Starlette/FastAPI.

## Smoke de API consumida por la web

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

- La URL de API del frontend se configura con `NEXT_PUBLIC_API_BASE_URL`; por defecto usa `http://127.0.0.1:8000`.
- El mixer es local a la UI en esta version; no modifica todavia el `ArrangementProject` persistido.
- El editor de acordes/formas permite edicion local y regeneracion; no persiste cambios directos porque no existe endpoint de patch de proyecto en el alcance actual.
