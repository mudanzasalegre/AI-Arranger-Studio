# 07 — API spec

## Endpoints principales

### GET /health

Devuelve estado.

### POST /v1/prompts/compile

Entrada:

```json
{
  "prompt": "hard bop en Do menor a 132 bpm con sexteto",
  "seed": 1234
}
```

Salida: `GenerationSpec`.

### POST /v1/projects/generate

Entrada:

```json
{
  "prompt": "hard bop nocturno en Do menor...",
  "options": {
    "export": true,
    "validate": true
  },
  "seed": 1234
}
```

Salida:

```json
{
  "project_id": "...",
  "status": "generated",
  "files": [],
  "validation": {}
}
```

### GET /v1/projects/{id}

Devuelve metadata y manifest.

### GET /v1/projects/{id}/file?kind=musicxml

Descarga archivo.

### POST /v1/projects/{id}/regenerate

Entrada:

```json
{
  "target": {
    "track": "trombone",
    "bars": [9, 10, 11, 12]
  },
  "instruction": "menos movimiento, registro más cómodo",
  "seed": 999
}
```

### GET /v1/projects/{id}/validation

Devuelve reporte.

### POST /v1/datasets/import

Importa carpeta o lote de archivos.

### GET /v1/patterns/search

Busca patrones aprendidos por estilo/rol/contexto.
