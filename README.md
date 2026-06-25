# AI Arranger Studio â€” Starter Pack

AplicaciÃ³n simbÃ³lica text-to-MIDI/text-to-score para generar arreglos multipista editables.

## FilosofÃ­a

No se genera audio primero. Se genera una representaciÃ³n musical interna rica y desde ahÃ­ se exporta:

- MIDI multipista
- MIDI por instrumento
- MusicXML
- PDF de full score
- PDF de particellas
- audio preview
- JSON `ArrangementProject`

Primera versiÃ³n: sin IA pesada. La generaciÃ³n se hace con reglas, plantillas, patrones, anÃ¡lisis de datasets y selecciÃ³n probabilÃ­stica reproducible. La arquitectura queda preparada para sustituir generadores por modelos de IA mÃ¡s adelante.

## Lectura obligatoria para Codex

1. `docs/00_MASTER_RECIPE.md`
2. `docs/01_CODEX_WORKFLOW.md`
3. `docs/02_ARCHITECTURE.md`
4. `docs/03_MUSICAL_SPEC_JAZZ.md`
5. `docs/04_OBJECTIVES.md`
6. `docs/05_DATASET_LEARNING_MODE.md`
7. `docs/06_VALIDATION_AND_QA.md`
8. `docs/07_API_SPEC.md`
9. `docs/08_EXPORT_SPEC.md`
10. `docs/09_AI_EXTENSION_CONTRACT.md`
11. `docs/10_PACKAGING_AND_SMOKE.md`
12. `docs/11_FUTURE_AI_TRAINING.md`

## Objetivo del MVP potente

Entrada:

```json
{
  "prompt": "hard bop nocturno en Do menor, quinteto con saxo alto, trompeta, trombÃ³n, piano, contrabajo y baterÃ­a, 132 bpm, forma blues menor de 12 compases, con shout chorus final",
  "seed": 1234
}
```

Salida:

```text
outputs/<project_id>/
  arrangement_project.json
  full_arrangement.mid
  midi_tracks/
    drums.mid
    double_bass.mid
    piano.mid
    alto_sax.mid
    trumpet.mid
    trombone.mid
  full_score.musicxml
  full_score.pdf
  parts_pdf/
    alto_sax.pdf
    trumpet.pdf
    trombone.pdf
    piano.pdf
    double_bass.pdf
    drums.pdf
  audio_preview.wav|mp3
  validation_report.html
  validation_report.json
```

## Stack propuesto

- Backend: Python + FastAPI
- NÃºcleo musical: paquete Python `arranger_core`
- ExportaciÃ³n: MIDI + MusicXML + MuseScore CLI
- Frontend: Next.js + OpenSheetMusicDisplay
- Dataset mode: ingesta MIDI/MusicXML, normalizaciÃ³n, extracciÃ³n de patrones y manifiestos de licencia
- IA futura: adaptadores enchufables por rol musical

## Inicializar el proyecto

Desde la raiz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
npm --prefix apps/web install
```

Comprobar que todo esta bien:

```powershell
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
```

Arrancar servicios en terminales separadas:

```powershell
python -m uvicorn app.main:app --app-dir apps/api --reload --port 8000
npm --prefix apps/web run dev
```

Equivalentes con GNU Make, si esta instalado:

```bash
make setup
make setup-web
make lint
make test
make api
make web
```

La API queda en `http://127.0.0.1:8000` y el frontend en `http://127.0.0.1:3000`.

## Roadmap actual y control de versiones

La implementacion incremental vigente esta documentada en `docs/nextimplementation/`.
El backlog actual parte de Epic 15/16 y continua por PR-00 a PR-19.

Este repositorio no versiona datasets MIDI, salidas generadas, pesos de modelos ni
datos privados. En particular, `midi_databases/`, `outputs/`, `models/`,
`data/raw/`, `data/private/`, `data/processed/`, archivos MIDI/MusicXML privados
y checkpoints de modelos deben permanecer locales. Usa `.env.example` como
plantilla y guarda secretos reales solo en `.env`.

## Generar y empaquetar el demo jazz

Con las dependencias instaladas, el flujo de smoke test completo es:

```bash
make package-smoke
```

Esto genera `outputs/obj13_demo_jazz/`, valida el proyecto exportado y crea
`outputs/obj13_demo_jazz.zip`. Si MuseScore CLI esta instalado, tambien exporta
`full_score.pdf` y PDFs de particellas.

Targets separados:

```bash
make demo-jazz
make export-demo
make validate-demo
make zip-demo
make demo-check
```

Equivalentes en PowerShell:

```powershell
python scripts/demo_jazz.py
python scripts/validate_demo.py
python scripts/zip_demo.py
python scripts/package_smoke.py
```

La guia completa de instalacion y smoke tests esta en
`docs/10_PACKAGING_AND_SMOKE.md`.

## Compilar un prompt

Por CLI:

```powershell
python scripts/compile_prompt.py --prompt "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria" --seed 1234
```

Por API, con el servidor arrancado:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/prompts/compile -Method Post -ContentType application/json -Body '{"prompt":"hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria","seed":1234}'
```

## Primer comando recomendado para Codex

```text
Lee todo docs/*.md y configs/*.yaml. Implementa el Objetivo 0 de docs/04_OBJECTIVES.md completo. No avances al Objetivo 1 hasta que `make test` y `make lint` funcionen.
```

