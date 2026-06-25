# Objetivo 0 - Inicializar repo ejecutable

Fecha: 2026-06-24

## Alcance implementado

- Monorepo minimo ejecutable con `apps/api`, `apps/web`, `packages/arranger_core`, `packages/dataset_tools`, `configs`, `examples` y `outputs`.
- `Makefile` con targets `setup`, `setup-web`, `test`, `lint`, `api`, `web` y `demo-check`.
- Configuracion raiz en `pyproject.toml` para Ruff y Pytest.
- `.gitignore` para caches, dependencias locales y salidas generadas.
- API FastAPI con `GET /health`.
- Pantalla inicial Next.js en `apps/web/app`.
- Tests de bootstrap, configs YAML, API `/health` y scaffold web.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron los comandos equivalentes a los targets del Makefile.

- `python -m pip install -e packages/arranger_core -e packages/dataset_tools -e apps/api pytest ruff httpx pydantic pyyaml fastapi uvicorn pretty_midi music21`: OK.
- `python -m ruff check apps packages scripts`: OK.
- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 5 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web install`: OK. NPM reporta 2 vulnerabilidades moderadas transitivas; no se ejecuto `npm audit fix --force`.
- `npm --prefix apps/web run lint`: OK.
- `npm --prefix apps/web run build`: OK.
- Smoke API: Uvicorn temporal en `127.0.0.1:8010` y `GET /health`: OK.

## Smoke de generacion/exportacion

No aplica en Objetivo 0. La generacion musical, export MIDI/MusicXML/PDF y validacion musical real empiezan en objetivos posteriores.

## Notas

- No se implemento logica musical nueva ni se amplio el alcance hacia `ArrangementProject`, prompt compiler, generadores o exportadores.
- `make api` y `make web` quedan definidos para entornos con GNU Make disponible.
