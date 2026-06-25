# Objetivo 4 - Text prompt a GenerationSpec

Fecha: 2026-06-24

## Alcance implementado

- `PromptCompiler` determinista en `arranger_core.prompt_compiler`.
- Funcion publica `compile_prompt(prompt, seed=...)`.
- Diccionarios bilingues ES/EN para:
  - estilos,
  - tonalidades,
  - tempo,
  - formas,
  - ensembles,
  - instrumentos,
  - densidad,
  - mood,
  - roles basicos de bajo, piano, bateria y vientos.
- Campo opcional `mood` en `GenerationSpec`.
- Fallbacks inteligentes desde defaults y perfiles de estilo YAML.
- Endpoint `POST /v1/prompts/compile` conectado al compilador real.
- CLI `scripts/compile_prompt.py` con `--prompt`, `--seed` y `--output`.
- README actualizado con uso de CLI y API.

## Caso de aceptacion

Prompt:

```text
hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria
```

Produce:

- `style`: `hard_bop`
- `key`: `C minor`
- `tempo`: `132`
- `form`: `minor_blues_12`
- `ensemble`: `jazz_sextet`
- `instruments`: `drum_kit`, `double_bass`, `piano`, `alto_sax`, `trumpet_bflat`, `trombone`

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 34 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- Smoke CLI con `scripts/compile_prompt.py` y salida en `outputs/obj4_smoke/generation_spec.json`: OK.
- Smoke API real con Uvicorn temporal y `POST /v1/prompts/compile`: OK.

## Smoke de generacion/exportacion

No aplica en Objetivo 4. Este objetivo compila prompt a `GenerationSpec`; la generacion musical real empieza en objetivos posteriores.

## Notas

- La API no contiene logica musical; solo valida entrada y llama al core.
- Los detalles de extraccion y fallbacks quedan en `GenerationSpec.constraints`.
