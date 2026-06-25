# Objetivo 2 - Conocimiento musical base

Fecha: 2026-06-24

## Alcance implementado

- `requirements.txt` raiz para instalacion tipica con `python -m pip install -r requirements.txt`.
- `Makefile`, Dockerfile y README actualizados para usar `requirements.txt`.
- Loader YAML `MusicConfigLoader` y `load_yaml_file`.
- Utilidades de notas en `music_theory.py`: pitch class, MIDI, transposicion y nombres.
- `InstrumentCatalog` con instrumentos, ensembles y transposicion written/sounding.
- `ChordParser` por gramatica para acordes jazz complejos, slash chords, extensiones, alteraciones, adds y omits.
- `ScaleCatalog` para intervalos, notas y pitch classes de escalas.
- `ProgressionLibrary` para plantillas de progresion YAML.
- `StyleProfile` y `StyleProfileCatalog` para perfiles jazz.
- `PatternLibrary` para patrones de comping, walking bass y drums.
- Exports publicos desde `arranger_core`.

## Casos cubiertos

- Acordes requeridos: `F#m7b5`, `B7alt`, `Ebmaj7#11`, `G13b9`, `CmMaj9`, `D7#5#9/Ab`.
- Calculo de tonos base, pitch classes, tensiones, alteraciones y bajo slash.
- Carga de estilos `hard_bop`, `modal_jazz`, `jazz_ballad`, `bossa_nova`.
- Carga de progresiones como `minor_blues_12`.
- Tests de transposicion para trompeta Bb, saxo alto Eb y contrabajo.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m pip install -r requirements.txt`: OK.
- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 23 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- Smoke de conocimiento musical: parseo de acordes complejos, transposicion de trompeta, carga de progresion y estilo: OK.

## Smoke de generacion/exportacion

No aplica en Objetivo 2. No se implementaron generadores ni exportadores MIDI/MusicXML/PDF; esos entregables empiezan en objetivos posteriores.

## Notas

- El parser evita una lista cerrada de acordes y trabaja por root + calidad + extensiones + alteraciones + slash bass.
- La transposicion usa `transposition_semitones` de `configs/instruments.yaml`, donde el valor indica sounding pitch relativo a written pitch.
