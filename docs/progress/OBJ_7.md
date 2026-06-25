# Objetivo 7 - Generadores rule-based por rol

Fecha: 2026-06-24

## Alcance implementado

- Modulo `arranger_core.role_generators`.
- Orquestador `RuleBasedArranger`.
- Funcion publica `generate_arrangement(spec, ...)`.
- Contrato `GenerationContext` y protocolo `RoleGenerator`.
- Generadores:
  - `DrumsGenerator`,
  - `WalkingBassGenerator`,
  - `PianoCompingGenerator`,
  - `MelodyGenerator`,
  - `HornResponseGenerator`,
  - `ShoutChorusGenerator`,
  - `Humanizer`.
- Integracion con `HarmonyFormEngine` para forma y cifrado.
- Integracion con `LeadSheetGenerator` para la melodia principal.
- Seleccion de instrumentos desde `GenerationSpec.instruments` o `ensemble`.
- Trio jazz con bateria, contrabajo y piano; el piano integra melodia en una segunda voz.
- Quartet/quintet/sextet con lead horn, respuestas de vientos y shout chorus en los ultimos 4 compases.

## Reglas musicales implementadas

- Bateria:
  - ride swing en subdivisiones de corchea,
  - hi-hat pedal en 2 y 4,
  - kick ligero,
  - snare ghost,
  - fills cada 4 compases y al final.
- Bajo:
  - walking bass a negras,
  - tiempo 1 en raiz,
  - notas guia/quinta en tiempos interiores,
  - aproximacion cromatica al siguiente acorde.
- Piano:
  - comping sincopado por densidad,
  - voicings rootless basicos,
  - evita duplicar la raiz en las notas marcadas como rootless.
- Melodia:
  - reutiliza el generador de lead sheet del objetivo 6.
- Vientos:
  - respuestas en huecos de frase,
  - hits de shout chorus en los ultimos 4 compases,
  - rangos comodos desde `configs/instruments.yaml`.
- Humanizacion:
  - variacion reproducible de velocities,
  - anotaciones `humanized_timing_ms` sin alterar duraciones notadas.

## Caso de aceptacion

- `jazz_trio` genera 3 pistas: drums, double bass, piano.
- `jazz_quartet_alto` genera 4 pistas con lead horn.
- `jazz_quintet` y `jazz_sextet` generan respuestas de vientos.
- `jazz_sextet` genera trompeta y trombon con `horn_response`.
- El shout chorus aparece en los ultimos 4 compases de pistas de viento.
- Los proyectos generados validan duraciones de compas.
- El exportador crea un MIDI por pista.
- MusicXML exportado es parseable.

## Verificacion ejecutada

En esta maquina Windows no esta instalado `make`, asi que se ejecutaron comandos equivalentes.

- `python -m ruff check apps packages scripts tests`: OK.
- `python -m pytest -q`: OK, 51 tests pasan. Queda un warning externo de `fastapi.testclient`/Starlette sobre `httpx`.
- `python scripts/bootstrap_check.py`: OK.
- `npm --prefix apps/web run lint`: OK.
- `python scripts/arrangement_smoke.py`: OK.

## Smoke de generacion/exportacion

`scripts/arrangement_smoke.py` genero tres paquetes sin PDF:

- `outputs/obj7_arrangement_demo/jazz_trio`
- `outputs/obj7_arrangement_demo/jazz_quartet`
- `outputs/obj7_arrangement_demo/jazz_sextet`

Resultado observado:

```text
Generated jazz_trio: 12 bars, 3 tracks, 3 MIDI track files
Generated jazz_quartet: 32 bars, 4 tracks, 4 MIDI track files
Generated jazz_sextet: 12 bars, 6 tracks, 6 MIDI track files
```

Cada paquete contiene `arrangement_project.json`, `generation_spec.json`, `full_arrangement.mid`,
`full_score.musicxml`, `export_manifest.json`, `validation_report.json` y `midi_tracks/`.

## Notas

- No se han implementado validadores musicales avanzados; pertenecen al objetivo 8.
- La humanizacion deja anotaciones de timing, pero no desplaza offsets de notacion para mantener compases validos.
- Los patrones son rule-based y no copian nombres ni melodias de standards.
