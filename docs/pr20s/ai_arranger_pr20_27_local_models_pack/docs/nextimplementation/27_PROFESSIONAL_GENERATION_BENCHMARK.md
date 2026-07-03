# PR-27 — Professional Generation Benchmark

## Objetivo

Medir si la app ya genera proyectos MIDI suficientemente sólidos para DAW/MuseScore usando modelos locales y validadores.

## Benchmarks

Config en:

```text
configs/professional_benchmarks.yaml
```

Casos iniciales:

```text
hard_bop_minor_blues_sextet
bebop_blues_quintet
modal_jazz_quartet
jazz_ballad_trio
bossa_quartet
```

## Ejecución

```bash
python scripts/models/professional_generation_benchmark.py \
  --config configs/professional_benchmarks.yaml \
  --api http://127.0.0.1:8000
```

## Salida

```text
outputs/professional_benchmark/
  summary.json
  summary.md
  hard_bop_minor_blues_sextet/
    full_arrangement.mid
    full_score.musicxml
    validation_report.json
    model_trace.json
    package.zip
```

## Métricas mínimas

```text
- validation_errors = 0
- pistas requeridas presentes
- MIDI full existe y no está vacío
- MusicXML existe
- si IA se usa: model_trace existe
- no hay pending takes en export final
- take accept/reject funciona
```

## Métricas musicales recomendadas

```text
- notas por compás por rol
- rango por instrumento
- large leaps por instrumento melódico
- beat1_root_score en bajo
- approach_to_next_root_score en bajo
- rootless_violations en piano
- breath_rest_count en vientos
- fill_bar_count en batería
```

El repo ya tiene `scripts/golden_generate.py`; usar sus métricas como referencia.

## Acceptance criteria

```text
- Los 5 benchmarks corren de forma reproducible con seed.
- Cada benchmark genera paquete exportable.
- No hay errores de validación bloqueantes.
- El resumen indica qué backends se usaron.
- Si un modelo local falla, el benchmark cae a rule-based/retrieval y lo registra.
```
