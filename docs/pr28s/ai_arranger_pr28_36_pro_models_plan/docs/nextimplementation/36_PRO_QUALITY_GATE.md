# PR-36 — Pro quality gate, benchmark y release candidate

## Objetivo

Bloquear exportaciones que sigan sonando flojas o sean musicalmente inválidas.

## QualityGate

Crear:

```text
packages/arranger_core/arranger_core/quality/pro_quality_gate.py
configs/quality_thresholds.pro.yaml
scripts/models_pro/pro_quality_gate.py
```

## Métricas mínimas

### Globales

- sin errores bloqueantes;
- todos los compases completos;
- tracks no vacíos;
- MusicXML exportable;
- MIDI por pista exportable.

### Bajo

- beat1 root score mínimo;
- approach-to-next-root score mínimo;
- rango correcto;
- densidad controlada.

### Piano

- polifonía razonable;
- rootless violations limitadas;
- no invadir grave;
- voice leading básico.

### Drums

- canal 10;
- hits mínimos;
- fills antes de secciones;
- variación de velocity.

### Melodía/vientos

- breath rests;
- rango cómodo;
- resoluciones en cadencia;
- silencio/espacio suficiente;
- no saltos extremos excesivos.

### Modelo

- `model_trace.json` completo;
- licencias revisadas;
- no usar backends no comerciales en export comercial;
- no exportar pending takes.

## Calificación

```text
A: release candidate
B: usable/editable
C: needs manual editing
D: reject
```

## Acceptance

```bash
python scripts/models_pro/professional_benchmark_gate.py
```

Debe generar:

```text
outputs/pro_benchmarks/benchmark_summary.json
outputs/pro_benchmarks/benchmark_summary.md
```

con al menos:

```text
5 demos generadas
0 errores bloqueantes
rating medio >= B
ningún export comercial con modelo/dataset prohibido
```
