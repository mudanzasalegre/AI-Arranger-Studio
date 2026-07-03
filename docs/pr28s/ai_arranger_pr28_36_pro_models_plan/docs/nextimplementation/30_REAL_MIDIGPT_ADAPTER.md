# PR-30 — MIDI-GPT real adapter + infill profesional

## Objetivo

Sustituir el adapter genérico por una integración exacta con la API real de MIDI-GPT.

## Problema actual

El backend actual intenta invocar métodos genéricos (`generate_infill`, `generate_track`, `generate`) sobre el engine. Eso no es suficiente para la API real, cuyo flujo debe ser:

```python
from midigpt import Score
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

engine = InferenceEngine.from_pretrained("yellow")
score = Score.from_midi(context_midi)
request = GenerationRequest(...)
result = engine.session(score, request).run()
result.to_midi(output_path)
```

## Cambios obligatorios

### 1. Resolver track index

Crear método:

```python
_resolve_track_index(score, track_id, project)
```

Debe mapear `ArrangementProject.track.id` a track MIDI-GPT por:

1. nombre de pista exacto;
2. metadata `track_id`;
3. instrumento/rol;
4. fallback por orden.

Si falla, error controlado.

### 2. Convertir compases 1-based → 0-based

La API de AI Arranger usa barras 1-based. MIDI-GPT usa índices 0-based.

```python
target_bars = [bar - 1 for bar in request.bars]
```

### 3. Ignorar pistas bloqueadas

Para cada track de `score.tracks`:

```python
TrackPrompt(id=i, bars=[], ignore=True)
```

excepto el target.

### 4. Config por task

Para `infill_bars`:

```python
TrackPrompt(id=target_idx, bars=target_bars)
```

Para `generate_track`:

```python
TrackPrompt(id=target_idx, bars=target_bars, autoregressive=True)
```

### 5. Atributos

Mapear `density` a `note_density`.

Ejemplo:

```text
low         -> 2
medium      -> 5
medium_high -> 7
high        -> 8
```

### 6. Chunking

`yellow` soporta 4 y 8 compases. Si el usuario pide más, dividir en chunks de 4/8 compases y fusionar.

### 7. Retry

Hasta 3 intentos:

```text
temperatura: 1.0 → 0.85 → 0.7
top_p: 0.95 → 0.9 → 0.85
polyphony_hard_limit: según instrumento
```

### 8. Postprocesado

Después de `result.to_midi`:

- importar artifact;
- extraer solo pista/compases target;
- validar duración;
- validar rango;
- validar armonía;
- validar densidad;
- crear take.

## Acceptance

```bash
python scripts/models/smoke_midigpt.py
python scripts/models_pro/midigpt_project_infill_smoke.py
```

Debe producir:

```text
outputs/pro_benchmarks/midigpt_infill_smoke/
  base_project.json
  context.mid
  generated_raw.mid
  candidate_project.json
  validation_report.json
  take_manifest.json
```
