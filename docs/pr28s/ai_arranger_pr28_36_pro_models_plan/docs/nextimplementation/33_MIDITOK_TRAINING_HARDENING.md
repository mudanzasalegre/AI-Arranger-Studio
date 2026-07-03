# PR-33 — MidiTok training pipeline hardening

## Objetivo

Convertir el smoke MidiTok en un pipeline de training útil.

## Estado actual

Ya existe `MidiTokRealTokenizer` y un smoke que:

- crea fixture MIDI por rol;
- exporta segmentos por rol;
- bloquea licencia `research_only`;
- comprueba reconstrucción;
- genera reportes.

## Cambios obligatorios

### 1. Entrada desde Dataset Profiler

Crear adapter:

```text
packages/training/training/datasets/from_dataset_manifest.py
```

Debe convertir `dataset_tools` manifest + role classifier a `MidiTokSource`.

### 2. Splits reales

Ahora los segmentos elegibles entran como `train`. Añadir split reproducible:

```text
train 80%
val   10%
test  10%
```

por `source_file_id` hash estable.

### 3. Información de pérdida

Mantener:

```text
note_count_input
note_count_reconstructed
information_loss_ratio
```

Bloquear si:

```text
max_information_loss_ratio > threshold
```

### 4. Tokenizer real configurable

Permitir:

```yaml
tokenizer:
  family: REMI
  use_programs: true
  use_tempos: true
  use_time_signatures: true
  beat_res:
    0_4: 8
    4_12: 4
```

### 5. Quality filters

No tokenizar fuentes con:

- licencia bloqueada;
- role unknown;
- menos de N notas;
- duración demasiado corta;
- time signature no soportado;
- MIDI corrupto.

### 6. Output estable

```text
data/processed/tokenized/
  manifests/
    tokenizer.json
    miditok_config.json
    tokenization_summary.json
    license_report.json
    segments.jsonl
  melody/train.jsonl
  melody/val.jsonl
  melody/test.jsonl
  ...
```

## Acceptance

```bash
python scripts/models/smoke_miditok.py
python scripts/models_pro/miditok_dataset_from_manifest_smoke.py
```

Debe pasar con fuentes synthetic y rechazar fuentes no comerciales.
