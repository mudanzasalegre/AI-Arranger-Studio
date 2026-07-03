# PR-34 — Modelos propios por rol: bootstrap entrenable y no dummy

## Objetivo

Pasar de `DummyCustomRoleModelBackend` a backends propios funcionales, aunque sean pequeños.

## Enfoque correcto

No entrenar un gran modelo único.

Crear modelos por rol:

```text
melody
walking_bass
piano_comping
horn_responses
drums
```

## Baseline funcional obligatorio

Antes de un Transformer propio, implementar:

```text
StatisticalCustomRoleBackend
```

que use:

- patrones tokenizados;
- n-grams;
- Markov;
- retrieval ponderado;
- reglas musicales del core;
- export a tokens y/o MIDI parcial.

## Archivos

```text
packages/model_backends/model_backends/custom_role/statistical_backend.py
packages/training/training/models/role_ngram.py
scripts/models_pro/train_custom_role_ngram_models.py
scripts/models_pro/smoke_custom_role_ngram_models.py
```

## Checkpoint por rol

Cada rol debe tener:

```text
models/checkpoints/custom/<role>/<model_id>/
  model.json
  tokenizer.json
  config.yaml
  training_manifest.yaml
  license_report.json
  metrics.json
```

## Qué debe generar

- `melody`: frases y variaciones.
- `walking_bass`: líneas por compás con aproximaciones.
- `piano_comping`: voicings/ritmos.
- `horn_responses`: stabs/respuestas.
- `drums`: fills/grooves.

## Validación

El backend no decide si su output es bueno. Solo propone.

Debe pasar:

```text
ArtifactImporter
ProjectMerger
ValidationGate
QualityGate
```

## Acceptance

```bash
python scripts/models_pro/train_custom_role_ngram_models.py
python scripts/models_pro/smoke_custom_role_ngram_models.py
```

debe crear checkpoints y generar artifacts no dummy.
