# PR-22 — Text2MIDI local

## Objetivo

Instalar Text2MIDI localmente para producir bocetos MIDI desde prompt.

Text2MIDI se usa como sketch generator, no como productor final.

## Instalación

```bash
mkdir -p models/external_repos
# si git está disponible:
git clone https://github.com/AMAAI-Lab/text2midi models/external_repos/text2midi
cd models/external_repos/text2midi
python -m pip install -r requirements.txt
```

En Mac Apple Silicon/MPS, usar `requirements-mac.txt` si el repo lo requiere.

## Descarga de pesos

```bash
python scripts/models/download_text2midi.py \
  --checkpoint-dir models/checkpoints/text2midi
```

Debe descargar:

```text
models/checkpoints/text2midi/pytorch_model.bin
models/checkpoints/text2midi/vocab_remi.pkl
```

## Smoke test

```bash
python scripts/models/smoke_text2midi.py \
  --repo-dir models/external_repos/text2midi \
  --checkpoint-dir models/checkpoints/text2midi \
  --output outputs/model_artifacts/raw/text2midi_smoke.mid
```

## Integración recomendada

El backend actual puede no mapear directamente a la estructura real del repo externo. Implementar modo worker/subprocess:

```text
API -> Text2MidiBackend -> subprocess/worker -> output.mid -> artifact raw
```

Crear un wrapper interno:

```text
scripts/models/run_text2midi_inference.py
```

que:

1. Añada `models/external_repos/text2midi` al `PYTHONPATH`.
2. Cargue `model.transformer_model.Transformer`.
3. Cargue `pytorch_model.bin`.
4. Cargue `vocab_remi.pkl`.
5. Use `T5Tokenizer.from_pretrained("google/flan-t5-base")`.
6. Genere `output.mid`.

## Activación

```yaml
text2midi:
  enabled: true
```

## API test

```bash
curl -X POST http://127.0.0.1:8000/v1/ai/text-to-midi-sketch \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "text2midi",
    "prompt": "Hard bop minor blues in C minor, 132 BPM, jazz sextet with walking bass, piano comping, alto sax lead, trumpet and trombone responses.",
    "seed": 2201
  }'
```

Resultado permitido:

```text
status: sketch_validated
status: sketch_uncertain
```

Resultado no permitido:

```text
sketch promoted directly to final arrangement
```

## Acceptance criteria

```text
- Checkpoints descargados localmente.
- Smoke test produce MIDI.
- Endpoint sketch crea ArrangementProject sketch.
- Role classifier marca pistas y confidencias.
- Si roles son inciertos, `sketch_uncertain`.
- Nunca se crea export final automáticamente desde Text2MIDI.
```
