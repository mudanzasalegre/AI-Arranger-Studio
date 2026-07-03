# PR-31 — Text2MIDI real sketch backend + wrapper robusto

## Objetivo

Que Text2MIDI funcione localmente como generador de sketches, no como arreglo final.

## Estado actual

Ya existe:

```text
scripts/models/run_text2midi_inference.py
scripts/models/smoke_text2midi.py
```

y el wrapper carga:

```text
model/transformer_model.py
pytorch_model.bin
vocab_remi.pkl
google/flan-t5-base
```

## Cambios obligatorios

### 1. Installer

El instalador debe:

- clonar `AMAAI-Lab/text2midi`;
- instalar requirements correctos según plataforma;
- descargar `pytorch_model.bin`;
- descargar `vocab_remi.pkl`;
- descargar/cachear `google/flan-t5-base`;
- ejecutar smoke real.

### 2. Wrapper robusto

`run_text2midi_inference.py` debe:

- aceptar `--device auto|cpu|cuda|mps`;
- aceptar `--max-len`;
- aceptar `--temperature`;
- capturar traceback;
- escribir summary JSON aunque falle;
- no bloquear la API principal.

### 3. Backend

`Text2MidiBackend` debe ejecutar el wrapper por subprocess si no encuentra un paquete importable.

Orden:

```text
1. engine importable
2. subprocess wrapper
3. worker
4. error controlado
```

### 4. Sketch import

El resultado debe entrar como:

```text
sketch_ready
sketch_uncertain
sketch_rejected
```

Nunca como `arrangement_ready`.

### 5. Prompt templates

Añadir prompts de Text2MIDI en:

```text
configs/text2midi_prompt_templates.yaml
```

Reglas:

- siempre incluir `MIDI`;
- siempre incluir instrumentos;
- siempre incluir tonalidad;
- siempre incluir tempo;
- siempre incluir forma si procede;
- evitar pedir “realistic audio”.

## Acceptance

```bash
python scripts/models/smoke_text2midi.py
python scripts/models_pro/text2midi_sketch_import_smoke.py
```

El smoke debe generar un `.mid`, importarlo a `ArrangementProject sketch` y producir validation report.
