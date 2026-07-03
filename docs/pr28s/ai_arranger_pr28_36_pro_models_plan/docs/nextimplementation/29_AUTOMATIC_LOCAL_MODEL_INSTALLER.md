# PR-29 — Instalador automático de modelos locales

## Objetivo

Crear un instalador reproducible para:

- configurar `.env`;
- crear carpetas locales;
- instalar dependencias opcionales;
- descargar/cachear MIDI-GPT;
- clonar Text2MIDI;
- descargar pesos Text2MIDI;
- comprobar Ollama;
- escribir `models/manifests/install_report.json`;
- no subir nada pesado a Git.

## Archivos a implementar

```text
scripts/models_pro/install_all_local_models.py
scripts/models_pro/activate_pro_profile.py
scripts/models_pro/verify_all_models.py
configs/ai_models.pro.yaml
configs/local_model_runtime.pro.yaml
configs/model_install_plan.yaml
```

## Comando principal

```bash
python scripts/models_pro/install_all_local_models.py --profile pro --planner-model qwen3:8b
```

## Qué debe hacer

### 1. Comprobar Python

Aceptar Python 3.10–3.12.

### 2. Instalar requirements

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ai.txt
python -m pip install -r requirements-training-ai.txt
```

### 3. Crear carpetas

```text
models/hf_cache/hub
models/hf_cache/assets
models/external_repos/text2midi
models/checkpoints/text2midi
models/checkpoints/custom/{melody,bass,piano_comping,horns,drums}
models/manifests
outputs/model_artifacts/{raw,imported,rejected,validated}
outputs/model_smoke
outputs/pro_benchmarks
```

### 4. Setear cache local

Antes de importar `huggingface_hub`, debe setear:

```text
HF_HOME=./models/hf_cache
HF_HUB_CACHE=./models/hf_cache/hub
HF_ASSETS_CACHE=./models/hf_cache/assets
HF_HUB_DISABLE_TELEMETRY=1
```

### 5. MIDI-GPT

Ejecutar:

```bash
python scripts/models/download_midigpt.py
python scripts/models/smoke_midigpt.py
```

### 6. Text2MIDI

Si `models/external_repos/text2midi` no existe:

```bash
git clone https://github.com/AMAAI-Lab/text2midi models/external_repos/text2midi
```

Descargar:

```text
pytorch_model.bin
vocab_remi.pkl
```

Ejecutar:

```bash
python scripts/models/download_text2midi.py --checkpoint-dir models/checkpoints/text2midi
python scripts/models/smoke_text2midi.py
```

### 7. Ollama

No instalar Ollama automáticamente si no existe en el sistema. Detectarlo y dar instrucción clara.

Si existe:

```bash
ollama pull qwen3:8b
python scripts/models/smoke_ollama_planner.py
```

### 8. MidiTok

Ejecutar:

```bash
python scripts/models/smoke_miditok.py
```

### 9. Reporte

Escribir:

```text
models/manifests/install_report.json
models/manifests/model_status.json
```

## Criterio de aceptación

```bash
python scripts/models_pro/verify_all_models.py
```

devuelve `status: ok` o `status: partial_ok` con errores explícitos no fatales.
