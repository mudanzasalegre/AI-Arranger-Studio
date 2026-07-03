# PR-28 — Repo health, environment lock y baseline ejecutable

## Objetivo

Congelar el estado tras PR-27 y crear una base reproducible antes de instalar modelos.

## Tareas

1. Ejecutar:

```bash
python -m pip install -r requirements.txt
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web install
npm --prefix apps/web run lint
python scripts/package_smoke.py
python scripts/golden_generate.py
python scripts/ai_contract_smoke.py
python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.example.yaml
```

2. Crear:

```text
outputs/pro_audit/pr28_repo_health.json
outputs/pro_audit/pr28_repo_health.md
```

3. Corregir cualquier fallo antes de instalar modelos.

4. Confirmar que `.env.example` contiene:
   - `AI_MODELS_CONFIG`
   - `LOCAL_MODEL_RUNTIME_CONFIG`
   - `MODEL_REGISTRY_CONFIG`
   - `AI_MODELS_ROOT`
   - `HF_HOME`
   - `HF_HUB_CACHE`
   - `AI_ENABLE_MIDIGPT`
   - `AI_ENABLE_TEXT2MIDI`
   - `AI_ENABLE_LOCAL_LLM_PLANNER`
   - `AI_ENABLE_CUSTOM_ROLE_MODELS`

5. Confirmar que `.gitignore` excluye:
   - `models/`
   - `outputs/`
   - `data/raw/`
   - `data/private/`
   - `data/processed/`
   - `*.mid`
   - `*.musicxml`
   - `*.pt`
   - `*.bin`
   - `*.safetensors`

## Criterio de aceptación

```text
make lint
make test
make package-smoke
make golden-baseline
make ai-contract-smoke
```

deben pasar antes de PR-29.
