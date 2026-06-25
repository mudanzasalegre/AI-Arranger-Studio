# 18 — Mapa de archivos y stubs esperados

## Nuevos archivos principales

```text
packages/model_backends/pyproject.toml
packages/model_backends/model_backends/__init__.py
packages/model_backends/model_backends/base.py
packages/model_backends/model_backends/registry.py
packages/model_backends/model_backends/config.py
packages/model_backends/model_backends/errors.py
packages/model_backends/model_backends/artifact.py
packages/model_backends/model_backends/symbolic/__init__.py
packages/model_backends/model_backends/symbolic/mock_backend.py
packages/model_backends/model_backends/symbolic/midigpt_backend.py
packages/model_backends/model_backends/symbolic/text2midi_backend.py
```

## Nuevos archivos en arranger_core

```text
packages/arranger_core/arranger_core/ai/__init__.py
packages/arranger_core/arranger_core/ai/validation_gate.py
packages/arranger_core/arranger_core/ai/artifact_importer.py
packages/arranger_core/arranger_core/merge/model_artifact_merger.py
packages/arranger_core/arranger_core/takes/__init__.py
packages/arranger_core/arranger_core/takes/models.py
packages/arranger_core/arranger_core/takes/manager.py
packages/arranger_core/arranger_core/planning/llm/__init__.py
packages/arranger_core/arranger_core/planning/llm/schemas.py
packages/arranger_core/arranger_core/planning/llm/planner.py
packages/arranger_core/arranger_core/planning/llm/repair.py
packages/arranger_core/arranger_core/planning/llm/fallback.py
```

## Nuevos endpoints API

```text
apps/api/app/routes/ai_models.py
apps/api/app/routes/ai_planner.py
apps/api/app/routes/ai_generation.py
apps/api/app/routes/takes.py
```

## Config

```text
configs/ai_models.yaml
```

## Outputs

```text
outputs/model_artifacts/raw/.gitkeep
outputs/model_artifacts/imported/.gitkeep
outputs/model_artifacts/rejected/.gitkeep
outputs/model_artifacts/validated/.gitkeep
```

## Tests

```text
tests/model_backends/test_registry.py
tests/model_backends/test_mock_backend.py
tests/model_backends/test_missing_dependency.py
tests/ai/test_artifact_quarantine.py
tests/ai/test_project_merger.py
tests/ai/test_validation_gate.py
tests/ai/test_takes.py
tests/api/test_ai_models_endpoint.py
tests/api/test_ai_infill_mock.py
tests/api/test_text2midi_sketch_mock.py
tests/planning/test_llm_planner_json.py
```

## Config ejemplo

Ver:

```text
examples/ai_models.yaml
```
