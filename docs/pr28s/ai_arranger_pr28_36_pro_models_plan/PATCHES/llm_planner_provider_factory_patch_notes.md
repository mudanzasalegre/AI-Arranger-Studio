# Patch notes — connect Ollama provider to LlmPlanner

Create:

```text
packages/arranger_core/arranger_core/planning/provider_factory.py
```

Pseudo-code:

```python
from model_backends import build_model_backend_registry, load_ai_models_config, ModelBackendUnavailableError

def build_planner_provider_from_registry():
    try:
        config = load_ai_models_config()
        registry = build_model_backend_registry(
            config=config,
            include_disabled=False,
            include_unavailable=False,
        )
        return registry.get("local_llm_planner")
    except Exception:
        return None
```

Then in `apps/api/app/routes/ai_planner.py`:

```python
from arranger_core.planning.provider_factory import build_planner_provider_from_registry

provider = build_planner_provider_from_registry()
result = LlmPlanner(provider=provider).plan(...)
```

Acceptance:

When Ollama is running and `local_llm_planner.enabled=true`, endpoint response must include:

```json
{
  "planner": "llm",
  "fallback_used": false
}
```
