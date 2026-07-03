from __future__ import annotations

from arranger_core.planning.llm_planner import PlannerJsonProvider

LOCAL_LLM_PLANNER_BACKEND_ID = "local_llm_planner"


def build_planner_provider_from_registry() -> PlannerJsonProvider | None:
    try:
        from model_backends import build_model_backend_registry, load_ai_models_config
        from model_backends.errors import (
            ModelBackendConfigurationError,
            ModelBackendUnavailableError,
        )
    except ImportError:
        return None

    try:
        config = load_ai_models_config()
    except ModelBackendConfigurationError:
        return None

    backend_config = config.backends.get(LOCAL_LLM_PLANNER_BACKEND_ID)
    if backend_config is None or not backend_config.enabled:
        return None

    try:
        registry = build_model_backend_registry(
            config=config,
            include_disabled=False,
            include_unavailable=True,
        )
        return registry.get(LOCAL_LLM_PLANNER_BACKEND_ID)
    except (KeyError, ModelBackendConfigurationError, ModelBackendUnavailableError):
        return None
