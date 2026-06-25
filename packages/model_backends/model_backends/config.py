from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_backends.base import CommercialUse, ModelCapabilities, ModelTask
from model_backends.errors import ModelBackendConfigurationError
from model_backends.registry import ModelBackendRegistry

AI_MODELS_CONFIG_ENV = "AI_MODELS_CONFIG"


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    type: str = "symbolic"
    adapter: str
    model_name: str | None = None
    output_dir: str | None = None
    commercial_use: CommercialUse = "unknown"
    dependency_mode: str = "required"
    install_hint: str | None = None
    tasks: list[ModelTask] = Field(default_factory=list)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    def adapter_kwargs(self, backend_id: str, default_output_dir: str | None) -> dict[str, Any]:
        kwargs = dict(self.model_extra or {})
        kwargs["backend_id"] = backend_id
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name
        if self.output_dir is not None:
            kwargs["output_dir"] = self.output_dir
        elif default_output_dir is not None:
            kwargs["output_dir"] = default_output_dir
        if self.install_hint is not None:
            kwargs["install_hint"] = self.install_hint
        return kwargs


class AIModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


def default_ai_models_config_path() -> Path:
    configured = os.environ.get(AI_MODELS_CONFIG_ENV)
    if configured:
        return Path(configured).expanduser()

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.extend(parent / "configs" / "ai_models.yaml" for parent in (cwd, *cwd.parents))
    module_path = Path(__file__).resolve()
    candidates.extend(parent / "configs" / "ai_models.yaml" for parent in module_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd / "configs" / "ai_models.yaml"


def load_ai_models_config(path: str | Path | None = None) -> AIModelsConfig:
    config_path = Path(path).expanduser() if path is not None else default_ai_models_config_path()
    if not config_path.exists():
        raise ModelBackendConfigurationError(f"AI models config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ModelBackendConfigurationError(f"AI models config must be a mapping: {config_path}")
    return AIModelsConfig.model_validate(raw)


def build_model_backend_registry(
    *,
    config: AIModelsConfig | None = None,
    config_path: str | Path | None = None,
    include_disabled: bool = False,
    include_unavailable: bool = True,
) -> ModelBackendRegistry:
    ai_config = config or load_ai_models_config(config_path)
    registry = ModelBackendRegistry()
    default_output_dir = ai_config.settings.get("artifact_raw_dir", "outputs/model_artifacts/raw")

    for backend_id, backend_config in ai_config.backends.items():
        if not backend_config.enabled:
            if include_disabled:
                registry.register_configured(
                    backend_id=backend_id,
                    status="disabled",
                    backend_type=backend_config.type,
                    enabled=False,
                    adapter=backend_config.adapter,
                    capabilities=backend_config.capabilities,
                    tasks=backend_config.tasks,
                    commercial_use=backend_config.commercial_use,
                    install_hint=backend_config.install_hint,
                )
            continue

        try:
            backend_class = _import_adapter(backend_config.adapter)
            backend = backend_class(
                **backend_config.adapter_kwargs(
                    backend_id=backend_id,
                    default_output_dir=str(default_output_dir),
                )
            )
            status = "available"
            error = None
            is_available = getattr(backend, "is_available", None)
            if callable(is_available) and not bool(is_available()):
                status = "unavailable"
                error = (
                    getattr(backend, "unavailable_reason", None)
                    or "optional dependency missing"
                )
                if not include_unavailable:
                    continue
            registry.register(
                backend,
                status=status,
                backend_type=backend_config.type,
                enabled=True,
                adapter=backend_config.adapter,
                tasks=backend_config.tasks,
                commercial_use=backend_config.commercial_use,
                install_hint=backend_config.install_hint,
                error=error,
            )
        except Exception as exc:
            if not include_unavailable:
                continue
            registry.register_configured(
                backend_id=backend_id,
                status="unavailable",
                backend_type=backend_config.type,
                enabled=True,
                adapter=backend_config.adapter,
                capabilities=backend_config.capabilities,
                tasks=backend_config.tasks,
                commercial_use=backend_config.commercial_use,
                install_hint=backend_config.install_hint,
                error=str(exc),
            )

    return registry


def _import_adapter(adapter_path: str) -> type[Any]:
    if "." not in adapter_path:
        raise ModelBackendConfigurationError(f"Invalid adapter path: {adapter_path}")
    module_name, class_name = adapter_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    try:
        adapter = getattr(module, class_name)
    except AttributeError as exc:
        raise ModelBackendConfigurationError(f"Adapter class not found: {adapter_path}") from exc
    if not isinstance(adapter, type):
        raise ModelBackendConfigurationError(f"Adapter is not a class: {adapter_path}")
    return adapter
