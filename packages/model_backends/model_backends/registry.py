from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from model_backends.base import CommercialUse, ModelCapabilities, ModelTask, MusicModelBackend
from model_backends.errors import (
    ModelBackendConfigurationError,
    ModelBackendUnavailableError,
)

BackendStatus = Literal["available", "unavailable", "disabled"]


class RegisteredModelBackend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    backend_type: str = "symbolic"
    enabled: bool = True
    status: BackendStatus = "available"
    adapter: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    tasks: list[ModelTask] = Field(default_factory=list)
    commercial_use: CommercialUse = "unknown"
    install_hint: str | None = None
    error: str | None = None


class ModelBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, MusicModelBackend] = {}
        self._records: dict[str, RegisteredModelBackend] = {}

    def register(
        self,
        backend: MusicModelBackend,
        *,
        status: BackendStatus = "available",
        backend_type: str = "symbolic",
        enabled: bool = True,
        adapter: str | None = None,
        tasks: list[ModelTask] | None = None,
        commercial_use: CommercialUse | None = None,
        install_hint: str | None = None,
        error: str | None = None,
    ) -> None:
        backend_id = backend.backend_id
        if backend_id in self._records:
            raise ModelBackendConfigurationError(f"Duplicate model backend id: {backend_id}")
        self._backends[backend_id] = backend
        self._records[backend_id] = RegisteredModelBackend(
            id=backend_id,
            backend_type=backend_type,
            enabled=enabled,
            status=status,
            adapter=adapter,
            capabilities=backend.capabilities,
            tasks=tasks or [],
            commercial_use=commercial_use or backend.capabilities.commercial_use,
            install_hint=install_hint,
            error=error,
        )

    def register_configured(
        self,
        *,
        backend_id: str,
        status: BackendStatus,
        backend_type: str,
        enabled: bool,
        adapter: str | None,
        capabilities: ModelCapabilities | None = None,
        tasks: list[ModelTask] | None = None,
        commercial_use: CommercialUse = "unknown",
        install_hint: str | None = None,
        error: str | None = None,
    ) -> None:
        if backend_id in self._records:
            raise ModelBackendConfigurationError(f"Duplicate model backend id: {backend_id}")
        self._records[backend_id] = RegisteredModelBackend(
            id=backend_id,
            backend_type=backend_type,
            enabled=enabled,
            status=status,
            adapter=adapter,
            capabilities=capabilities or ModelCapabilities(commercial_use=commercial_use),
            tasks=tasks or [],
            commercial_use=commercial_use,
            install_hint=install_hint,
            error=error,
        )

    def get(self, backend_id: str) -> MusicModelBackend:
        if backend_id not in self._records:
            raise KeyError(f"Model backend not registered: {backend_id}")
        record = self._records[backend_id]
        if record.status == "disabled":
            raise ModelBackendUnavailableError(f"Model backend disabled: {backend_id}")
        if record.status == "unavailable":
            detail = f": {record.error}" if record.error else ""
            raise ModelBackendUnavailableError(f"Model backend unavailable: {backend_id}{detail}")
        return self._backends[backend_id]

    def list(self, *, include_disabled: bool = True) -> list[dict[str, object]]:
        records = [
            record
            for record in self._records.values()
            if include_disabled or record.status != "disabled"
        ]
        records.sort(key=lambda item: item.id)
        return [record.model_dump(mode="json") for record in records]

    def ids(self, *, include_disabled: bool = True) -> list[str]:
        return [
            str(item["id"])
            for item in self.list(include_disabled=include_disabled)
        ]
