from __future__ import annotations

from typing import Any

from model_backends.base import (
    ModelCapabilities,
    ModelGenerationRequest,
    ModelGenerationResult,
)


class CustomRoleModelBackend:
    """Minimal interface for locally trained single-role model backends."""

    backend_id: str
    role: str
    capabilities: ModelCapabilities

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        raise NotImplementedError

    @property
    def registry_metadata(self) -> dict[str, Any]:
        return {"role": self.role}
