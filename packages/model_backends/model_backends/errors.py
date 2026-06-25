from __future__ import annotations


class ModelBackendError(RuntimeError):
    """Base error for model backend integration failures."""


class ModelBackendConfigurationError(ModelBackendError):
    """Raised when a model backend cannot be configured."""


class ModelBackendUnavailableError(ModelBackendError):
    """Raised when an optional backend dependency or service is unavailable."""


class ModelGenerationError(ModelBackendError):
    """Raised when a backend fails while generating a candidate artifact."""


class UnsupportedModelTaskError(ModelGenerationError):
    """Raised when a backend receives a task outside its supported capabilities."""
