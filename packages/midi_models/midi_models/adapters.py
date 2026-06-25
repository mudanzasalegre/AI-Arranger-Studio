from __future__ import annotations

from arranger_core import ModelRequest, ModelResponse


class MidiTokBackendAdapter:
    name = "miditok-unconfigured"
    version = "0.1.0"

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError(
            "MidiTok backend is a placeholder. Install and configure MidiTok before use."
        )


class ExternalModelBackendAdapter:
    name = "external-model-unconfigured"
    version = "0.1.0"

    def __init__(self, *, endpoint: str | None = None, model_name: str | None = None) -> None:
        self.endpoint = endpoint
        self.model_name = model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError(
            "External model backend is a placeholder. Configure a model endpoint before use."
        )
