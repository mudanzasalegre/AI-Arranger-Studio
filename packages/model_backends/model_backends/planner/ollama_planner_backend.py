from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Callable
from typing import Any

from model_backends.base import ModelCapabilities
from model_backends.errors import ModelBackendUnavailableError, ModelGenerationError

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.S)
_STYLE_IDS = ["hard_bop", "jazz_ballad", "bossa_nova", "funk_jazz", "jazz_waltz"]
_FORM_IDS = ["minor_blues_12", "blues_12", "rhythm_changes_32", "song_form_32"]
_ENSEMBLE_IDS = ["jazz_sextet", "jazz_quartet", "jazz_trio", "sax_quartet"]
_INSTRUMENT_IDS = [
    "drum_kit",
    "double_bass",
    "piano",
    "alto_sax",
    "tenor_sax",
    "trumpet_bflat",
    "trombone",
    "clarinet_bflat",
    "flute",
]
_ROLE_IDS = ["drums", "walking_bass", "comping", "melody", "horn_response"]


class OllamaPlannerBackend:
    backend_id = "local_llm_planner"
    backend_version = "0.1.0"
    unavailable_reason = "Ollama planner is not configured"
    capabilities = ModelCapabilities(
        symbolic_midi=False,
        multitrack=False,
        bar_infill=False,
        track_generation=False,
        text_prompt=True,
        json_planning=True,
        token_output=False,
        supports_training=False,
        commercial_use="review_required",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.2,
        num_ctx: int = 4096,
        use_json_schema: bool = True,
        install_hint: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name or os.environ.get("OLLAMA_PLANNER_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            or os.environ.get("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120")
        )
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.use_json_schema = bool(use_json_schema)
        self.install_hint = install_hint or (
            "Install Ollama, run `ollama pull qwen3:8b`, and keep `ollama serve` running."
        )
        self._client_factory = client_factory

    def is_available(self) -> bool:
        if importlib.util.find_spec("httpx") is None:
            self.unavailable_reason = "Ollama planner dependency missing: python module httpx"
            return False
        try:
            models = self._list_models()
        except Exception as exc:
            self.unavailable_reason = f"Ollama planner is unavailable: {exc}"
            return False
        if models and self.model_name not in models:
            self.unavailable_reason = (
                f"Ollama model {self.model_name!r} is not installed. "
                f"Available models: {sorted(models)}"
            )
            return False
        self.unavailable_reason = ""
        return True

    def generate_plan_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        previous_error: str | None = None,
    ) -> str:
        if importlib.util.find_spec("httpx") is None:
            raise ModelBackendUnavailableError(
                f"Missing httpx. Install hint: {self.install_hint}"
            )
        messages = [
            {"role": "system", "content": _strict_system_prompt(system_prompt)},
            {
                "role": "user",
                "content": _user_prompt(prompt, previous_error=previous_error),
            },
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": _response_format(self.use_json_schema),
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            data = self._post_json("/chat", payload)
        except Exception as exc:
            if self.use_json_schema:
                fallback_payload = {**payload, "format": "json"}
                try:
                    data = self._post_json("/chat", fallback_payload)
                except Exception as fallback_exc:
                    raise ModelBackendUnavailableError(
                        f"Ollama planner request failed: {fallback_exc}. "
                        f"Install hint: {self.install_hint}"
                    ) from fallback_exc
            else:
                raise ModelBackendUnavailableError(
                    f"Ollama planner request failed: {exc}. Install hint: {self.install_hint}"
                ) from exc

        content = _response_content(data)
        if not content:
            raise ModelGenerationError("Ollama planner returned an empty response")
        return _extract_json_text(content)

    def _list_models(self) -> set[str]:
        data = self._get_json("/tags")
        models = data.get("models", [])
        if not isinstance(models, list):
            return set()
        names = {
            str(item.get("name"))
            for item in models
            if isinstance(item, dict) and item.get("name")
        }
        return names

    def _get_json(self, path: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(self.base_url + path)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ModelGenerationError("Ollama returned a non-object JSON response")
        return data

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(self.base_url + path, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ModelGenerationError("Ollama returned a non-object JSON response")
        return data

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory(timeout=self.timeout_seconds)
        import httpx

        return httpx.Client(timeout=self.timeout_seconds)


def _strict_system_prompt(system_prompt: str) -> str:
    return (
        f"{system_prompt}\n"
        "Return a single JSON object only. Do not wrap it in markdown. "
        "Do not include note events, pitches, MIDI paths, audio requests, or export requests.\n"
        "Use these exact top-level keys: schema_version, style, substyle, tempo, meter, "
        "key, form, ensemble, instruments, sections, generation_strategy, role_intents.\n"
        "Each section must use: name, start_bar, end_bar, energy, density_by_role, "
        "groove_feel, role_focus. Do not use SongPlan, SectionPlan, PhrasePlan, "
        "GrooveMap, RoleIntent, or GenerationStrategy as top-level keys.\n"
        "Use catalog ids such as hard_bop, minor_blues_12, jazz_sextet, drum_kit, "
        "double_bass, piano, alto_sax, trumpet_bflat, trombone."
    )


def _user_prompt(prompt: str, *, previous_error: str | None) -> str:
    repair = ""
    if previous_error:
        repair = (
            "\nThe previous response failed validation. Fix only the JSON planning "
            f"object. Validation error: {previous_error}"
        )
    return f"{prompt.strip() or 'Create a valid symbolic song plan.'}{repair}"


def _response_content(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    if data.get("response"):
        return str(data["response"])
    return ""


def _extract_json_text(content: str) -> str:
    text = content.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise ModelGenerationError(
                "Ollama planner response did not contain JSON"
            ) from None
        candidate = match.group(0)
        json.loads(candidate)
        return candidate


def _response_format(use_json_schema: bool) -> str | dict[str, Any]:
    if not use_json_schema:
        return "json"
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "style",
            "tempo",
            "meter",
            "key",
            "form",
            "ensemble",
            "instruments",
            "sections",
            "generation_strategy",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "style": {"type": "string", "enum": _STYLE_IDS},
            "substyle": {"type": ["string", "null"]},
            "tempo": {"type": "integer", "minimum": 40, "maximum": 260},
            "meter": {"type": "string"},
            "key": {"type": "string"},
            "form": {"type": "string", "enum": _FORM_IDS},
            "ensemble": {"type": "string", "enum": _ENSEMBLE_IDS},
            "instruments": {
                "type": "array",
                "items": {"type": "string", "enum": _INSTRUMENT_IDS},
            },
            "sections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "start_bar", "end_bar", "energy"],
                    "properties": {
                        "name": {"type": "string"},
                        "start_bar": {"type": "integer", "minimum": 1},
                        "end_bar": {"type": "integer", "minimum": 1},
                        "energy": {"type": "number", "minimum": 0, "maximum": 1},
                        "density_by_role": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "groove_feel": {"type": ["string", "null"]},
                        "role_focus": {
                            "type": "array",
                            "items": {"type": "string", "enum": _ROLE_IDS},
                        },
                        "notes": {"type": ["string", "null"]},
                    },
                },
            },
            "generation_strategy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mode",
                    "priority_roles",
                    "forbid_audio_models",
                    "allow_note_generation",
                    "allow_midi_export",
                ],
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": [
                            "rule_based",
                            "llm_plan",
                            "hybrid_symbolic",
                            "retrieval_ready",
                        ],
                    },
                    "priority_roles": {
                        "type": "array",
                        "items": {"type": "string", "enum": _ROLE_IDS},
                    },
                    "role_intents": {
                        "type": "array",
                        "items": _role_intent_schema(),
                    },
                    "forbid_audio_models": {"type": "boolean"},
                    "allow_note_generation": {"type": "boolean"},
                    "allow_midi_export": {"type": "boolean"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
            },
            "role_intents": {
                "type": "array",
                "items": _role_intent_schema(),
            },
        },
    }


def _role_intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "density"],
        "properties": {
            "role": {"type": "string", "enum": _ROLE_IDS},
            "instruments": {
                "type": "array",
                "items": {"type": "string", "enum": _INSTRUMENT_IDS},
            },
            "target_sections": {"type": "array", "items": {"type": "string"}},
            "density": {"type": "number", "minimum": 0, "maximum": 1},
            "strategy": {"type": ["string", "null"]},
            "allowed_operations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "plan_song",
                        "patch_plan",
                        "build_role_intent",
                        "choose_generation_strategy",
                    ],
                },
            },
            "constraints": {"type": "object", "additionalProperties": True},
        },
    }
