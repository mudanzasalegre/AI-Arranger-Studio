from __future__ import annotations

import json

from model_backends.planner.ollama_planner_backend import OllamaPlannerBackend


def test_ollama_planner_backend_posts_strict_json_request():
    calls = []
    backend = OllamaPlannerBackend(
        model_name="qwen3:8b",
        base_url="http://ollama.test/api",
        timeout_seconds=12,
        client_factory=lambda **kwargs: _FakeClient(
            calls,
            {
                "message": {
                    "content": "prefix " + json.dumps(_valid_patch()) + " suffix",
                }
            },
        ),
    )

    raw = backend.generate_plan_json(
        prompt="plan hard bop",
        system_prompt="system rules",
        previous_error="missing tempo",
    )

    payload = json.loads(raw)
    assert payload["style"] == "hard_bop"
    request = calls[0]
    assert request["method"] == "POST"
    assert request["url"] == "http://ollama.test/api/chat"
    assert request["json"]["model"] == "qwen3:8b"
    assert request["json"]["format"]["type"] == "object"
    assert "Do not include note events" in request["json"]["messages"][0]["content"]
    assert "missing tempo" in request["json"]["messages"][1]["content"]


def test_ollama_planner_backend_availability_checks_model_tags():
    backend = OllamaPlannerBackend(
        model_name="qwen3:8b",
        client_factory=lambda **kwargs: _FakeClient(
            [],
            {"models": [{"name": "qwen3:8b"}]},
        ),
    )

    assert backend.is_available() is True


def test_ollama_planner_backend_falls_back_to_json_format_when_schema_fails():
    calls = []
    backend = OllamaPlannerBackend(
        model_name="qwen3:8b",
        base_url="http://ollama.test/api",
        client_factory=lambda **kwargs: _SchemaFallbackClient(calls),
    )

    raw = backend.generate_plan_json(prompt="plan hard bop", system_prompt="system rules")

    assert json.loads(raw)["style"] == "hard_bop"
    assert [call["json"]["format"] for call in calls] == [
        calls[0]["json"]["format"],
        "json",
    ]
    assert calls[0]["json"]["format"]["type"] == "object"


def test_ollama_planner_backend_unavailable_when_model_missing():
    backend = OllamaPlannerBackend(
        model_name="qwen3:8b",
        client_factory=lambda **kwargs: _FakeClient(
            [],
            {"models": [{"name": "mistral:7b"}]},
        ),
    )

    assert backend.is_available() is False
    assert "qwen3:8b" in backend.unavailable_reason


class _FakeClient:
    def __init__(self, calls, response_json):
        self.calls = calls
        self.response_json = response_json

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get(self, url):
        self.calls.append({"method": "GET", "url": url})
        return _FakeResponse(self.response_json)

    def post(self, url, *, json):
        self.calls.append({"method": "POST", "url": url, "json": json})
        return _FakeResponse(self.response_json)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _SchemaFallbackClient:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def post(self, url, *, json):
        self.calls.append({"method": "POST", "url": url, "json": json})
        if len(self.calls) == 1:
            raise RuntimeError("schema format unsupported")
        return _FakeResponse({"message": {"content": json_module_dumps(_valid_patch())}})


def json_module_dumps(value):
    return json.dumps(value)


def _valid_patch() -> dict:
    return {
        "style": "hard_bop",
        "tempo": 132,
        "meter": "4/4",
        "key": "C minor",
        "form": "minor_blues_12",
        "ensemble": "jazz_sextet",
        "instruments": ["drum_kit", "double_bass", "piano", "alto_sax"],
        "sections": [
            {
                "name": "Head",
                "start_bar": 1,
                "end_bar": 12,
                "energy": 0.7,
            }
        ],
        "generation_strategy": {
            "mode": "llm_plan",
            "priority_roles": ["melody"],
            "forbid_audio_models": True,
            "allow_note_generation": False,
            "allow_midi_export": False,
        },
    }
