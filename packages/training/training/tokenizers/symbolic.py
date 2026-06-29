from __future__ import annotations

import re
from typing import Any

from dataset_tools import ExtractedPattern

_TOKEN_SAFE = re.compile(r"[^A-Za-z0-9_.:+/-]+")


class MidiTokBridgeTokenizer:
    """Deterministic symbolic tokenizer used until a real MidiTok pipeline is wired in."""

    name = "symbolic_miditok_bridge"
    version = "0.1.0"

    def encode_pattern(self, pattern: ExtractedPattern, *, role: str) -> list[str]:
        tokens = [
            "BOS",
            f"ROLE={_safe(role)}",
            f"CATEGORY={_safe(pattern.category)}",
            f"STYLE={_safe(pattern.style)}",
            f"QUALITY={pattern.quality}",
        ]
        for tag in sorted(pattern.tags):
            tokens.append(f"TAG={_safe(tag)}")
        tokens.extend(_context_chord_tokens(pattern))
        tokens.extend(_flatten_tokens("CTX", pattern.context))
        tokens.extend(_flatten_tokens("PAYLOAD", pattern.payload))
        tokens.append("EOS")
        return tokens


def build_miditok_bridge_config(*, roles: list[str] | tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "tokenizer_family": "symbolic_pattern_miditok_bridge",
        "tokenizer_status": "config_only_no_miditok_dependency",
        "compatible_target": "MidiTok",
        "roles": list(roles),
        "pitch_range": [21, 108],
        "beat_resolution": {"0_4": 8, "4_12": 4},
        "max_bar_embedding": 256,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "token_fields": [
            "role",
            "category",
            "style",
            "quality",
            "tags",
            "chord_context",
            "section_context",
            "payload",
        ],
        "notes": [
            "This file is a stable bridge config for future MidiTok integration.",
            "No heavy tokenizer dependency is imported by the exporter.",
        ],
    }


def _context_chord_tokens(pattern: ExtractedPattern) -> list[str]:
    chords = pattern.payload.get("chords")
    if not isinstance(chords, list):
        chords = pattern.context.get("chord_context")
    if not isinstance(chords, list):
        return []
    return [f"CHORD={_safe(chord)}" for chord in chords]


def _flatten_tokens(prefix: str, value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            tokens.extend(_flatten_tokens(f"{prefix}.{_safe(key)}", value[key]))
        return tokens
    if isinstance(value, list):
        for index, item in enumerate(value):
            tokens.extend(_flatten_tokens(f"{prefix}[{index}]", item))
        return tokens
    tokens.append(f"{prefix}={_safe(value)}")
    return tokens


def _safe(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "none"
    return _TOKEN_SAFE.sub("_", text)
