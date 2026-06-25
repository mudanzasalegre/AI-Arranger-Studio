from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arranger_core import (
    ChordParser,
    DeterministicWalkingBassBackend,
    ModelRequest,
    ModelResponse,
    midi_to_note,
    note_to_midi,
    note_token,
)
from dataset_tools import DatasetSplitSummary, ExtractedPattern, PatternIndex

SYMBOLIC_MODEL_VERSION = "0.1.0"


class SymbolicPatternModelBackend:
    name = "symbolic-pattern-model"

    def __init__(self, model: dict[str, Any], *, model_path: str | Path | None = None) -> None:
        self.model = model
        self.model_path = str(model_path) if model_path is not None else None
        self.version = str(model.get("model_version", SYMBOLIC_MODEL_VERSION))
        self._fallback = DeterministicWalkingBassBackend()
        self._chord_parser = ChordParser.load_default()

    @classmethod
    def load(cls, path: str | Path) -> SymbolicPatternModelBackend:
        model_path = Path(path)
        return cls(
            json.loads(model_path.read_text(encoding="utf-8")),
            model_path=model_path,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.role != "walking_bass":
            return self._fallback.generate(request)

        cells = _patterns_for_request(self.model, "walking_bass_cells", request.style)
        if not cells:
            return self._fallback.generate(request)

        bar_count = int(request.controls.get("bar_count") or len(request.chord_context) or 1)
        beats_by_bar = request.controls.get("beats_per_bar", [])
        chords = request.chord_context or ["Cm7"]
        anchor = note_to_midi("C2")
        tokens: list[str] = []
        for bar_number in range(1, bar_count + 1):
            cell = cells[(request.seed + bar_number - 1) % len(cells)]
            intervals = _pattern_intervals(cell)
            if not intervals:
                intervals = [0, 3, 7, 10]
            chord_symbol = chords[(bar_number - 1) % len(chords)]
            root_pc = _root_pc(chord_symbol, self._chord_parser)
            beat_count = _beat_count_for_bar(beats_by_bar, bar_number)
            for beat_index, interval in enumerate(_fit_intervals(intervals, beat_count)):
                pitch, anchor = _nearest_bass_pitch((root_pc + interval) % 12, anchor)
                tokens.append(
                    note_token(
                        bar=bar_number,
                        start=float(beat_index),
                        duration=1.0,
                        pitch=pitch,
                        velocity=74 if beat_index == 0 else 68,
                    )
                )

        return ModelResponse(
            role=request.role,
            target_tokens=tokens,
            backend_name=self.name,
            backend_version=self.version,
            metadata={
                "model_path": self.model_path,
                "model_name": self.model.get("model_name", self.name),
                "trained_at": self.model.get("trained_at"),
                "source_roots": self.model.get("source_roots", []),
                "pattern_count": self.model.get("pattern_count", 0),
            },
        )


def train_symbolic_pattern_model(
    pattern_index: PatternIndex,
    output_path: str | Path,
    *,
    source_roots: list[str],
    training_summary: DatasetSplitSummary,
    model_name: str = "jazzvar-release2-symbolic",
    max_patterns_per_category: int = 1000,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    patterns_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in sorted({pattern.category for pattern in pattern_index.patterns}):
        category_patterns = [
            pattern
            for pattern in pattern_index.patterns
            if pattern.category == category
            and pattern.usable_for_training
            and pattern.license.strip().lower()
            not in {"", "unknown", "proprietary", "all rights reserved"}
        ]
        category_patterns.sort(
            key=lambda item: (-item.weight, -item.quality, item.fingerprint, item.id)
        )
        patterns_by_category[category] = [
            _compact_pattern(pattern)
            for pattern in category_patterns[:max_patterns_per_category]
        ]

    category_counts = Counter(pattern.category for pattern in pattern_index.patterns)
    role_counts = Counter(pattern.role for pattern in pattern_index.patterns)
    model = {
        "schema_version": "0.1.0",
        "model_type": "symbolic_pattern_backend",
        "model_name": model_name,
        "model_version": SYMBOLIC_MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "source_roots": source_roots,
        "pattern_count": len(pattern_index.patterns),
        "stored_pattern_count": sum(len(items) for items in patterns_by_category.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "training_summary": training_summary.model_dump(mode="json"),
        "patterns_by_category": patterns_by_category,
        "metadata": metadata or {},
    }
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model


def _compact_pattern(pattern: ExtractedPattern) -> dict[str, Any]:
    return {
        "id": pattern.id,
        "category": pattern.category,
        "role": pattern.role,
        "style": pattern.style,
        "quality": pattern.quality,
        "weight": pattern.weight,
        "source_file_id": pattern.source_file_id,
        "source_hash": pattern.source_hash,
        "license": pattern.license,
        "tags": pattern.tags,
        "context": pattern.context,
        "payload": pattern.payload,
        "fingerprint": pattern.fingerprint,
    }


def _patterns_for_request(
    model: dict[str, Any],
    category: str,
    style: str,
) -> list[dict[str, Any]]:
    patterns_by_category = model.get("patterns_by_category", {})
    raw_patterns = patterns_by_category.get(category, [])
    if not isinstance(raw_patterns, list):
        return []
    candidates = [pattern for pattern in raw_patterns if isinstance(pattern, dict)]
    style_matches = [
        pattern
        for pattern in candidates
        if pattern.get("style") in {style, "jazz", "unknown"}
    ]
    selected = style_matches or candidates
    return sorted(
        selected,
        key=lambda item: (
            -float(item.get("weight", 0.0) or 0.0),
            -int(item.get("quality", 0) or 0),
            str(item.get("id", "")),
        ),
    )


def _pattern_intervals(pattern: dict[str, Any]) -> list[int]:
    payload = pattern.get("payload")
    if not isinstance(payload, dict):
        return []
    raw_intervals = payload.get("pitch_intervals")
    if not isinstance(raw_intervals, list):
        return []
    intervals: list[int] = []
    for value in raw_intervals:
        try:
            intervals.append(int(value))
        except (TypeError, ValueError):
            continue
    return intervals


def _fit_intervals(intervals: list[int], beat_count: int) -> list[int]:
    defaults = [0, 3, 7, 10]
    fitted = [0, *intervals[1:beat_count]]
    fitted.extend(defaults[len(fitted) : beat_count])
    return fitted[:beat_count]


def _beat_count_for_bar(raw_beats_by_bar: Any, bar_number: int) -> int:
    if isinstance(raw_beats_by_bar, list) and bar_number <= len(raw_beats_by_bar):
        try:
            return max(1, round(float(raw_beats_by_bar[bar_number - 1])))
        except (TypeError, ValueError):
            return 4
    return 4


def _root_pc(chord_symbol: str, parser: ChordParser) -> int:
    try:
        return parser.parse(chord_symbol).root_pc
    except ValueError:
        return 0


def _nearest_bass_pitch(target_pc: int, anchor_midi: int) -> tuple[str, int]:
    low = note_to_midi("E1")
    high = note_to_midi("C4")
    candidates = [
        midi_note
        for midi_note in range(low, high + 1)
        if midi_note % 12 == target_pc % 12
    ]
    selected = min(candidates, key=lambda midi_note: abs(midi_note - anchor_midi))
    return midi_to_note(selected, prefer_sharps=False), selected
