from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.schema import ArrangementProject, ChordSymbol, GenerationSpec
from arranger_core.song_planner import SectionPlan, SongPlan

RETRIEVAL_MODEL_VERSION = "0.1.0"
RetrievalRole = Literal[
    "melody",
    "bass",
    "walking_bass",
    "drums",
    "comping",
    "piano",
    "horns",
    "horn_response",
    "pad",
    "solo",
    "harmony",
    "unknown",
]

ROLE_ALIASES = {
    "bass": {"bass", "walking_bass", "double_bass"},
    "walking_bass": {"bass", "walking_bass", "double_bass"},
    "comping": {"comping", "piano", "keys", "guitar"},
    "piano": {"comping", "piano", "keys", "guitar"},
    "horns": {"horns", "horn_response", "brass", "horn_section"},
    "horn_response": {"horns", "horn_response", "brass", "horn_section"},
    "drums": {"drums", "drum_kit", "percussion"},
    "melody": {"melody", "lead", "solo"},
    "solo": {"solo", "melody", "lead"},
    "pad": {"pad", "strings", "synth"},
    "harmony": {"harmony", "progressions", "chords"},
    "unknown": {"unknown"},
}


class RetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalQuery(RetrievalModel):
    category: str
    role: str | None = None
    style: str = "unknown"
    instrument: str | None = None
    density: str | float | None = None
    tempo: int | None = None
    meter: str | None = None
    chord_context: list[str] = Field(default_factory=list)
    section_function: str | None = None
    tags: list[str] = Field(default_factory=list)
    min_quality: int = Field(default=1, ge=1, le=5)
    limit: int = Field(default=5, ge=1, le=100)
    seed: int = 0
    commercial_training: Literal["allowed", "forbidden", "review_required"] | None = None
    local_learning_allowed: bool = True
    max_similarity_to_source: float = Field(default=0.92, ge=0.0, le=1.0)


class RetrievalCandidate(RetrievalModel):
    pattern: dict[str, Any]
    score: float
    reasons: list[str] = Field(default_factory=list)
    transformations_applied: list[str] = Field(default_factory=list)
    similarity_to_source: float = 1.0


class PatternRetriever:
    def __init__(self, patterns: list[dict[str, Any]]) -> None:
        self.patterns = [deepcopy(pattern) for pattern in patterns if isinstance(pattern, dict)]

    @classmethod
    def from_grouped_patterns(
        cls,
        grouped_patterns: dict[str, list[dict[str, Any]]],
    ) -> PatternRetriever:
        patterns = [
            pattern
            for category_patterns in grouped_patterns.values()
            for pattern in category_patterns
            if isinstance(pattern, dict)
        ]
        return cls(patterns)

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for pattern in self.patterns:
            if not _basic_match(pattern, query):
                continue
            score, reasons = _score_pattern(pattern, query)
            candidates.append(
                RetrievalCandidate(
                    pattern=deepcopy(pattern),
                    score=score,
                    reasons=reasons,
                )
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.score,
                _stable_tie_break(item.pattern, query.seed),
                str(item.pattern.get("id", "")),
            ),
        )
        return ordered[: query.limit]

    def best(self, query: RetrievalQuery) -> RetrievalCandidate | None:
        matches = self.search(query.model_copy(update={"limit": 1}))
        return matches[0] if matches else None


class PatternAdapter:
    def adapt(
        self,
        candidate: RetrievalCandidate,
        query: RetrievalQuery,
    ) -> RetrievalCandidate:
        original_pattern = deepcopy(candidate.pattern)
        pattern = deepcopy(candidate.pattern)
        payload = pattern.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        payload = deepcopy(payload)

        transformations = list(candidate.transformations_applied)
        category = str(pattern.get("category", query.category))
        seed = _stable_seed(str(pattern.get("id", "")), query.seed)
        if category == "walking_bass_cells":
            payload, transformations = _adapt_interval_payload(
                payload,
                key="pitch_intervals",
                seed=seed,
                transformations=transformations,
                keep_first_zero=True,
            )
        elif category == "melodic_motifs":
            payload, transformations = _adapt_interval_payload(
                payload,
                key="relative_degrees",
                seed=seed,
                transformations=transformations,
                keep_first_zero=False,
            )
        elif category == "piano_voicings":
            payload, transformations = _adapt_interval_payload(
                payload,
                key="relative_notes",
                seed=seed,
                transformations=transformations,
                keep_first_zero=True,
                sort_result=True,
            )
        elif category == "drum_grooves":
            payload, transformations = _adapt_drum_payload(
                payload,
                seed=seed,
                transformations=transformations,
            )
        elif category == "horn_responses":
            payload, transformations = _adapt_interval_payload(
                payload,
                key="relative_notes",
                seed=seed,
                transformations=transformations,
                keep_first_zero=False,
            )
        elif category == "progressions":
            payload, transformations = _adapt_progression_payload(
                payload,
                seed=seed,
                transformations=transformations,
            )

        pattern["payload"] = payload
        similarity = _payload_similarity(original_pattern.get("payload", {}), payload)
        if similarity > query.max_similarity_to_source:
            payload, extra = _force_extra_variation(
                category=category,
                payload=payload,
                seed=seed,
            )
            pattern["payload"] = payload
            transformations.extend(extra)
            similarity = _payload_similarity(original_pattern.get("payload", {}), payload)

        retrieval_context = {
            "schema_version": RETRIEVAL_MODEL_VERSION,
            "query": query.model_dump(mode="json"),
            "score": candidate.score,
            "reasons": candidate.reasons,
            "transformations_applied": transformations,
            "similarity_to_source": round(similarity, 6),
            "source_pattern_id": pattern.get("id"),
            "source_file_hash": pattern.get("source_hash"),
        }
        pattern_context = dict(pattern.get("context") or {})
        pattern_context["retrieval"] = retrieval_context
        pattern_context["transformations_applied"] = transformations
        pattern_context["similarity_to_source"] = round(similarity, 6)
        pattern["context"] = pattern_context
        return RetrievalCandidate(
            pattern=pattern,
            score=candidate.score,
            reasons=candidate.reasons,
            transformations_applied=transformations,
            similarity_to_source=round(similarity, 6),
        )


def retrieve_pattern(
    context: Any,
    *,
    category: str,
    role: str,
    instrument: str | None = None,
    density: str | float | None = None,
    section_function: str | None = None,
) -> dict[str, Any] | None:
    matches = retrieve_patterns(
        context,
        category=category,
        role=role,
        instrument=instrument,
        density=density,
        section_function=section_function,
        limit=1,
    )
    return matches[0] if matches else None


def retrieve_patterns(
    context: Any,
    *,
    category: str,
    role: str,
    instrument: str | None = None,
    density: str | float | None = None,
    section_function: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    grouped = getattr(context, "learned_patterns", {})
    if not isinstance(grouped, dict) or not grouped:
        return []
    query = retrieval_query_from_context(
        context,
        category=category,
        role=role,
        instrument=instrument,
        density=density,
        section_function=section_function,
        limit=limit,
    )
    retriever = PatternRetriever.from_grouped_patterns(grouped)
    adapter = PatternAdapter()
    return [adapter.adapt(candidate, query).pattern for candidate in retriever.search(query)]


def retrieval_query_from_context(
    context: Any,
    *,
    category: str,
    role: str,
    instrument: str | None = None,
    density: str | float | None = None,
    section_function: str | None = None,
    limit: int = 5,
) -> RetrievalQuery:
    spec: GenerationSpec = context.spec
    project: ArrangementProject = context.project
    section = _section_for_query(context)
    return RetrievalQuery(
        category=category,
        role=role,
        style=spec.style,
        instrument=instrument,
        density=density if density is not None else spec.density,
        tempo=spec.tempo,
        meter=spec.meter,
        chord_context=_chord_context(project.chord_grid),
        section_function=section_function or (section.function if section else None),
        tags=list(spec.constraints.get("retrieval_tags", [])),
        min_quality=int(spec.constraints.get("retrieval_min_quality", 1) or 1),
        limit=limit,
        seed=spec.seed,
        commercial_training=spec.constraints.get("retrieval_commercial_training"),
        local_learning_allowed=bool(spec.constraints.get("retrieval_local_learning", True)),
        max_similarity_to_source=float(
            spec.constraints.get("retrieval_max_similarity", 0.92) or 0.92
        ),
    )


def retrieval_trace(pattern: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pattern:
        return None
    context = pattern.get("context")
    if not isinstance(context, dict):
        return None
    trace = context.get("retrieval")
    return trace if isinstance(trace, dict) else None


def _basic_match(pattern: dict[str, Any], query: RetrievalQuery) -> bool:
    if str(pattern.get("category", "")) != query.category:
        return False
    if query.role and not _role_matches(str(pattern.get("role", "")), query.role):
        return False
    if int(pattern.get("quality", 0) or 0) < query.min_quality:
        return False
    if pattern.get("usable_for_pattern_extraction", True) is False:
        return False
    sensitivity = _pattern_sensitivity(pattern)
    commercial_training = sensitivity.get("commercial_training")
    if query.commercial_training and commercial_training != query.commercial_training:
        return False
    if not query.local_learning_allowed and sensitivity.get("local_learning_only"):
        return False
    if query.tags and not set(query.tags).issubset(set(pattern.get("tags", []))):
        return False
    return True


def _score_pattern(pattern: dict[str, Any], query: RetrievalQuery) -> tuple[float, list[str]]:
    reasons: list[str] = []
    quality = int(pattern.get("quality", 0) or 0)
    weight = float(pattern.get("weight", 1.0) or 1.0)
    score = min(0.25, quality / 5 * 0.25) + min(0.1, weight / 5 * 0.1)
    reasons.append(f"quality:{quality}")

    pattern_role = str(pattern.get("role", ""))
    if query.role and pattern_role == query.role:
        score += 0.2
        reasons.append("role_exact")
    elif query.role and _role_matches(pattern_role, query.role):
        score += 0.16
        reasons.append("role_alias")

    pattern_style = str(pattern.get("style", "unknown"))
    if pattern_style == query.style:
        score += 0.22
        reasons.append("style_exact")
    elif pattern_style == "unknown":
        score += 0.06
        reasons.append("style_unknown")
    elif _style_family(pattern_style) == _style_family(query.style):
        score += 0.1
        reasons.append("style_family")

    context = pattern.get("context") if isinstance(pattern.get("context"), dict) else {}
    payload = pattern.get("payload") if isinstance(pattern.get("payload"), dict) else {}
    role_confidence = float(context.get("role_confidence", 1.0) or 1.0)
    score += min(0.1, max(0.0, role_confidence) * 0.1)
    if role_confidence < 0.75:
        reasons.append("role_confidence_low")
    else:
        reasons.append("role_confidence")

    if query.instrument and _instrument_matches(query.instrument, context, pattern):
        score += 0.08
        reasons.append("instrument_match")
    if query.meter and str(context.get("meter", query.meter)) == query.meter:
        score += 0.04
        reasons.append("meter_match")
    chord_score = _chord_context_score(query.chord_context, context, payload)
    if chord_score:
        score += chord_score
        reasons.append("harmonic_context")
    density_score = _density_score(query.density, context, payload)
    if density_score:
        score += density_score
        reasons.append("density_match")
    if query.section_function and context.get("section_function") == query.section_function:
        score += 0.04
        reasons.append("section_function")

    sensitivity = _pattern_sensitivity(pattern)
    if sensitivity.get("level") == "high":
        score -= 0.04
        reasons.append("sensitivity_high")
    elif sensitivity.get("level") == "low":
        score += 0.02
        reasons.append("sensitivity_low")

    return round(max(0.0, min(1.0, score)), 6), reasons


def _adapt_interval_payload(
    payload: dict[str, Any],
    *,
    key: str,
    seed: int,
    transformations: list[str],
    keep_first_zero: bool,
    sort_result: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    raw_values = payload.get(key)
    if not isinstance(raw_values, list) or len(raw_values) < 2:
        return payload, transformations
    values = [_coerce_int(value) for value in raw_values]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return payload, transformations
    adapted = list(values)
    if len(adapted) > 2:
        rotate_by = 1 + seed % (len(adapted) - 1)
        head = [0] if keep_first_zero else [adapted[0]]
        tail = adapted[1:] if keep_first_zero else adapted[1:]
        tail = tail[rotate_by % len(tail) :] + tail[: rotate_by % len(tail)]
        adapted = [*head, *tail]
        transformations.append(f"rotate_intervals:{rotate_by}")
    target_index = 1 + seed % max(1, len(adapted) - 1)
    direction = 1 if seed % 2 == 0 else -1
    adapted[target_index] = max(-12, min(18, adapted[target_index] + direction * 2))
    transformations.append(f"ornament_interval:{target_index}:{direction * 2}")
    if keep_first_zero:
        adapted[0] = 0
    if sort_result:
        adapted = [adapted[0], *sorted(dict.fromkeys(adapted[1:]))]
    payload[key] = adapted[: len(raw_values)]
    return payload, transformations


def _adapt_drum_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    transformations: list[str],
) -> tuple[dict[str, Any], list[str]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return payload, transformations
    shift = 0.5 if seed % 2 == 0 else -0.5
    adapted_events = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        copied = dict(event)
        try:
            beat = float(copied.get("beat", 0.0))
        except (TypeError, ValueError):
            beat = 0.0
        copied["beat"] = round((beat + shift) % 4.0, 3)
        adapted_events.append(copied)
    if adapted_events:
        payload["events"] = sorted(
            adapted_events,
            key=lambda item: (item["beat"], item.get("pitch", 0)),
        )
        transformations.append(f"shift_drum_onsets:{shift:+g}")
    return payload, transformations


def _adapt_progression_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    transformations: list[str],
) -> tuple[dict[str, Any], list[str]]:
    chords = payload.get("chords")
    if isinstance(chords, list) and len(chords) > 2:
        rotate_by = 1 + seed % (len(chords) - 1)
        payload["chords"] = [*chords[rotate_by:], *chords[:rotate_by]]
        transformations.append(f"rotate_progression:{rotate_by}")
    return payload, transformations


def _force_extra_variation(
    *,
    category: str,
    payload: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    if category == "drum_grooves":
        return _adapt_drum_payload(payload, seed=seed + 1, transformations=[])
    if category == "progressions":
        return _adapt_progression_payload(payload, seed=seed + 1, transformations=[])
    key = {
        "walking_bass_cells": "pitch_intervals",
        "melodic_motifs": "relative_degrees",
        "piano_voicings": "relative_notes",
        "horn_responses": "relative_notes",
    }.get(category)
    if key is None:
        return payload, []
    return _adapt_interval_payload(
        payload,
        key=key,
        seed=seed + 1,
        transformations=[],
        keep_first_zero=category in {"walking_bass_cells", "piano_voicings"},
        sort_result=category == "piano_voicings",
    )


def _role_matches(pattern_role: str, query_role: str) -> bool:
    pattern_roles = ROLE_ALIASES.get(pattern_role, {pattern_role})
    query_roles = ROLE_ALIASES.get(query_role, {query_role})
    return bool(pattern_roles & query_roles)


def _instrument_matches(
    instrument: str,
    context: dict[str, Any],
    pattern: dict[str, Any],
) -> bool:
    normalized = instrument.lower()
    candidates = {
        str(context.get("instrument_guess", "")).lower(),
        str(pattern.get("role", "")).lower(),
        *(str(tag).lower() for tag in pattern.get("tags", [])),
    }
    return any(
        normalized in candidate or candidate in normalized
        for candidate in candidates
        if candidate
    )


def _chord_context_score(
    query_chords: list[str],
    context: dict[str, Any],
    payload: dict[str, Any],
) -> float:
    pattern_chords = context.get("chord_context") or payload.get("chords") or []
    if not isinstance(pattern_chords, list) or not query_chords:
        return 0.0
    query_roots = {_chord_root(chord) for chord in query_chords}
    pattern_roots = {_chord_root(str(chord)) for chord in pattern_chords}
    if not query_roots or not pattern_roots:
        return 0.0
    overlap = len(query_roots & pattern_roots) / len(query_roots | pattern_roots)
    return round(overlap * 0.12, 6)


def _density_score(
    density: str | float | None,
    context: dict[str, Any],
    payload: dict[str, Any],
) -> float:
    if density is None:
        return 0.0
    target = _density_value(density)
    raw = context.get("density", payload.get("density"))
    if raw is None:
        return 0.0
    candidate = _density_value(raw)
    return round(max(0.0, 1.0 - abs(target - candidate)) * 0.06, 6)


def _density_value(value: str | float | int) -> float:
    if isinstance(value, int | float):
        numeric = float(value)
        return numeric / 8 if numeric > 1 else numeric
    return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(str(value), 0.55)


def _pattern_sensitivity(pattern: dict[str, Any]) -> dict[str, Any]:
    context = pattern.get("context") if isinstance(pattern.get("context"), dict) else {}
    sensitivity = context.get("pattern_sensitivity")
    if isinstance(sensitivity, dict):
        return sensitivity
    return {
        "level": "review",
        "commercial_training": pattern.get("commercial_training", "review_required"),
        "local_learning_only": pattern.get("local_learning_only", False),
    }


def _payload_similarity(left: Any, right: Any) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {f"{key}:{token}" for key, item in value.items() for token in _tokens(item)}
    if isinstance(value, list):
        return {f"{index}:{token}" for index, item in enumerate(value) for token in _tokens(item)}
    return {str(value)}


def _style_family(style: str) -> str:
    if style in {"hard_bop", "bebop", "swing", "jazz_ballad"}:
        return "swing_jazz"
    if style in {"modal_jazz", "funk_jazz"}:
        return "modern_jazz"
    return style


def _chord_context(chord_grid: list[ChordSymbol]) -> list[str]:
    return [chord.symbol for chord in chord_grid[:16] if chord.symbol]


def _chord_root(symbol: str) -> str:
    for length in (2, 1):
        root = symbol[:length]
        if root and root[0].upper() in "ABCDEFG":
            return root.upper()
    return symbol[:1].upper()


def _section_for_query(context: Any) -> SectionPlan | None:
    song_plan = getattr(context, "song_plan", None)
    if not isinstance(song_plan, SongPlan) or not song_plan.sections:
        return None
    return song_plan.sections[0]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_tie_break(pattern: dict[str, Any], seed: int) -> int:
    return _stable_seed(str(pattern.get("id", "")), seed)


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{value}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16)
