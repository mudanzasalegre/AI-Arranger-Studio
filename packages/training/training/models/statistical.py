from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from training.datasets.tokenized_dataset import (
    TOKENIZATION_ROLES,
    TRAINING_SPLITS,
    TokenizedSegment,
    load_tokenized_segments,
)

BASELINE_ROLE_MODEL_TYPES = {
    "melody": "ngram_melody",
    "bass": "ngram_bass",
    "piano_comping": "markov_voicings",
    "horn_responses": "pattern_probability_model",
    "drums": "drum_fill_retrieval",
}

StatisticalModelType = Literal[
    "ngram_melody",
    "ngram_bass",
    "markov_voicings",
    "drum_fill_retrieval",
    "pattern_probability_model",
]


class StatisticalModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatisticalRoleModelArtifact(StatisticalModelBase):
    schema_version: str = "0.1.0"
    model_type: StatisticalModelType
    role: str
    order: int
    seed: int
    trained_at: str
    segment_count: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    vocabulary: list[str] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    context_counts: dict[str, int] = Field(default_factory=dict)
    next_token_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    start_context: list[str] = Field(default_factory=list)
    representative_patterns: list[dict[str, Any]] = Field(default_factory=list)
    pattern_probabilities: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatisticalBaselineSummary(StatisticalModelBase):
    schema_version: str = "0.1.0"
    generated_at: str
    seed: int
    ngram_order: int
    source_segments_path: str
    total_segments: int
    model_count: int
    model_paths: dict[str, str]
    pattern_probability_model_path: str
    comparison_report_path: str
    summary_path: str
    role_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)


class StatisticalRoleModel:
    def __init__(self, artifact: StatisticalRoleModelArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def load(cls, path: str | Path) -> StatisticalRoleModel:
        return cls(
            StatisticalRoleModelArtifact.model_validate_json(
                Path(path).read_text(encoding="utf-8")
            )
        )

    def generate(
        self,
        *,
        seed: int,
        max_tokens: int = 64,
        prefix: list[str] | None = None,
    ) -> list[str]:
        tokens = list(prefix or self.artifact.start_context)
        if not tokens:
            tokens = ["BOS"]
        while len(tokens) < max_tokens:
            context = _context_key(tokens[-max(1, self.artifact.order - 1) :])
            choices = self.artifact.next_token_counts.get(context)
            if not choices:
                choices = self.artifact.next_token_counts.get(_context_key(["BOS"]))
            if not choices:
                break
            next_token = _weighted_choice(choices, seed=seed, step=len(tokens), context=context)
            tokens.append(next_token)
            if next_token == "EOS":
                break
        return tokens[:max_tokens]

    def score(self, tokens: list[str]) -> dict[str, float]:
        return _score_tokens(self.artifact, tokens)


def train_baseline_statistical_models(
    tokenized_segments: str | Path | list[TokenizedSegment],
    output_dir: str | Path,
    *,
    seed: int = 1800,
    ngram_order: int = 3,
) -> StatisticalBaselineSummary:
    segments_path = ""
    if isinstance(tokenized_segments, str | Path):
        segments_path = str(tokenized_segments)
        segments = load_tokenized_segments(tokenized_segments)
    else:
        segments = list(tokenized_segments)
        segments_path = "<in-memory>"

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    role_models: dict[str, StatisticalRoleModelArtifact] = {}
    model_paths: dict[str, str] = {}
    generated_at = datetime.now(UTC).isoformat()
    for role in TOKENIZATION_ROLES:
        role_segments = [segment for segment in segments if segment.role == role]
        artifact = _train_role_model(
            role_segments,
            role=role,
            model_type=BASELINE_ROLE_MODEL_TYPES[role],
            order=_role_order(role, ngram_order),
            seed=seed,
            trained_at=generated_at,
        )
        role_models[role] = artifact
        model_path = models_dir / f"{role}_{artifact.model_type}.json"
        model_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
        model_paths[role] = str(model_path)

    pattern_probability_artifact = _train_pattern_probability_model(
        segments,
        seed=seed,
        trained_at=generated_at,
    )
    pattern_probability_model_path = models_dir / "pattern_probability_model.json"
    pattern_probability_model_path.write_text(
        pattern_probability_artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    comparison_path = output_path / "baseline_comparison.json"
    comparison = _comparison_report(segments, role_models)
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    role_counts = Counter(segment.role for segment in segments)
    split_counts = Counter(segment.split for segment in segments)
    summary_path = output_path / "statistical_training_summary.json"
    summary = StatisticalBaselineSummary(
        generated_at=generated_at,
        seed=seed,
        ngram_order=ngram_order,
        source_segments_path=segments_path,
        total_segments=len(segments),
        model_count=len(role_models) + 1,
        model_paths=model_paths,
        pattern_probability_model_path=str(pattern_probability_model_path),
        comparison_report_path=str(comparison_path),
        summary_path=str(summary_path),
        role_counts={role: role_counts.get(role, 0) for role in TOKENIZATION_ROLES},
        split_counts={split: split_counts.get(split, 0) for split in TRAINING_SPLITS},
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def _train_role_model(
    segments: list[TokenizedSegment],
    *,
    role: str,
    model_type: str,
    order: int,
    seed: int,
    trained_at: str,
) -> StatisticalRoleModelArtifact:
    train_segments = [segment for segment in segments if segment.split == "train"] or segments
    vocabulary = sorted({token for segment in train_segments for token in segment.tokens})
    token_counts = Counter(token for segment in train_segments for token in segment.tokens)
    context_counts: Counter[str] = Counter()
    next_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    start_context = ["BOS"]

    for segment in train_segments:
        tokens = segment.tokens or ["BOS", "EOS"]
        if tokens:
            start_context = tokens[: max(1, order - 1)]
        padded = ["BOS"] * max(1, order - 1) + tokens + ["EOS"]
        for index in range(max(1, order - 1), len(padded)):
            context = _context_key(padded[index - max(1, order - 1) : index])
            next_token = padded[index]
            context_counts[context] += 1
            next_token_counts[context][next_token] += 1

    return StatisticalRoleModelArtifact(
        model_type=model_type,  # type: ignore[arg-type]
        role=role,
        order=order,
        seed=seed,
        trained_at=trained_at,
        segment_count=len(segments),
        split_counts={
            split: sum(1 for segment in segments if segment.split == split)
            for split in TRAINING_SPLITS
        },
        vocabulary=vocabulary,
        token_counts=dict(sorted(token_counts.items())),
        context_counts=dict(sorted(context_counts.items())),
        next_token_counts={
            context: dict(sorted(counts.items()))
            for context, counts in sorted(next_token_counts.items())
        },
        start_context=start_context,
        representative_patterns=_representative_patterns(train_segments),
        pattern_probabilities=_pattern_probabilities(train_segments),
        metadata={
            "style_counts": dict(sorted(Counter(segment.style for segment in segments).items())),
            "license_counts": dict(
                sorted(Counter(segment.license for segment in segments).items())
            ),
            "source_dataset_counts": dict(
                sorted(Counter(segment.source_dataset for segment in segments).items())
            ),
            "training_split": "train" if any(s.split == "train" for s in segments) else "all",
        },
    )


def _train_pattern_probability_model(
    segments: list[TokenizedSegment],
    *,
    seed: int,
    trained_at: str,
) -> StatisticalRoleModelArtifact:
    pattern_counts = Counter(segment.source_pattern_id for segment in segments)
    total = sum(pattern_counts.values()) or 1
    role_counts = Counter(segment.role for segment in segments)
    return StatisticalRoleModelArtifact(
        model_type="pattern_probability_model",
        role="all_roles",
        order=1,
        seed=seed,
        trained_at=trained_at,
        segment_count=len(segments),
        split_counts={
            split: sum(1 for segment in segments if segment.split == split)
            for split in TRAINING_SPLITS
        },
        vocabulary=sorted(pattern_counts),
        token_counts=dict(sorted(pattern_counts.items())),
        context_counts={"PATTERN": total},
        next_token_counts={"PATTERN": dict(sorted(pattern_counts.items()))},
        start_context=["PATTERN"],
        representative_patterns=_representative_patterns(segments),
        pattern_probabilities={
            pattern_id: round(count / total, 8)
            for pattern_id, count in sorted(pattern_counts.items())
        },
        metadata={
            "role_counts": {role: role_counts.get(role, 0) for role in TOKENIZATION_ROLES},
            "purpose": "global_pattern_probability_baseline",
        },
    )


def _comparison_report(
    segments: list[TokenizedSegment],
    role_models: dict[str, StatisticalRoleModelArtifact],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "comparison_targets": ["statistical", "rule_based_proxy", "retrieval_proxy"],
        "roles": {},
    }
    for role, artifact in role_models.items():
        role_segments = [segment for segment in segments if segment.role == role]
        eval_segments = [
            segment for segment in role_segments if segment.split in {"val", "test"}
        ] or role_segments
        statistical_scores = [_score_tokens(artifact, segment.tokens) for segment in eval_segments]
        vocabulary_size = max(1, len(artifact.vocabulary))
        token_count = sum(len(segment.tokens) for segment in eval_segments)
        train_segments = [
            segment for segment in role_segments if segment.split == "train"
        ] or role_segments
        retrieval_scores = [
            _best_jaccard(segment.tokens, [candidate.tokens for candidate in train_segments])
            for segment in eval_segments
        ]
        statistical_perplexity = _mean(
            [score["perplexity"] for score in statistical_scores if score["token_count"] > 0]
        )
        rule_based_perplexity = float(vocabulary_size)
        report["roles"][role] = {
            "model_type": artifact.model_type,
            "segment_count": len(role_segments),
            "evaluation_segment_count": len(eval_segments),
            "statistical": {
                "token_count": token_count,
                "perplexity": round(statistical_perplexity, 6),
                "coverage": round(_coverage(artifact, eval_segments), 6),
            },
            "rule_based_proxy": {
                "token_count": token_count,
                "perplexity": round(rule_based_perplexity, 6),
                "coverage": 1.0 if vocabulary_size else 0.0,
                "description": "Uniform vocabulary baseline for rule-based comparison.",
            },
            "retrieval_proxy": {
                "mean_best_jaccard": round(_mean(retrieval_scores), 6),
                "exact_match_rate": round(
                    _mean([1.0 if score >= 0.999 else 0.0 for score in retrieval_scores]),
                    6,
                ),
                "description": "Best token-set similarity against train segments.",
            },
            "winner_by_perplexity": (
                "statistical"
                if statistical_perplexity <= rule_based_perplexity
                else "rule_based_proxy"
            ),
        }
    return report


def _score_tokens(artifact: StatisticalRoleModelArtifact, tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {"token_count": 0.0, "negative_log_likelihood": 0.0, "perplexity": 1.0}
    context_size = max(1, artifact.order - 1)
    padded = ["BOS"] * context_size + tokens + ["EOS"]
    vocabulary_size = max(1, len(artifact.vocabulary))
    nll = 0.0
    count = 0
    for index in range(context_size, len(padded)):
        context = _context_key(padded[index - context_size : index])
        next_token = padded[index]
        next_counts = artifact.next_token_counts.get(context, {})
        numerator = next_counts.get(next_token, 0) + 1
        denominator = artifact.context_counts.get(context, 0) + vocabulary_size
        nll -= math.log(numerator / denominator)
        count += 1
    avg_nll = nll / count if count else 0.0
    return {
        "token_count": float(count),
        "negative_log_likelihood": round(avg_nll, 8),
        "perplexity": round(math.exp(avg_nll), 8),
    }


def _role_order(role: str, requested_order: int) -> int:
    if role in {"piano_comping", "drums", "horn_responses"}:
        return max(2, min(requested_order, 2))
    return max(2, requested_order)


def _context_key(tokens: list[str]) -> str:
    return "\u241f".join(tokens)


def _weighted_choice(
    choices: dict[str, int],
    *,
    seed: int,
    step: int,
    context: str,
) -> str:
    ordered = sorted((token, max(0, count)) for token, count in choices.items() if count > 0)
    if not ordered:
        return "EOS"
    total = sum(count for _, count in ordered)
    digest = hashlib.sha256(f"{seed}:{step}:{context}".encode()).hexdigest()
    ticket = int(digest[:12], 16) % total
    cumulative = 0
    for token, count in ordered:
        cumulative += count
        if ticket < cumulative:
            return token
    return ordered[-1][0]


def _representative_patterns(
    segments: list[TokenizedSegment],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    ranked = sorted(
        segments,
        key=lambda segment: (-segment.quality, segment.source_pattern_id, segment.id),
    )
    return [
        {
            "segment_id": segment.id,
            "source_pattern_id": segment.source_pattern_id,
            "style": segment.style,
            "source_hash": segment.source_hash,
            "license": segment.license,
            "token_count": segment.token_count,
            "fingerprint": segment.fingerprint,
        }
        for segment in ranked[:limit]
    ]


def _pattern_probabilities(segments: list[TokenizedSegment]) -> dict[str, float]:
    counts = Counter(segment.source_pattern_id for segment in segments)
    total = sum(counts.values()) or 1
    return {
        pattern_id: round(count / total, 8)
        for pattern_id, count in sorted(counts.items())
    }


def _coverage(
    artifact: StatisticalRoleModelArtifact,
    segments: list[TokenizedSegment],
) -> float:
    vocabulary = set(artifact.vocabulary)
    tokens = [token for segment in segments for token in segment.tokens]
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if token in vocabulary) / len(tokens)


def _best_jaccard(tokens: list[str], references: list[list[str]]) -> float:
    if not references:
        return 0.0
    token_set = set(tokens)
    if not token_set:
        return 0.0
    best = 0.0
    for reference in references:
        reference_set = set(reference)
        union = token_set | reference_set
        score = len(token_set & reference_set) / len(union) if union else 0.0
        best = max(best, score)
    return best


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
