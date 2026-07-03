from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CUSTOM_ROLE_NGRAM_VERSION = "0.1.0"
CUSTOM_ROLE_NGRAM_MODEL_TYPE = "custom_role_ngram"
CUSTOM_ROLE_TRAINING_ROLES: tuple[str, ...] = (
    "melody",
    "walking_bass",
    "piano_comping",
    "horn_responses",
    "drums",
)
CUSTOM_ROLE_MODEL_IDS = {
    "melody": "jazz_melody_v001",
    "walking_bass": "jazz_walking_bass_v001",
    "piano_comping": "jazz_piano_comping_v001",
    "horn_responses": "jazz_horn_responses_v001",
    "drums": "jazz_drums_v001",
}
CUSTOM_ROLE_CHECKPOINT_DIRS = {
    "melody": "melody",
    "walking_bass": "bass",
    "piano_comping": "piano_comping",
    "horn_responses": "horns",
    "drums": "drums",
}
ROLE_ALIASES = {
    "bass": "walking_bass",
    "double_bass": "walking_bass",
    "horn_response": "horn_responses",
    "horns": "horn_responses",
    "piano": "piano_comping",
    "comping": "piano_comping",
}
TRAINING_SPLITS = ("train", "val", "test")
BLOCKED_TRAINING_LICENSES = {
    "",
    "unknown",
    "proprietary",
    "all rights reserved",
    "all-rights-reserved",
    "private",
    "research_only",
    "research-only",
    "research only",
    "non_commercial",
    "non-commercial",
    "noncommercial",
    "cc-by-nc",
    "cc-by-nc-sa",
}
BLOCKED_COMMERCIAL_FLAGS = {
    "blocked",
    "forbidden",
    "not_allowed",
    "research_only",
    "research-only",
    "non_commercial",
    "non-commercial",
}

DatasetSplit = Literal["train", "val", "test"]


class RoleNgramModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleTrainingSegment(RoleNgramModelBase):
    id: str
    role: str
    split: DatasetSplit = "train"
    tokens: list[str]
    style: str = "unknown"
    source_file_id: str = "unknown"
    source_path: str = ""
    source_hash: str = ""
    source_dataset: str = "unknown"
    license: str = "unknown"
    commercial_training: str = "review_required"
    train_eligible: bool = True
    quality: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoleNgramCheckpoint(RoleNgramModelBase):
    schema_version: str = CUSTOM_ROLE_NGRAM_VERSION
    model_type: str = CUSTOM_ROLE_NGRAM_MODEL_TYPE
    role: str
    model_id: str
    order: int
    seed: int
    trained_at: str
    segment_count: int
    train_segment_count: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    vocabulary: list[str] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    context_counts: dict[str, int] = Field(default_factory=dict)
    next_token_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    start_context: list[str] = Field(default_factory=list)
    representative_sequences: list[list[str]] = Field(default_factory=list)
    retrieval_index: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustomRoleCheckpointRecord(RoleNgramModelBase):
    role: str
    model_id: str
    checkpoint_dir: str
    model_path: str
    tokenizer_path: str
    config_path: str
    training_manifest_path: str
    license_report_path: str
    metrics_path: str
    segment_count: int
    vocabulary_size: int
    commercial_allowed: bool


class CustomRoleNgramTrainingSummary(RoleNgramModelBase):
    schema_version: str = CUSTOM_ROLE_NGRAM_VERSION
    generated_at: str
    seed: int
    ngram_order: int
    source_segments_path: str
    checkpoint_root: str
    roles: list[str]
    total_segments: int
    rejected_segments: int
    checkpoints: dict[str, CustomRoleCheckpointRecord]
    summary_path: str


class RoleNgramModel:
    def __init__(self, checkpoint: RoleNgramCheckpoint) -> None:
        self.checkpoint = checkpoint

    @classmethod
    def load(cls, path: str | Path) -> RoleNgramModel:
        return cls(RoleNgramCheckpoint.model_validate_json(Path(path).read_text(encoding="utf-8")))

    def generate(
        self,
        *,
        seed: int,
        max_tokens: int = 64,
        prefix: list[str] | None = None,
    ) -> list[str]:
        tokens = list(prefix or self.checkpoint.start_context or ["BOS"])
        context_size = max(1, self.checkpoint.order - 1)
        while len(tokens) < max_tokens:
            context = _context_key(tokens[-context_size:])
            choices = self.checkpoint.next_token_counts.get(context)
            if not choices:
                choices = self.checkpoint.next_token_counts.get(_context_key(["BOS"]))
            if not choices:
                choices = self.checkpoint.next_token_counts.get(_context_key(["BOS", "BOS"]))
            if not choices:
                break
            next_token = _weighted_choice(choices, seed=seed, step=len(tokens), context=context)
            tokens.append(next_token)
            if next_token == "EOS":
                break
        if tokens[-1:] != ["EOS"] and len(tokens) < max_tokens:
            tokens.append("EOS")
        return tokens[:max_tokens]

    def score(self, tokens: list[str]) -> dict[str, float]:
        return _score_tokens(self.checkpoint, tokens)


def train_custom_role_ngram_checkpoints(
    tokenized_segments: str | Path | list[RoleTrainingSegment] | list[dict[str, Any]],
    checkpoint_root: str | Path,
    *,
    seed: int = 3400,
    ngram_order: int = 3,
    summary_path: str | Path | None = None,
    clean: bool = True,
) -> CustomRoleNgramTrainingSummary:
    segments_path = (
        str(tokenized_segments)
        if isinstance(tokenized_segments, str | Path)
        else "<in-memory>"
    )
    all_segments = load_role_training_segments(tokenized_segments)
    eligible, rejected = _eligible_training_segments(all_segments)
    generated_at = datetime.now(UTC).isoformat()
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    checkpoints: dict[str, CustomRoleCheckpointRecord] = {}
    for role in CUSTOM_ROLE_TRAINING_ROLES:
        role_segments = [segment for segment in eligible if segment.role == role]
        if not role_segments:
            raise ValueError(f"No eligible training segments for custom role: {role}")
        model_id = CUSTOM_ROLE_MODEL_IDS[role]
        checkpoint_dir = root / CUSTOM_ROLE_CHECKPOINT_DIRS[role] / model_id
        if clean:
            _clean_checkpoint_dir(checkpoint_dir, root=root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = _train_role_checkpoint(
            role_segments,
            role=role,
            model_id=model_id,
            order=_role_order(role, ngram_order),
            seed=seed,
            trained_at=generated_at,
        )
        record = _write_checkpoint(
            checkpoint,
            checkpoint_dir,
            source_segments_path=segments_path,
            generated_at=generated_at,
            rejected_segments=rejected,
        )
        checkpoints[role] = record

    output_summary_path = (
        Path(summary_path) if summary_path else root / "custom_role_ngram_summary.json"
    )
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = CustomRoleNgramTrainingSummary(
        generated_at=generated_at,
        seed=seed,
        ngram_order=ngram_order,
        source_segments_path=segments_path,
        checkpoint_root=str(root),
        roles=list(CUSTOM_ROLE_TRAINING_ROLES),
        total_segments=len(eligible),
        rejected_segments=len(rejected),
        checkpoints=checkpoints,
        summary_path=str(output_summary_path),
    )
    output_summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def load_role_training_segments(
    tokenized_segments: str | Path | list[RoleTrainingSegment] | list[dict[str, Any]],
) -> list[RoleTrainingSegment]:
    if isinstance(tokenized_segments, str | Path):
        payloads = [
            json.loads(line)
            for line in Path(tokenized_segments).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payloads = [
            segment.model_dump(mode="json") if isinstance(segment, RoleTrainingSegment) else segment
            for segment in tokenized_segments
        ]
    return [_segment_from_payload(payload) for payload in payloads]


def canonical_custom_training_role(role: str) -> str:
    normalized = role.strip().lower().replace("-", "_").replace(" ", "_")
    return ROLE_ALIASES.get(normalized, normalized)


def checkpoint_dir_for_role(checkpoint_root: str | Path, role: str) -> Path:
    canonical_role = canonical_custom_training_role(role)
    return (
        Path(checkpoint_root)
        / CUSTOM_ROLE_CHECKPOINT_DIRS[canonical_role]
        / CUSTOM_ROLE_MODEL_IDS[canonical_role]
    )


def _segment_from_payload(payload: dict[str, Any]) -> RoleTrainingSegment:
    role = canonical_custom_training_role(str(payload.get("role", "")))
    tokens = _tokens_from_payload(payload)
    split = str(payload.get("split") or "train")
    if split not in TRAINING_SPLITS:
        split = "train"
    return RoleTrainingSegment(
        id=str(payload.get("id") or _stable_hash(json.dumps(payload, sort_keys=True))[:16]),
        role=role,
        split=split,  # type: ignore[arg-type]
        tokens=tokens,
        style=str(payload.get("style") or "unknown"),
        source_file_id=str(payload.get("source_file_id") or "unknown"),
        source_path=str(payload.get("source_path") or ""),
        source_hash=str(payload.get("source_hash") or ""),
        source_dataset=str(payload.get("source_dataset") or "unknown"),
        license=str(payload.get("license") or "unknown"),
        commercial_training=str(payload.get("commercial_training") or "review_required"),
        train_eligible=bool(payload.get("train_eligible", True)),
        quality=int(payload.get("quality") or 3),
        metadata=dict(payload.get("metadata") or {}),
    )


def _tokens_from_payload(payload: dict[str, Any]) -> list[str]:
    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        return [str(token) for token in tokens]
    token_sequences = payload.get("token_sequences")
    if isinstance(token_sequences, list):
        flattened = [
            str(token)
            for sequence in token_sequences
            if isinstance(sequence, list)
            for token in sequence
        ]
        if flattened:
            if flattened[0] != "BOS":
                flattened.insert(0, "BOS")
            if flattened[-1] != "EOS":
                flattened.append("EOS")
            return flattened
    return ["BOS", "EOS"]


def _eligible_training_segments(
    segments: list[RoleTrainingSegment],
) -> tuple[list[RoleTrainingSegment], list[RoleTrainingSegment]]:
    eligible: list[RoleTrainingSegment] = []
    rejected: list[RoleTrainingSegment] = []
    for segment in segments:
        if segment.role not in CUSTOM_ROLE_TRAINING_ROLES:
            rejected.append(segment)
            continue
        if not segment.train_eligible:
            rejected.append(segment)
            continue
        if segment.split not in TRAINING_SPLITS:
            rejected.append(segment)
            continue
        if _blocked_training(segment.license, segment.commercial_training):
            rejected.append(segment)
            continue
        if len(segment.tokens) < 2:
            rejected.append(segment)
            continue
        eligible.append(segment)
    return eligible, rejected


def _train_role_checkpoint(
    segments: list[RoleTrainingSegment],
    *,
    role: str,
    model_id: str,
    order: int,
    seed: int,
    trained_at: str,
) -> RoleNgramCheckpoint:
    train_segments = [segment for segment in segments if segment.split == "train"] or segments
    vocabulary = sorted({token for segment in train_segments for token in segment.tokens})
    token_counts = Counter(token for segment in train_segments for token in segment.tokens)
    context_counts: Counter[str] = Counter()
    next_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_size = max(1, order - 1)
    start_context = ["BOS"] * context_size

    for segment in train_segments:
        tokens = _clean_training_tokens(segment.tokens)
        if tokens:
            start_context = tokens[:context_size]
        padded = ["BOS"] * context_size + tokens + ["EOS"]
        for index in range(context_size, len(padded)):
            context = _context_key(padded[index - context_size : index])
            next_token = padded[index]
            context_counts[context] += 1
            next_token_counts[context][next_token] += 1

    metrics = _role_metrics(
        segments,
        context_counts=dict(context_counts),
        next_token_counts={context: dict(counts) for context, counts in next_token_counts.items()},
        vocabulary=vocabulary,
        order=order,
    )
    split_counts = Counter(segment.split for segment in segments)
    return RoleNgramCheckpoint(
        role=role,
        model_id=model_id,
        order=order,
        seed=seed,
        trained_at=trained_at,
        segment_count=len(segments),
        train_segment_count=len(train_segments),
        split_counts={split: split_counts.get(split, 0) for split in TRAINING_SPLITS},
        vocabulary=vocabulary,
        token_counts=dict(sorted(token_counts.items())),
        context_counts=dict(sorted(context_counts.items())),
        next_token_counts={
            context: dict(sorted(counts.items()))
            for context, counts in sorted(next_token_counts.items())
        },
        start_context=start_context,
        representative_sequences=[
            _clean_training_tokens(segment.tokens)[:48]
            for segment in sorted(
                train_segments,
                key=lambda item: (-item.quality, item.source_file_id, item.id),
            )[:8]
        ],
        retrieval_index=[
            {
                "id": segment.id,
                "split": segment.split,
                "style": segment.style,
                "source_file_id": segment.source_file_id,
                "source_hash": segment.source_hash,
                "license": segment.license,
                "commercial_training": segment.commercial_training,
                "train_eligible": segment.train_eligible,
                "token_count": len(segment.tokens),
                "tokens": _clean_training_tokens(segment.tokens)[:96],
            }
            for segment in train_segments[:24]
        ],
        metrics=metrics,
        metadata={
            "style_counts": dict(sorted(Counter(segment.style for segment in segments).items())),
            "source_dataset_counts": dict(
                sorted(Counter(segment.source_dataset for segment in segments).items())
            ),
            "license_counts": dict(
                sorted(Counter(segment.license for segment in segments).items())
            ),
        },
    )


def _write_checkpoint(
    checkpoint: RoleNgramCheckpoint,
    checkpoint_dir: Path,
    *,
    source_segments_path: str,
    generated_at: str,
    rejected_segments: list[RoleTrainingSegment],
) -> CustomRoleCheckpointRecord:
    model_path = checkpoint_dir / "model.json"
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    config_path = checkpoint_dir / "config.yaml"
    training_manifest_path = checkpoint_dir / "training_manifest.yaml"
    license_report_path = checkpoint_dir / "license_report.json"
    metrics_path = checkpoint_dir / "metrics.json"

    model_path.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tokenizer_path.write_text(
        json.dumps(
            {
                "schema_version": CUSTOM_ROLE_NGRAM_VERSION,
                "tokenizer": "ai_arranger_symbolic_role_tokens",
                "role": checkpoint.role,
                "model_id": checkpoint.model_id,
                "vocabulary": checkpoint.vocabulary,
                "special_tokens": ["BOS", "EOS", "PAD", "MASK"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": CUSTOM_ROLE_NGRAM_VERSION,
                "role": checkpoint.role,
                "model": {
                    "role": checkpoint.role,
                    "model_id": checkpoint.model_id,
                    "model_type": CUSTOM_ROLE_NGRAM_MODEL_TYPE,
                    "ngram_order": checkpoint.order,
                    "backend_adapter": (
                        "model_backends.custom_role.statistical_backend."
                        "StatisticalCustomRoleBackend"
                    ),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sources = _source_entries(checkpoint)
    training_manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": CUSTOM_ROLE_NGRAM_VERSION,
                "role": checkpoint.role,
                "model_id": checkpoint.model_id,
                "model_type": CUSTOM_ROLE_NGRAM_MODEL_TYPE,
                "generated_at": generated_at,
                "source_segments_path": source_segments_path,
                "datasets": _dataset_entries(sources),
                "segments": sources,
                "split_counts": checkpoint.split_counts,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    license_report = {
        "schema_version": CUSTOM_ROLE_NGRAM_VERSION,
        "status": "pass",
        "role": checkpoint.role,
        "model_id": checkpoint.model_id,
        "sources": sources,
        "segments": sources,
        "rejected_sources": [
            {
                "id": segment.id,
                "role": segment.role,
                "source_file_id": segment.source_file_id,
                "license": segment.license,
                "commercial_training": segment.commercial_training,
                "reason": "not_eligible_for_custom_role_training",
            }
            for segment in rejected_segments
            if segment.role == checkpoint.role
        ],
        "errors": [],
    }
    license_report_path.write_text(json.dumps(license_report, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(checkpoint.metrics, indent=2) + "\n", encoding="utf-8")
    return CustomRoleCheckpointRecord(
        role=checkpoint.role,
        model_id=checkpoint.model_id,
        checkpoint_dir=str(checkpoint_dir),
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        config_path=str(config_path),
        training_manifest_path=str(training_manifest_path),
        license_report_path=str(license_report_path),
        metrics_path=str(metrics_path),
        segment_count=checkpoint.segment_count,
        vocabulary_size=len(checkpoint.vocabulary),
        commercial_allowed=True,
    )


def _source_entries(checkpoint: RoleNgramCheckpoint) -> list[dict[str, Any]]:
    entries = []
    for item in checkpoint.retrieval_index:
        entries.append(
            {
                "segment_id": item["id"],
                "split": item["split"],
                "source_file_id": item["source_file_id"],
                "source_hash": item["source_hash"],
                "license": item["license"],
                "commercial_training": item["commercial_training"],
                "train_eligible": bool(item["train_eligible"]),
                "training_allowed": True,
            }
        )
    return entries


def _dataset_entries(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source["source_file_id"])
        grouped[source_id] = {
            "dataset_id": source_id,
            "source_file_id": source_id,
            "license": source["license"],
            "commercial_training": source["commercial_training"],
            "train_eligible": True,
            "training_allowed": True,
        }
    return list(grouped.values())


def _clean_checkpoint_dir(checkpoint_dir: Path, *, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_checkpoint = checkpoint_dir.resolve()
    if resolved_checkpoint == resolved_root or resolved_root not in resolved_checkpoint.parents:
        raise RuntimeError(f"Refusing to clean checkpoint outside root: {checkpoint_dir}")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def _clean_training_tokens(tokens: list[str]) -> list[str]:
    output = [token for token in tokens if token not in {"PAD", "MASK"}]
    if not output:
        return ["BOS", "EOS"]
    if output[0] != "BOS":
        output.insert(0, "BOS")
    if output[-1] != "EOS":
        output.append("EOS")
    return output


def _role_metrics(
    segments: list[RoleTrainingSegment],
    *,
    context_counts: dict[str, int],
    next_token_counts: dict[str, dict[str, int]],
    vocabulary: list[str],
    order: int,
) -> dict[str, Any]:
    checkpoint = RoleNgramCheckpoint(
        role=segments[0].role,
        model_id=CUSTOM_ROLE_MODEL_IDS[segments[0].role],
        order=order,
        seed=0,
        trained_at=datetime.now(UTC).isoformat(),
        segment_count=len(segments),
        train_segment_count=sum(1 for segment in segments if segment.split == "train"),
        vocabulary=vocabulary,
        context_counts=context_counts,
        next_token_counts=next_token_counts,
        start_context=["BOS"] * max(1, order - 1),
    )
    eval_segments = [
        segment for segment in segments if segment.split in {"val", "test"}
    ] or segments
    scores = [RoleNgramModel(checkpoint).score(segment.tokens) for segment in eval_segments]
    perplexities = [score["perplexity"] for score in scores if score["token_count"] > 0]
    split_counts = Counter(segment.split for segment in segments)
    return {
        "schema_version": CUSTOM_ROLE_NGRAM_VERSION,
        "role": segments[0].role,
        "model_type": CUSTOM_ROLE_NGRAM_MODEL_TYPE,
        "segment_count": len(segments),
        "split_counts": {split: split_counts.get(split, 0) for split in TRAINING_SPLITS},
        "vocabulary_size": len(vocabulary),
        "token_count": sum(len(segment.tokens) for segment in segments),
        "evaluation_segment_count": len(eval_segments),
        "evaluation_perplexity": round(sum(perplexities) / len(perplexities), 6)
        if perplexities
        else 0.0,
        "coverage": round(_coverage(vocabulary, eval_segments), 6),
    }


def _score_tokens(checkpoint: RoleNgramCheckpoint, tokens: list[str]) -> dict[str, float]:
    tokens = _clean_training_tokens(tokens)
    context_size = max(1, checkpoint.order - 1)
    padded = ["BOS"] * context_size + tokens + ["EOS"]
    vocabulary_size = max(1, len(checkpoint.vocabulary))
    nll = 0.0
    count = 0
    for index in range(context_size, len(padded)):
        context = _context_key(padded[index - context_size : index])
        next_token = padded[index]
        next_counts = checkpoint.next_token_counts.get(context, {})
        numerator = next_counts.get(next_token, 0) + 1
        denominator = checkpoint.context_counts.get(context, 0) + vocabulary_size
        nll -= math.log(numerator / denominator)
        count += 1
    avg_nll = nll / count if count else 0.0
    return {
        "token_count": float(count),
        "negative_log_likelihood": round(avg_nll, 8),
        "perplexity": round(math.exp(avg_nll), 8),
    }


def _coverage(vocabulary: list[str], segments: list[RoleTrainingSegment]) -> float:
    vocab = set(vocabulary)
    tokens = [token for segment in segments for token in segment.tokens]
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if token in vocab) / len(tokens)


def _role_order(role: str, requested_order: int) -> int:
    if role in {"piano_comping", "horn_responses", "drums"}:
        return max(2, min(requested_order, 2))
    return max(2, requested_order)


def _blocked_training(license_name: str, commercial_training: str) -> bool:
    return (
        _normalize_flag(license_name) in BLOCKED_TRAINING_LICENSES
        or _normalize_flag(commercial_training) in BLOCKED_COMMERCIAL_FLAGS
    )


def _weighted_choice(
    choices: dict[str, int],
    *,
    seed: int,
    step: int,
    context: str,
) -> str:
    ordered = sorted((token, count) for token, count in choices.items() if count > 0)
    if not ordered:
        return "EOS"
    total = sum(count for _, count in ordered)
    ticket = int(_stable_hash(f"{seed}:{step}:{context}")[:12], 16) % total
    cumulative = 0
    for token, count in ordered:
        cumulative += count
        if ticket < cumulative:
            return token
    return ordered[-1][0]


def _context_key(tokens: list[str]) -> str:
    return "\u241f".join(tokens)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_flag(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
