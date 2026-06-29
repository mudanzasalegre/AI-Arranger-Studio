from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dataset_tools import ExtractedPattern, PatternIndex
from pydantic import BaseModel, ConfigDict, Field

from training.tokenizers.symbolic import MidiTokBridgeTokenizer, build_miditok_bridge_config

TOKENIZATION_ROLES: tuple[str, ...] = (
    "melody",
    "bass",
    "piano_comping",
    "horn_responses",
    "drums",
)
TRAINING_SPLITS: tuple[str, ...] = ("train", "val", "test")
BLOCKED_TRAINING_LICENSES = {
    "",
    "unknown",
    "proprietary",
    "all rights reserved",
    "all-rights-reserved",
    "private",
}

RoleName = Literal["melody", "bass", "piano_comping", "horn_responses", "drums"]
SplitName = Literal["train", "val", "test"]

_CATEGORY_ROLE_MAP = {
    "melodic_motifs": "melody",
    "walking_bass_cells": "bass",
    "piano_voicings": "piano_comping",
    "horn_responses": "horn_responses",
    "drum_grooves": "drums",
}
_ROLE_ALIASES = {
    "lead": "melody",
    "melody": "melody",
    "solo": "melody",
    "bass": "bass",
    "double_bass": "bass",
    "walking_bass": "bass",
    "comping": "piano_comping",
    "piano": "piano_comping",
    "piano_comping": "piano_comping",
    "keys": "piano_comping",
    "horn": "horn_responses",
    "horn_response": "horn_responses",
    "horn_responses": "horn_responses",
    "horns": "horn_responses",
    "drum": "drums",
    "drums": "drums",
    "drum_kit": "drums",
}


class TrainingBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizedSegment(TrainingBaseModel):
    id: str
    split: SplitName = "train"
    role: RoleName
    style: str
    source_pattern_id: str
    source_file_id: str
    source_path: str
    source_hash: str
    source_dataset: str
    license: str
    commercial_training: str = "review_required"
    local_learning_only: bool = False
    quality: int
    tags: list[str] = Field(default_factory=list)
    chord_context: list[str] = Field(default_factory=list)
    section_context: dict[str, Any] = Field(default_factory=dict)
    tokens: list[str] = Field(default_factory=list)
    token_count: int
    fingerprint: str
    no_memorization_fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenizedSegmentMetadata(TrainingBaseModel):
    id: str
    split: SplitName
    role: RoleName
    style: str
    source_pattern_id: str
    source_file_id: str
    source_hash: str
    source_dataset: str
    license: str
    commercial_training: str
    local_learning_only: bool
    quality: int
    tags: list[str] = Field(default_factory=list)
    chord_context: list[str] = Field(default_factory=list)
    section_context: dict[str, Any] = Field(default_factory=dict)
    token_count: int
    fingerprint: str
    no_memorization_fingerprint: str


class TokenizedDatasetSummary(TrainingBaseModel):
    schema_version: str = "0.1.0"
    generated_at: str
    seed: int
    min_quality: int
    roles: list[str]
    total_patterns: int
    total_segments: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    skipped_not_training_allowed: int = 0
    skipped_blocked_license: int = 0
    skipped_unsupported_role: int = 0
    tokenized_segments_path: str
    metadata_path: str
    split_manifest_path: str
    miditok_config_path: str
    role_manifest_path: str
    summary_path: str


def export_tokenized_dataset(
    pattern_index: PatternIndex | str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    min_quality: int = 3,
    roles: list[str] | tuple[str, ...] = TOKENIZATION_ROLES,
    tokenizer: MidiTokBridgeTokenizer | None = None,
) -> TokenizedDatasetSummary:
    index = (
        PatternIndex.load_json(pattern_index)
        if isinstance(pattern_index, str | Path)
        else pattern_index
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_roles = tuple(role for role in roles if role in TOKENIZATION_ROLES)
    selected_role_set = set(selected_roles)
    tokenizer = tokenizer or MidiTokBridgeTokenizer()

    raw_segments: list[TokenizedSegment] = []
    skipped_not_training_allowed = 0
    skipped_blocked_license = 0
    skipped_unsupported_role = 0

    for pattern in index.patterns:
        if pattern.quality < min_quality or not pattern.usable_for_training:
            skipped_not_training_allowed += 1
            continue
        if _blocked_license(pattern.license):
            skipped_blocked_license += 1
            continue
        role = _canonical_role(pattern)
        if role not in selected_role_set:
            skipped_unsupported_role += 1
            continue
        raw_segments.append(_segment_for_pattern(pattern, role=role, tokenizer=tokenizer))

    segments = _assign_splits(raw_segments, seed=seed)
    metadata = [_metadata_for_segment(segment) for segment in segments]

    tokenized_segments_path = output_path / "tokenized_segments.jsonl"
    with tokenized_segments_path.open("w", encoding="utf-8") as file:
        for segment in segments:
            file.write(segment.model_dump_json() + "\n")

    metadata_path = output_path / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as file:
        for item in metadata:
            file.write(item.model_dump_json() + "\n")

    split_manifest_path = output_path / "dataset_splits.json"
    split_manifest = _split_manifest(segments, seed=seed)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    miditok_config_path = output_path / "miditok_config.json"
    miditok_config_path.write_text(
        json.dumps(build_miditok_bridge_config(roles=selected_roles), indent=2) + "\n",
        encoding="utf-8",
    )

    role_manifest_path = output_path / "role_manifest.json"
    role_manifest_path.write_text(
        json.dumps(_role_manifest(segments, selected_roles), indent=2) + "\n",
        encoding="utf-8",
    )

    role_counts = Counter(segment.role for segment in segments)
    split_counts = Counter(segment.split for segment in segments)
    summary_path = output_path / "training_summary.json"
    summary = TokenizedDatasetSummary(
        generated_at=datetime.now(UTC).isoformat(),
        seed=seed,
        min_quality=min_quality,
        roles=list(selected_roles),
        total_patterns=len(index.patterns),
        total_segments=len(segments),
        role_counts={role: role_counts.get(role, 0) for role in selected_roles},
        split_counts={split: split_counts.get(split, 0) for split in TRAINING_SPLITS},
        skipped_not_training_allowed=skipped_not_training_allowed,
        skipped_blocked_license=skipped_blocked_license,
        skipped_unsupported_role=skipped_unsupported_role,
        tokenized_segments_path=str(tokenized_segments_path),
        metadata_path=str(metadata_path),
        split_manifest_path=str(split_manifest_path),
        miditok_config_path=str(miditok_config_path),
        role_manifest_path=str(role_manifest_path),
        summary_path=str(summary_path),
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def load_tokenized_segments(path: str | Path) -> list[TokenizedSegment]:
    segments: list[TokenizedSegment] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            segments.append(TokenizedSegment.model_validate_json(line))
    return segments


def _segment_for_pattern(
    pattern: ExtractedPattern,
    *,
    role: str,
    tokenizer: MidiTokBridgeTokenizer,
) -> TokenizedSegment:
    tokens = tokenizer.encode_pattern(pattern, role=role)
    sensitivity = _pattern_sensitivity(pattern)
    return TokenizedSegment(
        id=f"segment_{role}_{_stable_hash(pattern.fingerprint or pattern.id)[:16]}",
        role=role,  # type: ignore[arg-type]
        style=pattern.style,
        source_pattern_id=pattern.id,
        source_file_id=pattern.source_file_id,
        source_path=pattern.source_path,
        source_hash=pattern.source_hash or pattern.fingerprint,
        source_dataset=_source_dataset(pattern),
        license=pattern.license,
        commercial_training=str(sensitivity.get("commercial_training", "review_required")),
        local_learning_only=bool(sensitivity.get("local_learning_only", False)),
        quality=pattern.quality,
        tags=sorted(pattern.tags),
        chord_context=_chord_context(pattern),
        section_context=_section_context(pattern),
        tokens=tokens,
        token_count=len(tokens),
        fingerprint=pattern.fingerprint,
        no_memorization_fingerprint=str(
            pattern.context.get("no_memorization_fingerprint") or pattern.fingerprint
        ),
        metadata={
            "category": pattern.category,
            "original_role": pattern.role,
            "usable_for_pattern_extraction": pattern.usable_for_pattern_extraction,
            "weight": pattern.weight,
            "tokenizer": MidiTokBridgeTokenizer.name,
            "tokenizer_version": MidiTokBridgeTokenizer.version,
            "pattern_sensitivity": sensitivity,
        },
    )


def _metadata_for_segment(segment: TokenizedSegment) -> TokenizedSegmentMetadata:
    return TokenizedSegmentMetadata(
        id=segment.id,
        split=segment.split,
        role=segment.role,
        style=segment.style,
        source_pattern_id=segment.source_pattern_id,
        source_file_id=segment.source_file_id,
        source_hash=segment.source_hash,
        source_dataset=segment.source_dataset,
        license=segment.license,
        commercial_training=segment.commercial_training,
        local_learning_only=segment.local_learning_only,
        quality=segment.quality,
        tags=segment.tags,
        chord_context=segment.chord_context,
        section_context=segment.section_context,
        token_count=segment.token_count,
        fingerprint=segment.fingerprint,
        no_memorization_fingerprint=segment.no_memorization_fingerprint,
    )


def _assign_splits(segments: list[TokenizedSegment], *, seed: int) -> list[TokenizedSegment]:
    ordered = sorted(segments, key=lambda item: _stable_hash(f"{seed}:{item.id}"))
    count = len(ordered)
    if count == 0:
        return []
    if count < 3:
        return [segment.model_copy(update={"split": "train"}) for segment in ordered]

    train_count = max(1, round(count * 0.8))
    val_count = max(1, round(count * 0.1))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1

    output: list[TokenizedSegment] = []
    for index, segment in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        output.append(segment.model_copy(update={"split": split}))
    return output


def _split_manifest(segments: list[TokenizedSegment], *, seed: int) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "seed": seed,
        "split_strategy": "stable_hash_80_10_10",
        "splits": {
            split: [segment.id for segment in segments if segment.split == split]
            for split in TRAINING_SPLITS
        },
        "split_counts": {
            split: sum(1 for segment in segments if segment.split == split)
            for split in TRAINING_SPLITS
        },
    }


def _role_manifest(segments: list[TokenizedSegment], roles: tuple[str, ...]) -> dict[str, Any]:
    by_role: dict[str, list[TokenizedSegment]] = {
        role: [segment for segment in segments if segment.role == role]
        for role in roles
    }
    return {
        "schema_version": "0.1.0",
        "roles": {
            role: {
                "segment_count": len(items),
                "styles": sorted({item.style for item in items}),
                "licenses": sorted({item.license for item in items}),
                "source_datasets": sorted({item.source_dataset for item in items}),
            }
            for role, items in by_role.items()
        },
    }


def _canonical_role(pattern: ExtractedPattern) -> str:
    category_role = _CATEGORY_ROLE_MAP.get(pattern.category)
    if category_role:
        return category_role
    return _ROLE_ALIASES.get(pattern.role.strip().lower(), "")


def _blocked_license(license_name: str) -> bool:
    return license_name.strip().lower() in BLOCKED_TRAINING_LICENSES


def _chord_context(pattern: ExtractedPattern) -> list[str]:
    chords = pattern.payload.get("chords")
    if isinstance(chords, list):
        return [str(chord) for chord in chords]
    context_chords = pattern.context.get("chord_context")
    if isinstance(context_chords, list):
        return [str(chord) for chord in context_chords]
    return []


def _section_context(pattern: ExtractedPattern) -> dict[str, Any]:
    raw_context = pattern.context.get("section_context")
    if isinstance(raw_context, dict):
        return raw_context
    section_context: dict[str, Any] = {}
    for key in ("section", "form", "bar_range", "phrase", "meter", "tempo"):
        if key in pattern.context:
            section_context[key] = pattern.context[key]
        elif key in pattern.payload:
            section_context[key] = pattern.payload[key]
    if "meter" not in section_context:
        section_context["meter"] = "4/4"
    return section_context


def _pattern_sensitivity(pattern: ExtractedPattern) -> dict[str, Any]:
    sensitivity = pattern.context.get("pattern_sensitivity")
    if isinstance(sensitivity, dict):
        return dict(sensitivity)
    return {
        "commercial_training": "review_required",
        "local_learning_only": False,
    }


def _source_dataset(pattern: ExtractedPattern) -> str:
    for key in ("source_dataset", "dataset_id", "dataset", "source"):
        value = pattern.context.get(key)
        if isinstance(value, str) and value.strip():
            return value
    source_path = Path(pattern.source_path)
    if len(source_path.parts) > 1:
        return source_path.parts[0]
    return "unknown"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
