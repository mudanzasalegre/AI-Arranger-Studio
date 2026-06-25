from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from arranger_core import ArrangementProject, NoteEvent

from dataset_tools.models import (
    DatasetSplitSummary,
    ExtractedPattern,
    FeatureRecord,
    FeatureStore,
    MemorizationReport,
    PatternIndex,
    TrainingExample,
)

TRAINING_SPLITS = ("train", "val", "test")
BLOCKED_TRAINING_LICENSES = {"", "unknown", "proprietary", "all rights reserved"}
ROLE_INSTRUMENTS = {
    "drums": "drum_kit",
    "walking_bass": "double_bass",
    "comping": "piano",
    "piano": "piano",
    "melody": "lead_instrument",
    "horn_response": "horn_section",
    "harmony": "harmony",
}


class PatternTokenizer:
    """Placeholder tokenizer for future symbolic model training."""

    def encode_pattern(self, pattern: ExtractedPattern) -> list[str]:
        tokens = [
            f"CATEGORY={pattern.category}",
            f"ROLE={pattern.role}",
            f"STYLE={pattern.style}",
            f"QUALITY={pattern.quality}",
        ]
        tokens.extend(_payload_tokens("CTX", pattern.context))
        tokens.extend(_payload_tokens("PAYLOAD", pattern.payload))
        return tokens

    def encode_project(
        self,
        project: ArrangementProject,
        *,
        role: str | None = None,
    ) -> list[str]:
        tokens = [
            f"PROJECT={project.project_id}",
            f"STYLE={project.generation_spec.style if project.generation_spec else 'unknown'}",
        ]
        for track in project.tracks:
            if role and track.role != role:
                continue
            tokens.append(f"TRACK|id={track.id}|role={track.role}|instrument={track.instrument}")
            for bar in track.bars:
                for event in bar.events:
                    if not isinstance(event, NoteEvent):
                        continue
                    tokens.append(
                        "NOTE"
                        f"|track={track.id}"
                        f"|bar={bar.number}"
                        f"|start={event.start:g}"
                        f"|duration={event.duration:g}"
                        f"|pitch={event.pitch}"
                    )
        return tokens


def build_training_examples(
    pattern_index: PatternIndex | str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    min_quality: int = 3,
    tokenizer: PatternTokenizer | None = None,
) -> DatasetSplitSummary:
    index = (
        PatternIndex.load_json(pattern_index)
        if isinstance(pattern_index, str | Path)
        else pattern_index
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer or PatternTokenizer()

    raw_examples: list[TrainingExample] = []
    skipped_not_allowed = 0
    skipped_blocked_license = 0
    for pattern in index.patterns:
        if pattern.quality < min_quality or not pattern.usable_for_training:
            skipped_not_allowed += 1
            continue
        if _blocked_license(pattern.license):
            skipped_blocked_license += 1
            continue
        raw_examples.append(_training_example(pattern, tokenizer))

    examples = _assign_splits(raw_examples, seed=seed)
    feature_store = _build_feature_store(examples)

    examples_path = output_path / "training_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(example.model_dump_json() + "\n")

    feature_store_path = output_path / "feature_store.json"
    feature_store.save_json(feature_store_path)

    split_counts = Counter(example.split for example in examples)
    split_manifest_path = output_path / "dataset_splits.json"
    split_manifest = {
        "schema_version": "0.1.0",
        "splits": {
            split: [example.id for example in examples if example.split == split]
            for split in TRAINING_SPLITS
        },
        "split_counts": dict(sorted(split_counts.items())),
    }
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = DatasetSplitSummary(
        total_examples=len(examples),
        split_counts={split: split_counts.get(split, 0) for split in TRAINING_SPLITS},
        skipped_not_training_allowed=skipped_not_allowed,
        skipped_blocked_license=skipped_blocked_license,
        training_examples_path=str(examples_path),
        feature_store_path=str(feature_store_path),
        split_manifest_path=str(split_manifest_path),
    )
    (output_path / "training_summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def token_jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def evaluate_memorization(
    candidates: Iterable[TrainingExample | Sequence[str]],
    references: Iterable[TrainingExample | Sequence[str]],
    *,
    threshold: float = 0.8,
) -> MemorizationReport:
    candidate_items = list(candidates)
    reference_items = list(references)
    matches: list[dict[str, Any]] = []
    max_similarity = 0.0
    for candidate_index, candidate in enumerate(candidate_items):
        candidate_tokens = _tokens(candidate)
        candidate_id = _item_id(candidate, f"candidate_{candidate_index}")
        best: tuple[str, float] | None = None
        for reference_index, reference in enumerate(reference_items):
            reference_id = _item_id(reference, f"reference_{reference_index}")
            similarity = token_jaccard_similarity(candidate_tokens, _tokens(reference))
            if best is None or similarity > best[1]:
                best = (reference_id, similarity)
        if best is None:
            continue
        max_similarity = max(max_similarity, best[1])
        if best[1] >= threshold:
            matches.append(
                {
                    "candidate_id": candidate_id,
                    "reference_id": best[0],
                    "similarity": round(best[1], 6),
                }
            )

    return MemorizationReport(
        status="fail" if matches else "pass",
        threshold=threshold,
        candidate_count=len(candidate_items),
        reference_count=len(reference_items),
        flagged_count=len(matches),
        max_similarity=round(max_similarity, 6),
        matches=matches,
    )


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(TrainingExample.model_validate_json(line))
    return examples


def _training_example(
    pattern: ExtractedPattern,
    tokenizer: PatternTokenizer,
) -> TrainingExample:
    source_hash = pattern.source_hash or pattern.fingerprint
    return TrainingExample(
        id=f"example_{pattern.fingerprint[:16]}",
        style=pattern.style,
        role=pattern.role,
        instrument=ROLE_INSTRUMENTS.get(pattern.role, "unknown"),
        chord_context=_chord_context(pattern),
        target_tokens=tokenizer.encode_pattern(pattern),
        source_file_id=pattern.source_file_id,
        source_path=pattern.source_path,
        source_hash=source_hash,
        license=pattern.license,
        usable_for_training=pattern.usable_for_training,
        pattern_id=pattern.id,
        pattern_fingerprint=pattern.fingerprint,
        metadata={
            "category": pattern.category,
            "quality": pattern.quality,
            "tags": pattern.tags,
        },
    )


def _assign_splits(
    examples: list[TrainingExample],
    *,
    seed: int,
) -> list[TrainingExample]:
    ordered = sorted(
        examples,
        key=lambda example: hashlib.sha256(f"{seed}:{example.id}".encode()).hexdigest(),
    )
    count = len(ordered)
    if count == 0:
        return []
    if count < 3:
        return [
            example.model_copy(update={"split": "train"})
            for example in ordered
        ]

    train_count = max(1, round(count * 0.8))
    val_count = max(1, round(count * 0.1))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1

    output: list[TrainingExample] = []
    for index, example in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        output.append(example.model_copy(update={"split": split}))
    return output


def _build_feature_store(examples: list[TrainingExample]) -> FeatureStore:
    store = FeatureStore()
    for example in examples:
        store.add(
            FeatureRecord(
                id=f"feature_{example.id}",
                example_id=example.id,
                role=example.role,
                style=example.style,
                source_file_id=example.source_file_id,
                license=example.license,
                usable_for_training=example.usable_for_training,
                values={
                    "target_token_count": len(example.target_tokens),
                    "previous_token_count": len(example.previous_tokens),
                    "chord_context_count": len(example.chord_context),
                    "quality": example.metadata.get("quality", 0),
                    "category": example.metadata.get("category", ""),
                },
            )
        )
    return store


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


def _payload_tokens(prefix: str, payload: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in sorted(payload.items()):
        if isinstance(value, list):
            for index, item in enumerate(value):
                tokens.append(f"{prefix}|{key}[{index}]={item}")
        elif isinstance(value, dict):
            tokens.extend(_payload_tokens(f"{prefix}|{key}", value))
        else:
            tokens.append(f"{prefix}|{key}={value}")
    return tokens


def _tokens(item: TrainingExample | Sequence[str]) -> Sequence[str]:
    if isinstance(item, TrainingExample):
        return item.target_tokens
    return item


def _item_id(item: TrainingExample | Sequence[str], fallback: str) -> str:
    if isinstance(item, TrainingExample):
        return item.id
    return fallback
