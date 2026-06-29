from __future__ import annotations

import json
from pathlib import Path

from dataset_tools import ExtractedPattern, PatternIndex
from training import (
    BASELINE_ROLE_MODEL_TYPES,
    StatisticalRoleModel,
    export_tokenized_dataset,
    train_baseline_statistical_models,
)


def test_statistical_baseline_training_writes_role_models_and_comparison(tmp_path):
    tokenized = export_tokenized_dataset(
        _pattern_index(),
        tmp_path / "tokenized",
        seed=1801,
    )

    summary = train_baseline_statistical_models(
        tokenized.tokenized_segments_path,
        tmp_path / "statistical",
        seed=1801,
        ngram_order=3,
    )
    comparison = json.loads(Path(summary.comparison_report_path).read_text(encoding="utf-8"))

    assert summary.model_count == 6
    assert set(summary.model_paths) == set(BASELINE_ROLE_MODEL_TYPES)
    assert Path(summary.pattern_probability_model_path).exists()
    assert set(comparison["roles"]) == set(BASELINE_ROLE_MODEL_TYPES)
    assert comparison["roles"]["melody"]["model_type"] == "ngram_melody"
    assert comparison["roles"]["bass"]["model_type"] == "ngram_bass"
    assert comparison["roles"]["piano_comping"]["model_type"] == "markov_voicings"
    assert comparison["roles"]["drums"]["model_type"] == "drum_fill_retrieval"
    assert "rule_based_proxy" in comparison["roles"]["melody"]
    assert "retrieval_proxy" in comparison["roles"]["melody"]


def test_statistical_role_model_loads_generates_and_scores_deterministically(tmp_path):
    tokenized = export_tokenized_dataset(
        _pattern_index(),
        tmp_path / "tokenized",
        seed=1802,
    )
    summary = train_baseline_statistical_models(
        tokenized.tokenized_segments_path,
        tmp_path / "statistical",
        seed=1802,
    )

    model = StatisticalRoleModel.load(summary.model_paths["bass"])
    first = model.generate(seed=1802, max_tokens=96)
    second = model.generate(seed=1802, max_tokens=96)
    score = model.score(first)

    assert first == second
    assert first[0] == "BOS"
    assert "EOS" in first
    assert score["token_count"] > 0
    assert score["perplexity"] > 0


def _pattern_index() -> PatternIndex:
    index = PatternIndex()
    specs = [
        ("melody", "melodic_motifs", "melody"),
        ("bass", "walking_bass_cells", "walking_bass"),
        ("piano", "piano_voicings", "comping"),
        ("horns", "horn_responses", "horn_response"),
        ("drums", "drum_grooves", "drums"),
    ]
    for repeat in range(3):
        for number, (suffix, category, role) in enumerate(specs):
            index.add(
                _pattern(
                    repeat * 10 + number,
                    suffix=suffix,
                    category=category,
                    role=role,
                )
            )
    return index


def _pattern(
    number: int,
    *,
    suffix: str,
    category: str,
    role: str,
) -> ExtractedPattern:
    return ExtractedPattern(
        id=f"test_pr18_{suffix}_{number}",
        category=category,
        role=role,
        style="hard_bop",
        quality=4,
        source_file_id=f"source_{number:02d}",
        source_path=f"synthetic/pr18/source_{number:02d}.mid",
        source_hash=f"source-hash-{number:02d}",
        license="CC0-1.0",
        usable_for_training=True,
        usable_for_pattern_extraction=True,
        tags=["pr18", role],
        context={
            "source_dataset": "synthetic_pr18",
            "chord_context": ["Cm7", "F7"],
            "section_context": {"section": "A", "bar_range": [1, 4], "meter": "4/4"},
            "pattern_sensitivity": {
                "commercial_training": "allowed",
                "local_learning_only": False,
            },
            "no_memorization_fingerprint": f"no-memo-{number:02d}",
        },
        payload={
            "chords": ["Cm7", "F7"],
            "rhythm": [0.0, 1.0, 2.0, 3.0],
            "pitch_intervals": [0, 3 + number % 3, 7, 10],
            "bar_range": [1, 4],
            "velocity_shape": [72, 66 + number % 4, 68, 70],
        },
        fingerprint=f"test-pr18-fingerprint-{number:02d}",
    )
