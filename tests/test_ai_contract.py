from __future__ import annotations

import json
from pathlib import Path

import pytest
from arranger_core import (
    AIWalkingBassGenerator,
    DeterministicWalkingBassBackend,
    GenerationSpec,
    ModelRequest,
    RuleBasedArranger,
    export_project,
)
from dataset_tools import (
    ExtractedPattern,
    FeatureStore,
    PatternIndex,
    PatternTokenizer,
    build_training_examples,
    evaluate_memorization,
    load_training_examples,
)
from midi_models import (
    ExternalModelBackendAdapter,
    MidiTokBackendAdapter,
    SymbolicPatternModelBackend,
    train_symbolic_pattern_model,
)


def test_ai_walking_bass_generator_can_replace_rule_based_bass_without_api_changes(tmp_path):
    arranger = RuleBasedArranger(
        bass_generator=AIWalkingBassGenerator(DeterministicWalkingBassBackend())
    )

    project = arranger.generate(
        GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=141),
        project_id="ai-bass-replacement",
    )

    assert project.metadata["arranger"] == "hybrid_rule_model_v0"
    assert "AIWalkingBassGenerator" in project.metadata["role_generators"]
    bass = next(track for track in project.tracks if track.id == "double_bass")
    assert bass.metadata["generator"] == "AIWalkingBassGenerator"
    assert bass.metadata["model_backend"] == "deterministic-walking-bass-placeholder"
    assert project.validate_bar_durations() == []
    assert any(
        event.annotations.get("model_backend") == bass.metadata["model_backend"]
        for bar in bass.bars
        for event in bar.events
        if getattr(event, "type", None) == "note"
    )

    manifest = export_project(project, tmp_path / "export", include_pdf=False)
    assert manifest["status"] == "exported"
    assert (tmp_path / "export/full_arrangement.mid").exists()


def test_training_examples_use_only_allowed_licensed_patterns_and_create_splits(tmp_path):
    index = _training_pattern_index()

    summary = build_training_examples(index, tmp_path / "training", seed=142)
    examples = load_training_examples(summary.training_examples_path)
    feature_store = FeatureStore.load_json(summary.feature_store_path)
    split_manifest = json.loads(Path(summary.split_manifest_path).read_text(encoding="utf-8"))

    assert summary.total_examples == 6
    assert summary.split_counts["train"] > 0
    assert summary.split_counts["val"] > 0
    assert summary.split_counts["test"] > 0
    assert summary.skipped_not_training_allowed == 1
    assert summary.skipped_blocked_license == 1
    assert all(example.usable_for_training for example in examples)
    assert {example.license for example in examples} == {"CC0-1.0"}
    blocked_sources = {"blocked_usage", "blocked_license"}
    assert not any(example.source_file_id in blocked_sources for example in examples)
    assert set(split_manifest["splits"]) == {"train", "val", "test"}
    assert len(feature_store.search(role="walking_bass", usable_for_training=True)) == 6


def test_tokenization_placeholder_and_memorization_report(tmp_path):
    pattern = _pattern(0)
    tokenizer = PatternTokenizer()
    tokens = tokenizer.encode_pattern(pattern)

    assert "CATEGORY=walking_bass_cells" in tokens
    assert any(token.startswith("PAYLOAD|pitch_intervals") for token in tokens)

    summary = build_training_examples(PatternIndex(patterns=[pattern]), tmp_path / "training")
    reference = load_training_examples(summary.training_examples_path)[0]
    flagged = evaluate_memorization([reference.target_tokens], [reference], threshold=0.95)
    distinct = evaluate_memorization(
        [["NOTE|pitch=C2", "NOTE|pitch=G2"]],
        [reference],
        threshold=0.95,
    )

    assert flagged.status == "fail"
    assert flagged.flagged_count == 1
    assert distinct.status == "pass"


def test_miditok_and_external_model_adapters_are_explicit_placeholders():
    request = ModelRequest(role="walking_bass")

    with pytest.raises(NotImplementedError):
        MidiTokBackendAdapter().generate(request)
    with pytest.raises(NotImplementedError):
        ExternalModelBackendAdapter(endpoint="http://localhost:9999").generate(request)


def test_symbolic_pattern_model_backend_loads_trained_artifact(tmp_path):
    pattern_index = PatternIndex(patterns=[_pattern(1), _pattern(2)])
    summary = build_training_examples(pattern_index, tmp_path / "training", seed=144)
    model_path = tmp_path / "model.json"
    train_symbolic_pattern_model(
        pattern_index,
        model_path,
        source_roots=["midi_databases/JAZZVAR_DATASET"],
        training_summary=summary,
    )

    backend = SymbolicPatternModelBackend.load(model_path)
    response = backend.generate(
        ModelRequest(
            role="walking_bass",
            style="hard_bop",
            seed=144,
            chord_context=["Cm7", "F7"],
            controls={"bar_count": 2, "beats_per_bar": [4, 4]},
        )
    )

    assert response.backend_name == "symbolic-pattern-model"
    assert response.metadata["source_roots"] == ["midi_databases/JAZZVAR_DATASET"]
    assert len(response.target_tokens) == 8
    assert all(token.startswith("NOTE|") for token in response.target_tokens)


def _training_pattern_index() -> PatternIndex:
    index = PatternIndex()
    for number in range(6):
        index.add(_pattern(number))
    index.add(
        _pattern(
            98,
            source_file_id="blocked_usage",
            usable_for_training=False,
        )
    )
    index.add(
        _pattern(
            99,
            source_file_id="blocked_license",
            license="unknown",
        )
    )
    return index


def _pattern(
    number: int,
    *,
    source_file_id: str | None = None,
    license: str = "CC0-1.0",
    usable_for_training: bool = True,
) -> ExtractedPattern:
    return ExtractedPattern(
        id=f"bass_cell_{number}",
        category="walking_bass_cells",
        role="walking_bass",
        style="hard_bop",
        quality=4,
        source_file_id=source_file_id or f"source_{number}",
        source_path=f"synthetic/source_{number}.mid",
        source_hash=f"hash-{number}",
        license=license,
        usable_for_training=usable_for_training,
        usable_for_pattern_extraction=True,
        tags=["walking_bass"],
        payload={
            "pitch_intervals": [0, 3 + number % 2, 7, 10],
            "rhythm": [0.0, 1.0, 2.0, 3.0],
            "contour": [1, 1, 1],
        },
        context={"chord_context": ["Cm7", "F7"]},
        fingerprint=f"fingerprint-{number}",
    )
