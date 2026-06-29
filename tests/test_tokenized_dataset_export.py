from __future__ import annotations

import json
from pathlib import Path

from dataset_tools import ExtractedPattern, PatternIndex
from training import TOKENIZATION_ROLES, export_tokenized_dataset, load_tokenized_segments


def test_tokenized_dataset_export_writes_role_datasets_and_metadata(tmp_path):
    index = _pattern_index()

    summary = export_tokenized_dataset(index, tmp_path / "tokenized", seed=1701)
    segments = load_tokenized_segments(summary.tokenized_segments_path)
    metadata_lines = [
        json.loads(line)
        for line in Path(summary.metadata_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_manifest = json.loads(Path(summary.split_manifest_path).read_text(encoding="utf-8"))
    miditok_config = json.loads(Path(summary.miditok_config_path).read_text(encoding="utf-8"))

    assert summary.total_segments == 5
    assert summary.role_counts == {role: 1 for role in TOKENIZATION_ROLES}
    assert summary.skipped_not_training_allowed == 1
    assert summary.skipped_blocked_license == 1
    assert {segment.role for segment in segments} == set(TOKENIZATION_ROLES)
    assert all(segment.license == "CC0-1.0" for segment in segments)
    assert all(segment.chord_context == ["Cm7", "F7"] for segment in segments)
    assert all(segment.section_context["section"] == "A" for segment in segments)
    assert all(segment.tokens[0] == "BOS" and segment.tokens[-1] == "EOS" for segment in segments)
    assert len(metadata_lines) == len(segments)
    assert set(split_manifest["splits"]) == {"train", "val", "test"}
    assert miditok_config["compatible_target"] == "MidiTok"
    assert miditok_config["roles"] == list(TOKENIZATION_ROLES)


def test_tokenized_dataset_export_is_reproducible(tmp_path):
    index = _pattern_index()

    first = export_tokenized_dataset(index, tmp_path / "first", seed=1702)
    second = export_tokenized_dataset(index, tmp_path / "second", seed=1702)

    assert Path(first.tokenized_segments_path).read_text(encoding="utf-8") == Path(
        second.tokenized_segments_path
    ).read_text(encoding="utf-8")
    assert json.loads(Path(first.split_manifest_path).read_text(encoding="utf-8")) == json.loads(
        Path(second.split_manifest_path).read_text(encoding="utf-8")
    )


def _pattern_index() -> PatternIndex:
    index = PatternIndex()
    specs = [
        ("melody", "melodic_motifs", "melody"),
        ("bass", "walking_bass_cells", "walking_bass"),
        ("piano", "piano_voicings", "comping"),
        ("horns", "horn_responses", "horn_response"),
        ("drums", "drum_grooves", "drums"),
    ]
    for number, (suffix, category, role) in enumerate(specs):
        index.add(_pattern(number, suffix=suffix, category=category, role=role))
    index.add(
        _pattern(
            98,
            suffix="blocked_training",
            category="walking_bass_cells",
            role="walking_bass",
            usable_for_training=False,
        )
    )
    index.add(
        _pattern(
            99,
            suffix="blocked_license",
            category="melodic_motifs",
            role="melody",
            license="unknown",
        )
    )
    return index


def _pattern(
    number: int,
    *,
    suffix: str,
    category: str,
    role: str,
    license: str = "CC0-1.0",
    usable_for_training: bool = True,
) -> ExtractedPattern:
    return ExtractedPattern(
        id=f"test_{suffix}_{number}",
        category=category,
        role=role,
        style="hard_bop",
        quality=4,
        source_file_id=f"source_{number:02d}",
        source_path=f"synthetic/pr17/source_{number:02d}.mid",
        source_hash=f"source-hash-{number:02d}",
        license=license,
        usable_for_training=usable_for_training,
        usable_for_pattern_extraction=True,
        tags=["pr17", role],
        context={
            "source_dataset": "synthetic_pr17",
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
            "pitch_intervals": [0, 3 + number % 2, 7, 10],
            "bar_range": [1, 4],
        },
        fingerprint=f"test-pr17-fingerprint-{number:02d}",
    )
