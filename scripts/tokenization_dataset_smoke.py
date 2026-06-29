from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_path in (
    ROOT / "packages" / "arranger_core",
    ROOT / "packages" / "dataset_tools",
    ROOT / "packages" / "training",
):
    sys.path.insert(0, str(package_path))

from dataset_tools import ExtractedPattern, PatternIndex  # noqa: E402
from training import (  # noqa: E402
    TOKENIZATION_ROLES,
    export_tokenized_dataset,
    load_tokenized_segments,
)

OUTPUT_ROOT = ROOT / "outputs" / "pr17_tokenization_smoke"


def main() -> None:
    outputs_root = (ROOT / "outputs").resolve()
    smoke_root = OUTPUT_ROOT.resolve()
    if outputs_root not in smoke_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {smoke_root}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    pattern_index = _pattern_index()
    pattern_index_path = smoke_root / "pattern_index.json"
    pattern_index.save_json(pattern_index_path)

    summary = export_tokenized_dataset(
        pattern_index_path,
        smoke_root / "tokenized",
        seed=1700,
        min_quality=3,
    )
    segments = load_tokenized_segments(summary.tokenized_segments_path)
    if summary.total_segments != len(TOKENIZATION_ROLES):
        raise RuntimeError(f"Expected one exported segment per role, got {summary.total_segments}")
    if {segment.role for segment in segments} != set(TOKENIZATION_ROLES):
        raise RuntimeError("Tokenized smoke did not cover all training roles")
    if summary.skipped_not_training_allowed != 1 or summary.skipped_blocked_license != 1:
        raise RuntimeError("Tokenized smoke did not enforce training/license skips")

    smoke_summary = {
        "status": "pass",
        "pattern_index_path": str(pattern_index_path),
        "total_segments": summary.total_segments,
        "role_counts": summary.role_counts,
        "split_counts": summary.split_counts,
        "tokenized_segments_path": summary.tokenized_segments_path,
        "metadata_path": summary.metadata_path,
        "miditok_config_path": summary.miditok_config_path,
    }
    (smoke_root / "smoke_summary.json").write_text(
        json.dumps(smoke_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(smoke_summary, indent=2))


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
        id=f"pr17_{suffix}_{number}",
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
        fingerprint=f"pr17-fingerprint-{number:02d}",
    )


if __name__ == "__main__":
    main()
