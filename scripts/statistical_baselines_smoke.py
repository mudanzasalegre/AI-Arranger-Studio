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
    BASELINE_ROLE_MODEL_TYPES,
    StatisticalRoleModel,
    export_tokenized_dataset,
    train_baseline_statistical_models,
)

OUTPUT_ROOT = ROOT / "outputs" / "pr18_statistical_smoke"


def main() -> None:
    outputs_root = (ROOT / "outputs").resolve()
    smoke_root = OUTPUT_ROOT.resolve()
    if outputs_root not in smoke_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {smoke_root}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    pattern_index = _pattern_index()
    tokenized = export_tokenized_dataset(
        pattern_index,
        smoke_root / "tokenized",
        seed=1800,
        min_quality=3,
    )
    summary = train_baseline_statistical_models(
        tokenized.tokenized_segments_path,
        smoke_root / "statistical",
        seed=1800,
        ngram_order=3,
    )
    comparison = json.loads(Path(summary.comparison_report_path).read_text(encoding="utf-8"))
    if set(summary.model_paths) != set(BASELINE_ROLE_MODEL_TYPES):
        raise RuntimeError("Statistical smoke did not write every role model")
    if set(comparison["roles"]) != set(BASELINE_ROLE_MODEL_TYPES):
        raise RuntimeError("Comparison report does not cover every role")

    melody = StatisticalRoleModel.load(summary.model_paths["melody"])
    generated = melody.generate(seed=1800, max_tokens=24)
    score = melody.score(generated)
    if not generated or score["perplexity"] <= 0:
        raise RuntimeError("Loaded statistical melody model cannot generate/score")

    smoke_summary = {
        "status": "pass",
        "tokenized_segments": tokenized.total_segments,
        "model_count": summary.model_count,
        "model_paths": summary.model_paths,
        "pattern_probability_model_path": summary.pattern_probability_model_path,
        "comparison_report_path": summary.comparison_report_path,
        "generated_preview": generated[:8],
        "generated_perplexity": score["perplexity"],
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
        id=f"pr18_{suffix}_{number}",
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
        fingerprint=f"pr18-fingerprint-{number:02d}",
    )


if __name__ == "__main__":
    main()
