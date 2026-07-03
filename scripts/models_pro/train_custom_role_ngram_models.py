from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for package in ("dataset_tools", "training"):
    path = str(ROOT / "packages" / package)
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_tools import ExtractedPattern, PatternIndex  # noqa: E402
from training import export_tokenized_dataset, train_custom_role_ngram_checkpoints  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pro_benchmarks" / "custom_role_ngram_training"
DEFAULT_CHECKPOINT_ROOT = ROOT / "models" / "checkpoints" / "custom"
DEFAULT_MANIFEST_PATH = ROOT / "models" / "manifests" / "custom_role_ngram_training_report.json"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = _repo_path(args.output_dir)
    checkpoint_root = _repo_path(args.checkpoint_root)
    segments_path = _repo_path(args.segments) if args.segments else None

    if segments_path is None:
        _clean_output_dir(output_dir)
        pattern_index_path = output_dir / "synthetic_pattern_index.json"
        pattern_index = _pattern_index(repeats=args.synthetic_repeats)
        pattern_index.save_json(pattern_index_path)
        tokenized = export_tokenized_dataset(
            pattern_index,
            output_dir / "tokenized",
            seed=args.seed,
            min_quality=3,
        )
        segments_path = Path(tokenized.tokenized_segments_path)
        tokenized_summary = tokenized.model_dump(mode="json")
    else:
        if not segments_path.exists():
            raise SystemExit(f"Tokenized role segment file not found: {segments_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern_index_path = None
        tokenized_summary = {"tokenized_segments_path": str(segments_path)}

    summary = train_custom_role_ngram_checkpoints(
        segments_path,
        checkpoint_root,
        seed=args.seed,
        ngram_order=args.ngram_order,
        summary_path=output_dir / "custom_role_ngram_summary.json",
        clean=not args.no_clean,
    )
    report = {
        "status": "ok",
        "schema_version": summary.schema_version,
        "script": "train_custom_role_ngram_models",
        "seed": args.seed,
        "ngram_order": args.ngram_order,
        "checkpoint_root": str(checkpoint_root),
        "source_segments_path": str(segments_path),
        "synthetic_pattern_index_path": str(pattern_index_path) if pattern_index_path else None,
        "tokenized_summary": tokenized_summary,
        "roles": summary.roles,
        "total_segments": summary.total_segments,
        "rejected_segments": summary.rejected_segments,
        "checkpoints": {
            role: record.model_dump(mode="json")
            for role, record in summary.checkpoints.items()
        },
        "summary_path": summary.summary_path,
    }
    report_path = output_dir / "train_report.json"
    _write_json(report_path, report)
    _write_json(DEFAULT_MANIFEST_PATH, report)
    print(json.dumps(report, indent=2))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train local statistical n-gram checkpoints for all custom role models."
    )
    parser.add_argument("--segments", default=None, help="Optional tokenized_segments.jsonl path.")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=3400)
    parser.add_argument("--ngram-order", type=int, default=3)
    parser.add_argument("--synthetic-repeats", type=int, default=8)
    parser.add_argument("--no-clean", action="store_true", help="Do not replace role checkpoints.")
    return parser.parse_args(argv)


def _pattern_index(*, repeats: int) -> PatternIndex:
    index = PatternIndex()
    specs = [
        ("melody", "melodic_motifs", "melody"),
        ("bass", "walking_bass_cells", "walking_bass"),
        ("piano", "piano_voicings", "comping"),
        ("horns", "horn_responses", "horn_response"),
        ("drums", "drum_grooves", "drums"),
    ]
    for repeat in range(max(3, repeats)):
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
    chord_pairs = [
        ["Cm7", "F7"],
        ["Fm7", "Bb7"],
        ["Ebmaj7", "Ab7"],
        ["Dm7b5", "G7alt"],
    ]
    chords = chord_pairs[number % len(chord_pairs)]
    return ExtractedPattern(
        id=f"pr34_{suffix}_{number}",
        category=category,
        role=role,
        style="hard_bop",
        quality=4,
        source_file_id=f"synthetic_pr34_source_{number:03d}",
        source_path=f"synthetic/pr34/source_{number:03d}.mid",
        source_hash=f"synthetic-pr34-source-hash-{number:03d}",
        license="CC0-1.0",
        usable_for_training=True,
        usable_for_pattern_extraction=True,
        tags=["pr34", role, "hard_bop"],
        context={
            "source_dataset": "synthetic_pr34_custom_role",
            "chord_context": chords,
            "section_context": {"section": "A", "bar_range": [1, 4], "meter": "4/4"},
            "pattern_sensitivity": {
                "commercial_training": "allowed",
                "local_learning_only": False,
            },
            "no_memorization_fingerprint": f"synthetic-pr34-no-memo-{number:03d}",
        },
        payload={
            "chords": chords,
            "rhythm": _rhythm_for_role(role, number),
            "pitch_intervals": _pitch_intervals_for_role(role, number),
            "bar_range": [1, 4],
            "velocity_shape": [72 + number % 8, 68, 74, 70],
        },
        fingerprint=f"synthetic-pr34-fingerprint-{suffix}-{number:03d}",
    )


def _rhythm_for_role(role: str, number: int) -> list[float]:
    if role == "drums":
        return [0.0, 1.0, 2.0, 3.0, 3.5 if number % 2 else 2.5]
    if role == "horn_response":
        return [2.0, 2.5, 3.0]
    if role == "comping":
        return [0.5, 2.5]
    return [0.0, 1.0, 2.0, 3.0]


def _pitch_intervals_for_role(role: str, number: int) -> list[int]:
    offset = number % 4
    if role == "walking_bass":
        return [0, 2 + offset, 5, 7 + offset]
    if role == "comping":
        return [0, 3, 7, 10 + offset]
    if role == "horn_response":
        return [7, 10 + offset, 12]
    if role == "drums":
        return [36, 42, 38, 46]
    return [0, 3 + offset, 7, 10]


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _clean_output_dir(path: Path) -> None:
    outputs_root = (ROOT / "outputs").resolve()
    resolved = path.resolve()
    if path.exists():
        if resolved == outputs_root or outputs_root not in resolved.parents:
            raise RuntimeError(f"Refusing to clean output path outside outputs/: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
