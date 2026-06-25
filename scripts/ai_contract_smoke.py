from __future__ import annotations

import json
import shutil
from pathlib import Path

from arranger_core import (
    AIWalkingBassGenerator,
    DeterministicWalkingBassBackend,
    GenerationSpec,
    RuleBasedArranger,
    export_project,
    validate_project,
)
from dataset_tools import (
    ExtractedPattern,
    PatternIndex,
    PatternTokenizer,
    build_training_examples,
    evaluate_memorization,
    load_training_examples,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/obj14_ai_contract"


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

    training_summary = build_training_examples(
        pattern_index,
        smoke_root / "training",
        seed=140,
    )
    training_examples = load_training_examples(training_summary.training_examples_path)

    arranger = RuleBasedArranger(
        bass_generator=AIWalkingBassGenerator(DeterministicWalkingBassBackend())
    )
    project = arranger.generate(
        GenerationSpec(
            prompt="OBJ14 AI contract smoke",
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=140,
        ),
        project_id="obj14-ai-contract-smoke",
    )
    report = validate_project(project)
    if report["errors"]:
        raise RuntimeError(f"AI contract smoke project failed validation: {report['errors']}")

    tokenizer = PatternTokenizer()
    generated_bass_tokens = tokenizer.encode_project(project, role="walking_bass")
    memorization = evaluate_memorization(
        [generated_bass_tokens],
        training_examples,
        threshold=0.95,
    )
    export_manifest = export_project(project, smoke_root / "generated", include_pdf=False)

    summary = {
        "status": "pass",
        "pattern_index_path": str(pattern_index_path),
        "training_examples": training_summary.total_examples,
        "split_counts": training_summary.split_counts,
        "feature_store_path": training_summary.feature_store_path,
        "arranger": project.metadata["arranger"],
        "role_generators": project.metadata["role_generators"],
        "bass_generator": project.tracks[1].metadata["generator"],
        "bass_backend": project.tracks[1].metadata["model_backend"],
        "validation_status": report["status"],
        "memorization_status": memorization.status,
        "memorization_max_similarity": memorization.max_similarity,
        "exported_files": len(export_manifest["files"]),
    }
    (smoke_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _pattern_index() -> PatternIndex:
    index = PatternIndex()
    for number in range(8):
        index.add(
            ExtractedPattern(
                id=f"obj14_bass_cell_{number}",
                category="walking_bass_cells",
                role="walking_bass",
                style="hard_bop",
                quality=4,
                source_file_id=f"source_{number:02d}",
                source_path=f"synthetic/source_{number:02d}.mid",
                source_hash=f"hash-{number:02d}",
                license="CC0-1.0",
                usable_for_training=True,
                usable_for_pattern_extraction=True,
                tags=["obj14", "walking_bass"],
                payload={
                    "pitch_intervals": [0, 3 + number % 2, 7, 10],
                    "rhythm": [0.0, 1.0, 2.0, 3.0],
                    "contour": [1, 1, 1],
                    "start_degree": 0,
                    "end_degree": 10,
                },
                context={"chord_context": ["Cm7", "F7"]},
                fingerprint=f"obj14-good-{number:02d}",
            )
        )
    index.add(
        ExtractedPattern(
            id="obj14_blocked_training",
            category="walking_bass_cells",
            role="walking_bass",
            style="hard_bop",
            quality=4,
            source_file_id="blocked",
            source_path="synthetic/blocked.mid",
            source_hash="blocked-hash",
            license="CC0-1.0",
            usable_for_training=False,
            usable_for_pattern_extraction=True,
            payload={"pitch_intervals": [0, 4, 7, 10]},
            fingerprint="obj14-blocked-training",
        )
    )
    return index


if __name__ == "__main__":
    main()
