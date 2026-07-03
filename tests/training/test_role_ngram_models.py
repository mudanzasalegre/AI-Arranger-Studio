from __future__ import annotations

import json

from training import (
    CUSTOM_ROLE_TRAINING_ROLES,
    RoleNgramModel,
    RoleTrainingSegment,
    checkpoint_dir_for_role,
    train_custom_role_ngram_checkpoints,
)

REQUIRED_CHECKPOINT_FILES = {
    "model.json",
    "tokenizer.json",
    "config.yaml",
    "training_manifest.yaml",
    "license_report.json",
    "metrics.json",
}


def test_train_custom_role_ngram_checkpoints_writes_required_files(tmp_path):
    summary = train_custom_role_ngram_checkpoints(
        _segments(),
        tmp_path / "custom",
        seed=3400,
        ngram_order=3,
        summary_path=tmp_path / "summary.json",
    )

    assert summary.roles == list(CUSTOM_ROLE_TRAINING_ROLES)
    assert summary.rejected_segments == 0
    for role in CUSTOM_ROLE_TRAINING_ROLES:
        checkpoint_dir = checkpoint_dir_for_role(tmp_path / "custom", role)
        assert {path.name for path in checkpoint_dir.iterdir()} >= REQUIRED_CHECKPOINT_FILES
        model = RoleNgramModel.load(checkpoint_dir / "model.json")
        generated = model.generate(seed=3400, max_tokens=24, prefix=["BOS", f"ROLE={role}"])
        assert generated[0] == "BOS"
        assert "EOS" in generated
        metrics = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["role"] == role
        assert metrics["model_type"] == "custom_role_ngram"


def _segments() -> list[RoleTrainingSegment]:
    segments: list[RoleTrainingSegment] = []
    for role in CUSTOM_ROLE_TRAINING_ROLES:
        for index, split in enumerate(("train", "val", "test")):
            segments.append(
                RoleTrainingSegment(
                    id=f"{role}_{index}",
                    role=role,
                    split=split,
                    tokens=[
                        "BOS",
                        f"ROLE={role}",
                        "STYLE=hard_bop",
                        f"STEP={index}",
                        "DUR=1.0",
                        "EOS",
                    ],
                    style="hard_bop",
                    source_file_id=f"source_{role}_{index}",
                    source_path=f"synthetic/{role}_{index}.mid",
                    source_hash=f"hash-{role}-{index}",
                    source_dataset="synthetic_unit",
                    license="CC0-1.0",
                    commercial_training="allowed",
                    quality=4,
                )
            )
    return segments
