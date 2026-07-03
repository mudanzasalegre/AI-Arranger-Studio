from __future__ import annotations

import json
from pathlib import Path

import yaml
from arranger_core import ProfessionalGenerationOptions, ProfessionalGenerationOrchestrator
from training import (
    CUSTOM_ROLE_TRAINING_ROLES,
    RoleTrainingSegment,
    checkpoint_dir_for_role,
    train_custom_role_ngram_checkpoints,
)


def test_professional_orchestrator_exports_custom_role_run(tmp_path):
    checkpoint_root = tmp_path / "checkpoints" / "custom"
    train_custom_role_ngram_checkpoints(_segments(), checkpoint_root, seed=3500)
    ai_config = _ai_config(tmp_path, checkpoint_root)
    thresholds = tmp_path / "quality_thresholds.pro.yaml"
    thresholds.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "max_blocking_errors": 0,
                    "min_tracks": 3,
                    "min_note_events": 40,
                },
                "ratings": {
                    "A": {"min_score": 0.88},
                    "B": {"min_score": 0.72},
                    "C": {"min_score": 0.55},
                    "D": {"min_score": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )

    result = ProfessionalGenerationOrchestrator(repo_root=Path.cwd()).generate(
        ProfessionalGenerationOptions(
            prompt="hard bop minor blues sextet, 132 bpm, alto sax and horns",
            profile="unit_pro",
            seed=3500,
            run_id="unit_professional",
            output_root=str(tmp_path / "outputs"),
            ai_config_path=str(ai_config),
            quality_thresholds_path=str(thresholds),
            use_llm_planner=False,
            use_custom_role_models=True,
            use_midigpt_infill=False,
            use_text2midi_sketch=False,
            min_rating="B",
        )
    )

    output_dir = Path(result.output_dir)
    assert result.status == "ok"
    assert result.quality["rating"] in {"A", "B"}
    assert (output_dir / "full_arrangement.mid").exists()
    assert (output_dir / "full_score.musicxml").exists()
    assert (output_dir / "quality_report.json").exists()
    assert result.model_trace["model_artifacts"]
    assert {take["status"] for take in result.takes_manifest["takes"]} == {"accepted"}
    exported_takes = json.loads((output_dir / "takes_manifest.json").read_text(encoding="utf-8"))
    assert {take["status"] for take in exported_takes["takes"]} == {"accepted"}


def _ai_config(tmp_path: Path, checkpoint_root: Path) -> Path:
    backends = {}
    role_backend_ids = {
        "melody": "custom_jazz_melody_v001",
        "walking_bass": "custom_jazz_walking_bass_v001",
        "piano_comping": "custom_jazz_piano_comping_v001",
        "horn_responses": "custom_jazz_horn_responses_v001",
        "drums": "custom_jazz_drums_v001",
    }
    for role, backend_id in role_backend_ids.items():
        backends[backend_id] = {
            "enabled": True,
            "type": "custom_role",
            "adapter": (
                "model_backends.custom_role.statistical_backend."
                "StatisticalCustomRoleBackend"
            ),
            "role": role,
            "checkpoint_dir": str(checkpoint_dir_for_role(checkpoint_root, role)),
            "commercial_use": "unknown",
            "tasks": ["generate_track", "infill_bars", "generate_variation"],
        }
    path = tmp_path / "ai_models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "backends": backends,
                "settings": {"artifact_raw_dir": str(tmp_path / "raw")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


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
                        f"CELL={index}",
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
