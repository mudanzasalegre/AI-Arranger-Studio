from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest
import yaml
from model_backends import (
    DummyCustomRoleModelBackend,
    ModelBackendUnavailableError,
    ModelGenerationError,
    ModelGenerationRequest,
    StatisticalCustomRoleBackend,
    build_model_backend_registry,
    inspect_custom_role_model,
)
from model_backends.config import AIModelsConfig, BackendConfig
from training import RoleTrainingSegment, train_custom_role_ngram_checkpoints


def test_custom_role_loader_marks_missing_checkpoint_unavailable(tmp_path):
    inspection = inspect_custom_role_model(
        {
            "backend_id": "custom_jazz_melody_v001",
            "role": "melody",
            "checkpoint_dir": str(tmp_path / "missing"),
        }
    )

    assert inspection.available is False
    assert inspection.commercial_use == "unknown"
    assert len(inspection.missing_files) == 6
    assert "training_manifest.yaml" in inspection.unavailable_reason


def test_custom_role_dummy_backend_generates_token_artifact(tmp_path):
    checkpoint_dir = _checkpoint(
        tmp_path / "jazz_melody_v001",
        role="melody",
        license_name="CC0-1.0",
        commercial_training="allowed",
    )
    backend = DummyCustomRoleModelBackend(
        backend_id="custom_jazz_melody_v001",
        role="melody",
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "raw",
    )

    assert backend.is_available() is True
    assert backend.capabilities.commercial_use == "allowed"

    result = backend.generate(
        ModelGenerationRequest(
            request_id="unit",
            task="infill_bars",
            role_intent={"role": "melody", "density": "medium"},
            bars=[1, 2],
            seed=2501,
        )
    )

    assert result.backend_id == "custom_jazz_melody_v001"
    assert result.artifacts[0].artifact_type == "tokens"
    payload = json.loads(Path(result.artifacts[0].path).read_text(encoding="utf-8"))
    assert payload["role"] == "melody"
    assert payload["target_tokens"][0] == "BOS"
    assert payload["target_tokens"][-1] == "EOS"
    assert result.artifacts[0].metadata["commercial_allowed"] is True


def test_custom_role_dummy_backend_rejects_role_mismatch(tmp_path):
    backend = DummyCustomRoleModelBackend(
        backend_id="custom_jazz_melody_v001",
        role="melody",
        checkpoint_dir=_checkpoint(tmp_path / "jazz_melody_v001", role="melody"),
        output_dir=tmp_path / "raw",
    )

    with pytest.raises(ModelGenerationError, match="only supports role melody"):
        backend.generate(
            ModelGenerationRequest(
                request_id="wrong-role",
                task="generate_track",
                role_intent={"role": "walking_bass"},
                bars=[1],
            )
        )


def test_custom_role_non_commercial_manifest_blocks_commercial_export(tmp_path):
    checkpoint_dir = _checkpoint(
        tmp_path / "jazz_horns_v001",
        role="horn_responses",
        license_name="CC-BY-NC",
        commercial_training="non_commercial",
    )
    backend = DummyCustomRoleModelBackend(
        backend_id="custom_jazz_horn_responses_v001",
        role="horn_responses",
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "raw",
    )

    assert backend.is_available() is True
    assert backend.capabilities.commercial_use == "non_commercial"
    private_result = backend.generate(
        ModelGenerationRequest(
            request_id="private",
            task="generate_variation",
            role_intent={"role": "horn_response"},
            bars=[1],
            metadata={"export_mode": "private"},
        )
    )
    assert private_result.artifacts[0].metadata["commercial_allowed"] is False

    with pytest.raises(ModelBackendUnavailableError, match="export_mode=commercial"):
        backend.generate(
            ModelGenerationRequest(
                request_id="commercial",
                task="generate_variation",
                role_intent={"role": "horn_responses"},
                bars=[1],
                metadata={"export_mode": "commercial"},
            )
        )


def test_custom_role_registry_lists_available_backend_without_weight_loading(tmp_path):
    checkpoint_dir = _checkpoint(
        tmp_path / "jazz_drums_v001",
        role="drums",
        license_name="CC0-1.0",
        commercial_training="allowed",
    )
    config = AIModelsConfig(
        backends={
            "custom_jazz_drums_v001": BackendConfig(
                enabled=True,
                type="custom_role",
                adapter="model_backends.custom_role.dummy_backend.DummyCustomRoleModelBackend",
                role="drums",
                checkpoint_dir=str(checkpoint_dir),
                commercial_use="unknown",
                tasks=["generate_track", "infill_bars", "generate_variation"],
            )
        }
    )

    registry = build_model_backend_registry(config=config, include_unavailable=True)
    model = registry.list()[0]

    assert model["id"] == "custom_jazz_drums_v001"
    assert model["status"] == "available"
    assert model["commercial_use"] == "allowed"
    assert model["metadata"]["role"] == "drums"
    assert model["metadata"]["commercial_allowed"] is True


def test_statistical_custom_role_backend_generates_midi_and_tokens(tmp_path):
    train_custom_role_ngram_checkpoints(
        _role_segments(),
        tmp_path / "custom",
        seed=3410,
        ngram_order=3,
    )
    checkpoint_dir = tmp_path / "custom" / "melody" / "jazz_melody_v001"
    backend = StatisticalCustomRoleBackend(
        backend_id="custom_jazz_melody_v001",
        role="melody",
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "raw",
    )

    assert backend.is_available() is True
    assert backend.capabilities.token_output is True
    result = backend.generate(
        ModelGenerationRequest(
            request_id="statistical-unit",
            task="infill_bars",
            role_intent={"role": "melody", "density": "medium"},
            track_id="alto_sax",
            bars=[1, 2],
            seed=3411,
            metadata={"export_mode": "commercial"},
        )
    )

    artifacts = {artifact.artifact_type: artifact for artifact in result.artifacts}
    assert set(artifacts) == {"midi", "tokens"}
    assert Path(artifacts["midi"].path).exists()
    payload = json.loads(Path(artifacts["tokens"].path).read_text(encoding="utf-8"))
    assert payload["generation_source"] == "statistical_custom_role_model"
    assert payload["model"]["model_type"] == "custom_role_ngram"
    assert payload["target_tokens"][0] == "BOS"
    assert artifacts["midi"].metadata["note_count"] > 0


def test_statistical_custom_role_drums_emit_supported_drum_pitches(tmp_path):
    train_custom_role_ngram_checkpoints(
        _role_segments(),
        tmp_path / "custom",
        seed=3412,
        ngram_order=3,
    )
    checkpoint_dir = tmp_path / "custom" / "drums" / "jazz_drums_v001"
    backend = StatisticalCustomRoleBackend(
        backend_id="custom_jazz_drums_v001",
        role="drums",
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "raw",
    )

    result = backend.generate(
        ModelGenerationRequest(
            request_id="statistical-drums-unit",
            task="infill_bars",
            role_intent={"role": "drums", "density": "medium"},
            track_id="drum_kit",
            bars=[11, 12],
            seed=3413,
            metadata={"export_mode": "commercial"},
        )
    )
    midi_artifact = next(
        artifact for artifact in result.artifacts if artifact.artifact_type == "midi"
    )
    midi_file = mido.MidiFile(midi_artifact.path)
    pitches = {
        int(message.note)
        for track in midi_file.tracks
        for message in track
        if not message.is_meta
        and message.type == "note_on"
        and int(getattr(message, "velocity", 0)) > 0
    }

    assert pitches <= {36, 38, 42, 44, 45, 47, 49, 50, 51}


def _checkpoint(
    path: Path,
    *,
    role: str,
    license_name: str = "CC0-1.0",
    commercial_training: str = "allowed",
) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"dummy checkpoint bytes")
    (path / "tokenizer.json").write_text('{"tokenizer": "dummy"}\n', encoding="utf-8")
    (path / "config.yaml").write_text(
        yaml.safe_dump({"role": role, "model_type": "dummy_custom_role"}),
        encoding="utf-8",
    )
    datasets = [
        {
            "dataset_id": "synthetic_pr25",
            "license": license_name,
            "commercial_training": commercial_training,
            "train_eligible": True,
        }
    ]
    (path / "training_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.1.0",
                "role": role,
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    (path / "license_report.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "status": "pass",
                "sources": datasets,
                "rejected_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"schema_version": "0.1.0", "role": role, "segment_count": 1}) + "\n",
        encoding="utf-8",
    )
    return path


def _role_segments() -> list[RoleTrainingSegment]:
    roles = ("melody", "walking_bass", "piano_comping", "horn_responses", "drums")
    segments: list[RoleTrainingSegment] = []
    for role in roles:
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
