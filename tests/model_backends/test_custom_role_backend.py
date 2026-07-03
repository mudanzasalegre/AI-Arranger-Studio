from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from model_backends import (
    DummyCustomRoleModelBackend,
    ModelBackendUnavailableError,
    ModelGenerationError,
    ModelGenerationRequest,
    build_model_backend_registry,
    inspect_custom_role_model,
)
from model_backends.config import AIModelsConfig, BackendConfig


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
    assert len(inspection.missing_files) == 5
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
    return path
