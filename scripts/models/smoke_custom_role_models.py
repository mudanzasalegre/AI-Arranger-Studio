from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for package in ("model_backends",):
    sys.path.insert(0, str(ROOT / "packages" / package))

from model_backends import (  # noqa: E402
    ModelBackendUnavailableError,
    ModelGenerationRequest,
    build_model_backend_registry,
    load_ai_models_config,
)
from model_backends.config import AIModelsConfig, BackendConfig  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "model_smoke" / "custom_role_models"


def main() -> None:
    outputs_root = (ROOT / "outputs").resolve()
    smoke_root = OUTPUT_ROOT.resolve()
    if outputs_root not in smoke_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {smoke_root}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    default_config = load_ai_models_config(ROOT / "configs" / "ai_models.yaml")
    default_registry = build_model_backend_registry(
        config=default_config,
        include_disabled=True,
        include_unavailable=True,
    )
    default_models = {model["id"]: model for model in default_registry.list()}
    custom_unavailable = [
        model_id
        for model_id, model in default_models.items()
        if model_id.startswith("custom_") and model["status"] == "unavailable"
    ]
    if len(custom_unavailable) != 5:
        raise RuntimeError(f"Expected 5 unavailable custom models, got {custom_unavailable}")

    allowed_checkpoint = _checkpoint(
        smoke_root / "checkpoints" / "jazz_melody_v001",
        role="melody",
        license_name="CC0-1.0",
        commercial_training="allowed",
    )
    non_commercial_checkpoint = _checkpoint(
        smoke_root / "checkpoints" / "jazz_horns_v001",
        role="horn_responses",
        license_name="CC-BY-NC",
        commercial_training="non_commercial",
    )

    config = AIModelsConfig(
        backends={
            "custom_jazz_melody_v001": _backend_config(
                role="melody",
                checkpoint_dir=allowed_checkpoint,
            ),
            "custom_jazz_horn_responses_v001": _backend_config(
                role="horn_responses",
                checkpoint_dir=non_commercial_checkpoint,
            ),
        },
        settings={"artifact_raw_dir": str(smoke_root / "raw")},
    )
    registry = build_model_backend_registry(config=config, include_unavailable=True)
    models = {model["id"]: model for model in registry.list()}
    if models["custom_jazz_melody_v001"]["status"] != "available":
        raise RuntimeError("Allowed custom melody checkpoint did not register as available")
    if models["custom_jazz_horn_responses_v001"]["commercial_use"] != "non_commercial":
        raise RuntimeError("Non-commercial custom horn checkpoint was not marked correctly")

    melody_backend = registry.get("custom_jazz_melody_v001")
    melody_result = melody_backend.generate(
        ModelGenerationRequest(
            request_id="smoke_melody",
            task="generate_track",
            role_intent={"role": "melody", "density": "medium"},
            bars=[1, 2],
            metadata={"export_mode": "commercial"},
        )
    )
    if melody_result.artifacts[0].artifact_type != "tokens":
        raise RuntimeError("Custom role dummy backend did not write token artifact")

    horn_backend = registry.get("custom_jazz_horn_responses_v001")
    commercial_blocked = False
    try:
        horn_backend.generate(
            ModelGenerationRequest(
                request_id="smoke_horns_commercial",
                task="generate_variation",
                role_intent={"role": "horn_responses"},
                bars=[1],
                metadata={"export_mode": "commercial"},
            )
        )
    except ModelBackendUnavailableError:
        commercial_blocked = True
    if not commercial_blocked:
        raise RuntimeError("Non-commercial custom role model was allowed for commercial export")

    report = {
        "status": "ok",
        "default_unavailable_custom_models": sorted(custom_unavailable),
        "available_smoke_models": {
            model_id: {
                "status": model["status"],
                "commercial_use": model["commercial_use"],
                "role": model["metadata"]["role"],
            }
            for model_id, model in models.items()
        },
        "melody_artifact": melody_result.artifacts[0].path,
        "commercial_blocked": commercial_blocked,
    }
    report_path = ROOT / "outputs/model_smoke/custom_role_model_smoke_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _backend_config(*, role: str, checkpoint_dir: Path) -> BackendConfig:
    return BackendConfig(
        enabled=True,
        type="custom_role",
        adapter="model_backends.custom_role.dummy_backend.DummyCustomRoleModelBackend",
        role=role,
        checkpoint_dir=str(checkpoint_dir),
        commercial_use="unknown",
        tasks=["generate_track", "infill_bars", "generate_variation"],
    )


def _checkpoint(
    path: Path,
    *,
    role: str,
    license_name: str,
    commercial_training: str,
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


if __name__ == "__main__":
    main()
