from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_backends.artifact import artifact_path, artifact_record
from model_backends.base import (
    ModelCapabilities,
    ModelGenerationRequest,
    ModelGenerationResult,
    ModelTask,
)
from model_backends.custom_role.base import CustomRoleModelBackend
from model_backends.custom_role.loader import (
    CUSTOM_ROLE_MODEL_VERSION,
    CustomRoleModelSpec,
    canonical_custom_role,
    inspect_custom_role_model,
)
from model_backends.errors import (
    ModelBackendUnavailableError,
    ModelGenerationError,
    UnsupportedModelTaskError,
)

CUSTOM_ROLE_TASKS: set[ModelTask] = {
    "generate_track",
    "infill_bars",
    "generate_variation",
}


class DummyCustomRoleModelBackend(CustomRoleModelBackend):
    """Safe bootstrap backend for a future locally trained single-role checkpoint."""

    backend_version = CUSTOM_ROLE_MODEL_VERSION

    def __init__(
        self,
        *,
        backend_id: str,
        role: str,
        checkpoint_dir: str | Path,
        model_file: str = "model.safetensors",
        tokenizer_file: str = "tokenizer.json",
        config_file: str = "config.yaml",
        training_manifest_file: str = "training_manifest.yaml",
        license_report_file: str = "license_report.json",
        output_dir: str | Path = "outputs/model_artifacts/raw",
        **_: Any,
    ) -> None:
        self.backend_id = backend_id
        self.role = canonical_custom_role(role)
        self.output_dir = Path(output_dir)
        self.spec = CustomRoleModelSpec(
            backend_id=backend_id,
            role=self.role,
            checkpoint_dir=str(checkpoint_dir),
            model_file=model_file,
            tokenizer_file=tokenizer_file,
            config_file=config_file,
            training_manifest_file=training_manifest_file,
            license_report_file=license_report_file,
        )
        self.inspection = inspect_custom_role_model(self.spec)
        self.capabilities = ModelCapabilities(
            symbolic_midi=False,
            multitrack=False,
            bar_infill=True,
            track_generation=True,
            text_prompt=False,
            json_planning=False,
            token_output=True,
            supports_training=True,
            commercial_use=self.inspection.commercial_use,
        )

    def is_available(self) -> bool:
        return self.inspection.available

    @property
    def unavailable_reason(self) -> str:
        return self.inspection.unavailable_reason

    @property
    def registry_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "checkpoint_dir": self.inspection.checkpoint_dir,
            "model_path": self.inspection.model_path,
            "training_manifest_path": self.inspection.training_manifest_path,
            "license_report_path": self.inspection.license_report_path,
            "commercial_allowed": self.inspection.commercial_allowed,
            "dataset_count": self.inspection.dataset_count,
            "rejected_source_count": self.inspection.rejected_source_count,
            "warnings": self.inspection.warnings,
        }

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task not in CUSTOM_ROLE_TASKS:
            raise UnsupportedModelTaskError(
                f"{self.backend_id} does not support task {request.task}"
            )
        if not self.inspection.available:
            raise ModelBackendUnavailableError(
                f"Custom role model unavailable: {self.inspection.unavailable_reason}"
            )

        requested_role = _requested_role(request)
        if requested_role != self.role:
            raise ModelGenerationError(
                f"{self.backend_id} only supports role {self.role}; got {requested_role}"
            )
        export_mode = str(request.metadata.get("export_mode") or "private")
        if export_mode == "commercial" and not self.inspection.commercial_allowed:
            raise ModelBackendUnavailableError(
                f"{self.backend_id} is not allowed for export_mode=commercial"
            )

        target_tokens = _dummy_tokens(self.role, request)
        path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".tokens.json",
            request_id=request.request_id,
        )
        payload = {
            "schema_version": CUSTOM_ROLE_MODEL_VERSION,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "role": self.role,
            "task": request.task,
            "target_tokens": target_tokens,
            "role_intent": request.role_intent or {},
            "bars": request.bars or [],
            "track_id": request.track_id,
            "export_mode": export_mode,
            "model": {
                "checkpoint_dir": self.inspection.checkpoint_dir,
                "training_manifest_path": self.inspection.training_manifest_path,
                "license_report_path": self.inspection.license_report_path,
                "commercial_allowed": self.inspection.commercial_allowed,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ModelGenerationResult(
            backend_id=self.backend_id,
            task=request.task,
            artifacts=[
                artifact_record(
                    "tokens",
                    path,
                    metadata={
                        "backend_id": self.backend_id,
                        "role": self.role,
                        "task": request.task,
                        "valid": True,
                        "token_count": len(target_tokens),
                        "commercial_allowed": self.inspection.commercial_allowed,
                    },
                )
            ],
            confidence=0.5,
            warnings=[
                "dummy_custom_role_backend_no_trained_inference",
                *self.inspection.warnings,
            ],
            raw_metadata={
                "backend_version": self.backend_version,
                "checkpoint_dir": self.inspection.checkpoint_dir,
                "tokenizer_path": self.inspection.tokenizer_path,
                "training_manifest_path": self.inspection.training_manifest_path,
                "license_report_path": self.inspection.license_report_path,
                "commercial_allowed": self.inspection.commercial_allowed,
            },
        )


def _requested_role(request: ModelGenerationRequest) -> str:
    role_intent = request.role_intent or {}
    raw_role = (
        role_intent.get("role")
        or request.metadata.get("target_role")
        or request.metadata.get("role")
        or ""
    )
    return canonical_custom_role(str(raw_role))


def _dummy_tokens(role: str, request: ModelGenerationRequest) -> list[str]:
    bars = request.bars or [1]
    density = str((request.role_intent or {}).get("density") or request.density or "medium")
    tokens = [
        "BOS",
        f"ROLE={role}",
        f"TASK={request.task}",
        f"DENSITY={density}",
    ]
    for bar in bars:
        tokens.append(f"BAR={bar}")
        if role == "drums":
            tokens.extend(
                [
                    f"DRUM|bar={bar}|start=0|piece=kick",
                    f"DRUM|bar={bar}|start=2|piece=snare",
                ]
            )
        elif role == "walking_bass":
            tokens.append(f"NOTE|bar={bar}|start=0|duration=1|pitch=C2|velocity=72")
        elif role == "piano_comping":
            tokens.append(f"CHORD|bar={bar}|start=0.5|duration=0.75|voicing=Cm7_shell")
        elif role == "horn_responses":
            tokens.append(f"NOTE|bar={bar}|start=2|duration=0.5|pitch=G4|velocity=78")
        else:
            tokens.append(f"NOTE|bar={bar}|start=0|duration=0.75|pitch=C4|velocity=80")
    tokens.append("EOS")
    return tokens
