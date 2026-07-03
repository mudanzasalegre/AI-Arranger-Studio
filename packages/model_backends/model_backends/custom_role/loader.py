from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_backends.base import CommercialUse

CUSTOM_ROLE_MODEL_VERSION = "0.1.0"

CUSTOM_ROLE_ALIASES = {
    "bass": "walking_bass",
    "double_bass": "walking_bass",
    "horn_response": "horn_responses",
    "horns": "horn_responses",
    "piano": "piano_comping",
    "comping": "piano_comping",
}
SUPPORTED_CUSTOM_ROLES = {
    "melody",
    "walking_bass",
    "piano_comping",
    "horn_responses",
    "drums",
}
BLOCKED_TRAINING_LICENSES = {
    "",
    "unknown",
    "proprietary",
    "all rights reserved",
    "all-rights-reserved",
    "private",
}
BLOCKED_COMMERCIAL_LICENSES = {
    "research_only",
    "research-only",
    "research only",
    "non_commercial",
    "non-commercial",
    "noncommercial",
    "cc-by-nc",
    "cc-by-nc-sa",
}
BLOCKED_COMMERCIAL_FLAGS = {
    "blocked",
    "forbidden",
    "not_allowed",
    "research_only",
    "research-only",
    "non_commercial",
    "non-commercial",
}


class CustomRoleLoaderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CustomRoleModelSpec(CustomRoleLoaderModel):
    backend_id: str
    role: str
    checkpoint_dir: str
    model_file: str = "model.safetensors"
    tokenizer_file: str = "tokenizer.json"
    config_file: str = "config.yaml"
    training_manifest_file: str = "training_manifest.yaml"
    license_report_file: str = "license_report.json"


class CustomRoleModelInspection(CustomRoleLoaderModel):
    backend_id: str
    role: str
    checkpoint_dir: str
    available: bool
    commercial_allowed: bool
    commercial_use: CommercialUse
    missing_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_path: str
    tokenizer_path: str
    config_path: str
    training_manifest_path: str
    license_report_path: str
    training_manifest: dict[str, Any] = Field(default_factory=dict)
    license_report: dict[str, Any] = Field(default_factory=dict)
    dataset_count: int = 0
    rejected_source_count: int = 0

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        reasons = [*self.missing_files, *self.errors]
        return "; ".join(reasons) or "custom role model unavailable"


def inspect_custom_role_model(
    spec: CustomRoleModelSpec | dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> CustomRoleModelInspection:
    model_spec = spec if isinstance(spec, CustomRoleModelSpec) else CustomRoleModelSpec(**spec)
    role = canonical_custom_role(model_spec.role)
    root = Path(repo_root).resolve() if repo_root is not None else None
    checkpoint_dir = _resolve_path(model_spec.checkpoint_dir, root=root)
    model_path = checkpoint_dir / model_spec.model_file
    tokenizer_path = checkpoint_dir / model_spec.tokenizer_file
    config_path = checkpoint_dir / model_spec.config_file
    training_manifest_path = checkpoint_dir / model_spec.training_manifest_file
    license_report_path = checkpoint_dir / model_spec.license_report_file

    paths = {
        "model_file": model_path,
        "tokenizer_file": tokenizer_path,
        "config_file": config_path,
        "training_manifest_file": training_manifest_path,
        "license_report_file": license_report_path,
    }
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.exists()]

    errors: list[str] = []
    warnings: list[str] = []
    if role not in SUPPORTED_CUSTOM_ROLES:
        errors.append(f"unsupported_role:{model_spec.role}")

    config = _read_mapping(config_path) if config_path.exists() else {}
    training_manifest = (
        _read_mapping(training_manifest_path) if training_manifest_path.exists() else {}
    )
    license_report = _read_mapping(license_report_path) if license_report_path.exists() else {}

    for source_name, payload in (
        ("config", config),
        ("training_manifest", training_manifest),
    ):
        declared_role = _declared_role(payload)
        if declared_role and canonical_custom_role(declared_role) != role:
            errors.append(f"{source_name}_role_mismatch:{declared_role}!={role}")

    license_review = _review_license_payloads(training_manifest, license_report)
    errors.extend(license_review["errors"])
    warnings.extend(license_review["warnings"])
    commercial_allowed = bool(license_review["commercial_allowed"]) and not errors and not missing
    available = not missing and not errors
    commercial_use: CommercialUse
    if not available:
        commercial_use = "unknown"
    elif commercial_allowed:
        commercial_use = "allowed"
    else:
        commercial_use = "non_commercial"

    return CustomRoleModelInspection(
        backend_id=model_spec.backend_id,
        role=role,
        checkpoint_dir=str(checkpoint_dir),
        available=available,
        commercial_allowed=commercial_allowed,
        commercial_use=commercial_use,
        missing_files=missing,
        errors=errors,
        warnings=warnings,
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        config_path=str(config_path),
        training_manifest_path=str(training_manifest_path),
        license_report_path=str(license_report_path),
        training_manifest=training_manifest,
        license_report=license_report,
        dataset_count=int(license_review["dataset_count"]),
        rejected_source_count=int(license_review["rejected_source_count"]),
    )


def canonical_custom_role(role: str) -> str:
    normalized = role.strip().lower().replace("-", "_").replace(" ", "_")
    return CUSTOM_ROLE_ALIASES.get(normalized, normalized)


def _resolve_path(value: str, *, root: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or root is None:
        return path
    return root / path


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Unable to read custom model manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Custom model manifest must be a mapping: {path}")
    return payload


def _declared_role(payload: dict[str, Any]) -> str:
    for key in ("role", "target_role", "model_role"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    model = payload.get("model")
    if isinstance(model, dict):
        value = model.get("role") or model.get("target_role")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _review_license_payloads(
    training_manifest: dict[str, Any],
    license_report: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    commercial_allowed = True
    dataset_entries = [
        *_entries(training_manifest, keys=("datasets", "sources", "segments")),
        *_entries(license_report, keys=("datasets", "sources", "segments")),
    ]
    seen_entries = {
        json.dumps(entry, sort_keys=True, default=str)
        for entry in dataset_entries
    }
    dataset_entries = [json.loads(entry) for entry in seen_entries]

    for index, entry in enumerate(dataset_entries):
        split = str(entry.get("split", "train")).strip().lower()
        if split == "rejected":
            continue
        license_name = _normalize_flag(str(entry.get("license", "")))
        commercial_flag = _normalize_flag(
            str(
                entry.get("commercial_training")
                or entry.get("commercial_use")
                or entry.get("usage")
                or ""
            )
        )
        if entry.get("train_eligible") is False or entry.get("training_allowed") is False:
            errors.append(f"dataset_not_trainable:{index}")
        if license_name in BLOCKED_TRAINING_LICENSES:
            errors.append(f"dataset_blocked_license:{index}:{license_name}")
        if license_name in BLOCKED_COMMERCIAL_LICENSES:
            commercial_allowed = False
            warnings.append(f"dataset_non_commercial_license:{index}:{license_name}")
        if commercial_flag in BLOCKED_COMMERCIAL_FLAGS:
            commercial_allowed = False
            warnings.append(f"dataset_commercial_blocked:{index}:{commercial_flag}")

    if license_report.get("status") in {"fail", "failed", "error"}:
        errors.append(f"license_report_status:{license_report.get('status')}")
    report_errors = license_report.get("errors")
    if isinstance(report_errors, list) and report_errors:
        errors.append("license_report_errors")
    rejected_sources = license_report.get("rejected_sources")
    rejected_source_count = len(rejected_sources) if isinstance(rejected_sources, list) else 0
    if rejected_source_count:
        warnings.append(f"license_report_rejected_sources:{rejected_source_count}")

    return {
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "commercial_allowed": commercial_allowed,
        "dataset_count": len(dataset_entries),
        "rejected_source_count": rejected_source_count,
    }


def _entries(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    return entries


def _normalize_flag(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
