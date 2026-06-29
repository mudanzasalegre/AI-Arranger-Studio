from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from arranger_core.schema import ArrangementProject

ReleaseExportMode = Literal["private", "commercial"]
RELEASE_GATE_VERSION = "0.1.0"
PASS_STATUSES = {"pass", "pass_with_warnings"}
BLOCKED_LICENSES = {"", "unknown", "proprietary", "all-rights-reserved", "all rights reserved"}


def validate_release_quality(
    project: ArrangementProject,
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    export_mode: ReleaseExportMode | None = None,
) -> dict[str, Any]:
    """Validate final-release constraints that go beyond MIDI/MusicXML exportability."""

    output_root = Path(output_dir)
    mode = _export_mode(export_mode)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "release_gate_version": RELEASE_GATE_VERSION,
        "export_mode": mode,
    }

    errors.extend(_project_validation_errors(project))
    errors.extend(_export_manifest_errors(manifest))
    take_errors, take_metrics = _take_acceptance_errors(output_root)
    errors.extend(take_errors)
    metrics.update(take_metrics)

    trace = _read_json(output_root / "model_trace.json")
    model_errors, model_metrics = _model_trace_errors(trace, export_mode=mode)
    errors.extend(model_errors)
    metrics.update(model_metrics)

    dataset_errors, dataset_warnings, dataset_metrics = _dataset_manifest_issues(
        project,
        export_mode=mode,
    )
    errors.extend(dataset_errors)
    warnings.extend(dataset_warnings)
    metrics.update(dataset_metrics)

    return _report(project.project_id, errors, warnings, metrics=metrics)


def _project_validation_errors(project: ArrangementProject) -> list[dict[str, Any]]:
    report = project.validation_report or {}
    if report.get("status") in {"", None}:
        return []
    if report.get("status") in PASS_STATUSES and not report.get("errors"):
        return []
    return [
        _issue(
            "error",
            "ReleaseQualityGate",
            "blocking_project_validation",
            "Project validation contains blocking errors",
            details={
                "validation_status": report.get("status"),
                "errors": len(report.get("errors", [])),
            },
        )
    ]


def _export_manifest_errors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if manifest.get("status") != "exported":
        errors.append(
            _issue(
                "error",
                "ReleaseQualityGate",
                "export_not_ready",
                "Export manifest is not marked as exported",
                details={"status": manifest.get("status")},
            )
        )
    available_kinds = {str(file_record.get("kind")) for file_record in manifest.get("files", [])}
    for kind in ("model_trace_json", "takes_manifest_json", "session_readme"):
        if kind not in available_kinds:
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "missing_release_file",
                    f"Release package is missing required file kind {kind!r}",
                    details={"kind": kind},
                )
            )
    return errors


def _take_acceptance_errors(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_manifest_path = output_root / "takes" / "takes_manifest.json"
    exported_manifest_path = output_root / "takes_manifest.json"
    source = _read_json(source_manifest_path)
    exported = _read_json(exported_manifest_path)
    errors: list[dict[str, Any]] = []
    source_takes = _take_list(source)
    exported_takes = _take_list(exported)
    metrics = {
        "source_take_count": len(source_takes),
        "exported_take_count": len(exported_takes),
        "pending_take_count": sum(1 for take in source_takes if take.get("status") == "pending"),
        "rejected_take_count": sum(1 for take in source_takes if take.get("status") == "rejected"),
    }

    active_take_id = source.get("active_take_id") or exported.get("active_take_id")
    if source and active_take_id:
        active = next(
            (take for take in source_takes if take.get("take_id") == active_take_id),
            None,
        )
        if active is None or active.get("status") != "accepted":
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "active_take_not_accepted",
                    "Active take must be accepted before final export",
                    details={"active_take_id": active_take_id},
                )
            )

    for take in source_takes:
        status = take.get("status")
        if status == "pending":
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "pending_take_present",
                    "Pending takes block final release export",
                    details={"take_id": take.get("take_id")},
                )
            )
        elif status == "rejected" and not _rejected_take_has_report(take):
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "rejected_take_without_report",
                    "Rejected take must keep rejection or validation report metadata",
                    details={"take_id": take.get("take_id")},
                )
            )
        elif status == "accepted" and not _accepted_take_validation_ok(take):
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "accepted_take_validation_not_passed",
                    "Accepted take does not have passing validation metadata",
                    details={
                        "take_id": take.get("take_id"),
                        "validation_status": _take_validation_status(take),
                    },
                )
            )

    for take in exported_takes:
        if take.get("status") != "accepted":
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "non_accepted_take_exported",
                    "Final exported takes manifest must contain only accepted takes",
                    details={"take_id": take.get("take_id"), "status": take.get("status")},
                )
            )
        if take.get("project_snapshot_path"):
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "take_snapshot_path_exported",
                    "Final takes manifest must not expose internal snapshot paths",
                    details={"take_id": take.get("take_id")},
                )
            )
    return errors, metrics


def _rejected_take_has_report(take: dict[str, Any]) -> bool:
    metadata = take.get("metadata") if isinstance(take.get("metadata"), dict) else {}
    return bool(
        metadata.get("rejection_reason")
        or metadata.get("validation_report_path")
        or metadata.get("validation_status")
    )


def _accepted_take_validation_ok(take: dict[str, Any]) -> bool:
    if take.get("source") == "rule_based":
        return True
    status = _take_validation_status(take)
    return status in PASS_STATUSES


def _take_validation_status(take: dict[str, Any]) -> str | None:
    metadata = take.get("metadata") if isinstance(take.get("metadata"), dict) else {}
    return metadata.get("validation_status")


def _model_trace_errors(
    trace: dict[str, Any],
    *,
    export_mode: ReleaseExportMode,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    artifacts = [
        artifact
        for artifact in trace.get("model_artifacts", [])
        if isinstance(artifact, dict)
    ]
    metrics = {
        "model_artifact_count": len(artifacts),
        "commercial_model_artifact_count": sum(
            1 for artifact in artifacts if artifact.get("commercial_use") == "allowed"
        ),
    }
    for artifact in artifacts:
        task = str(artifact.get("task", "")).lower()
        backend = str(artifact.get("backend_id", "")).lower()
        if any(token in task or token in backend for token in ("audio", "wav", "mp3")):
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "audio_backend_in_release",
                    "Audio backends are not allowed in symbolic MIDI release exports",
                    details={"take_id": artifact.get("take_id"), "backend_id": backend},
                )
            )
        commercial_use = str(artifact.get("commercial_use", "unknown"))
        if commercial_use == "forbidden":
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "model_license_forbidden",
                    "Model artifact is marked forbidden for release use",
                    details={"take_id": artifact.get("take_id")},
                )
            )
        elif export_mode == "commercial" and commercial_use != "allowed":
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "model_license_incompatible",
                    "Commercial export requires model artifacts with commercial_use=allowed",
                    details={
                        "take_id": artifact.get("take_id"),
                        "commercial_use": commercial_use,
                    },
                )
            )
    return errors, metrics


def _dataset_manifest_issues(
    project: ArrangementProject,
    *,
    export_mode: ReleaseExportMode,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    paths = _dataset_manifest_paths(project)
    metrics = {
        "dataset_manifest_count": len(paths),
        "dataset_manifest_entries": 0,
    }
    for manifest_path in paths:
        path = Path(manifest_path)
        if not path.exists():
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "dataset_manifest_missing",
                    "Declared dataset manifest does not exist",
                    details={"path": str(path)},
                )
            )
            continue
        payload = _read_json(path)
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            errors.append(
                _issue(
                    "error",
                    "ReleaseQualityGate",
                    "dataset_manifest_invalid",
                    "Dataset manifest must contain an entries list",
                    details={"path": str(path)},
                )
            )
            continue
        metrics["dataset_manifest_entries"] += len(entries)
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            license_name = str(entry.get("license", "")).strip().lower()
            if license_name in BLOCKED_LICENSES:
                errors.append(
                    _issue(
                        "error",
                        "ReleaseQualityGate",
                        "dataset_license_blocked",
                        "Dataset manifest entry has missing or blocked license",
                        details={"path": str(path), "index": index, "license": license_name},
                    )
                )
            if str(entry.get("license_confidence", "low")) == "low":
                warnings.append(
                    _issue(
                        "warning",
                        "ReleaseQualityGate",
                        "dataset_license_low_confidence",
                        "Dataset license confidence is low",
                        details={"path": str(path), "index": index},
                    )
                )
            if export_mode == "commercial":
                if entry.get("commercial_training") != "allowed" or entry.get(
                    "local_learning_only"
                ):
                    errors.append(
                        _issue(
                            "error",
                            "ReleaseQualityGate",
                            "dataset_commercial_use_incompatible",
                            (
                                "Commercial export requires dataset manifest entries "
                                "approved for commercial use"
                            ),
                            details={
                                "path": str(path),
                                "index": index,
                                "commercial_training": entry.get("commercial_training"),
                                "local_learning_only": entry.get("local_learning_only"),
                            },
                        )
                    )
    return errors, warnings, metrics


def _dataset_manifest_paths(project: ArrangementProject) -> list[str]:
    candidates: list[Any] = []
    metadata = project.metadata if isinstance(project.metadata, dict) else {}
    candidates.extend(
        [
            metadata.get("dataset_manifest_path"),
            metadata.get("dataset_manifest_paths"),
            metadata.get("dataset_manifests"),
        ]
    )
    if project.generation_spec is not None:
        constraints = project.generation_spec.constraints
        candidates.extend(
            [
                constraints.get("dataset_manifest_path"),
                constraints.get("dataset_manifest_paths"),
                constraints.get("dataset_manifests"),
            ]
        )
    paths: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            paths.append(candidate)
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.append(str(item["path"]))
        elif isinstance(candidate, dict) and isinstance(candidate.get("path"), str):
            paths.append(str(candidate["path"]))
    return sorted(set(paths))


def _take_list(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [take for take in manifest.get("takes", []) if isinstance(take, dict)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _export_mode(export_mode: ReleaseExportMode | None) -> ReleaseExportMode:
    raw = (export_mode or os.environ.get("EXPORT_MODE") or "private").strip().lower()
    return "commercial" if raw == "commercial" else "private"


def _report(
    project_id: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    status = "fail" if errors else "pass_with_warnings" if warnings else "pass"
    return {
        "status": status,
        "project_id": project_id,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            **metrics,
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }


def _issue(
    severity: Literal["error", "warning"],
    validator: str,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "validator": validator,
        "code": code,
        "message": message,
        "details": details or {},
    }
