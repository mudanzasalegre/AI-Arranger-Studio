from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from arranger_core.performance import PerformanceMapper
from arranger_core.schema import ArrangementProject
from arranger_core.takes.models import ArrangementTake, ModelArtifactRecord

TAKES_MANIFEST_VERSION = "0.1.0"
BASE_TAKE_ID = "take_base"


class TakeManager:
    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)
        self.takes_dir = self.project_dir / "takes"
        self.takes_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.takes_dir / "takes_manifest.json"

    def ensure_base_take(self, project: ArrangementProject) -> ArrangementTake:
        manifest = self._load_manifest()
        for take in manifest["takes"]:
            if take.take_id == BASE_TAKE_ID:
                return take

        snapshot_path = self.takes_dir / BASE_TAKE_ID / "arrangement_project.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        project.save_json(snapshot_path)
        take = ArrangementTake(
            take_id=BASE_TAKE_ID,
            project_id=project.project_id,
            source="rule_based",
            status="accepted",
            project_snapshot_path=str(snapshot_path),
            created_at=_utc_now(),
            updated_at=_utc_now(),
            metadata={"label": "Initial rule-based arrangement"},
        )
        manifest["active_take_id"] = BASE_TAKE_ID
        manifest["takes"].append(take)
        self._write_manifest(manifest)
        return take

    def list_takes(self, *, project: ArrangementProject | None = None) -> dict[str, Any]:
        if project is not None:
            self.ensure_base_take(project)
        manifest = self._load_manifest()
        takes = sorted(manifest["takes"], key=lambda item: item.created_at)
        return {
            "schema_version": TAKES_MANIFEST_VERSION,
            "project_id": manifest.get("project_id"),
            "active_take_id": manifest.get("active_take_id"),
            "count": len(takes),
            "takes": [take.model_dump(mode="json") for take in takes],
        }

    def create_pending_take(
        self,
        *,
        base_project: ArrangementProject,
        candidate_project: ArrangementProject,
        artifact_records: list[ModelArtifactRecord],
        validation_report: dict[str, Any],
        parent_take_id: str | None = None,
        track_id: str | None = None,
        bars: list[int] | None = None,
        instruction: str | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArrangementTake:
        self.ensure_base_take(base_project)
        if validation_report.get("status") == "fail":
            raise ValueError("Cannot create pending take from failing validation report")

        take_id = f"take_{uuid4().hex[:12]}"
        take_dir = self.takes_dir / take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = take_dir / "arrangement_project.json"
        validation_path = take_dir / "validation_report.json"
        candidate_project.metadata = {
            **candidate_project.metadata,
            "take_id": take_id,
            "take_status": "pending",
        }
        candidate_project.save_json(snapshot_path)
        validation_path.write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")

        first_record = artifact_records[0] if artifact_records else None
        take = ArrangementTake(
            take_id=take_id,
            project_id=base_project.project_id,
            parent_take_id=parent_take_id or self._load_manifest().get("active_take_id"),
            source="model",
            backend_id=first_record.backend_id if first_record else None,
            task=first_record.task if first_record else None,
            track_id=track_id,
            bars=bars or [],
            instruction=instruction,
            seed=seed,
            status="pending",
            validation_report_id=take_id,
            artifact_ids=[record.artifact_id for record in artifact_records],
            project_snapshot_path=str(snapshot_path),
            created_at=_utc_now(),
            updated_at=_utc_now(),
            metadata={
                "validation_report_path": str(validation_path),
                "validation_status": validation_report.get("status"),
                **(metadata or {}),
            },
        )
        manifest = self._load_manifest()
        manifest["takes"] = [item for item in manifest["takes"] if item.take_id != take.take_id]
        manifest["takes"].append(take)
        manifest["project_id"] = base_project.project_id
        self._write_manifest(manifest)
        return take

    def accept_take(self, take_id: str) -> tuple[ArrangementTake, ArrangementProject]:
        manifest = self._load_manifest()
        take = self._find_take(manifest, take_id)
        if take.status == "rejected":
            raise ValueError(f"Cannot accept rejected take: {take_id}")
        if take.project_snapshot_path is None:
            raise ValueError(f"Take has no project snapshot: {take_id}")

        project = ArrangementProject.load_json(take.project_snapshot_path)
        project = PerformanceMapper().apply(project, default_source="rule_based")
        project.metadata = {
            **project.metadata,
            "active_take_id": take_id,
            "take_status": "accepted",
        }
        project.save_json(self.project_dir / "arrangement_project.json")
        updated_take = take.model_copy(
            update={
                "status": "accepted",
                "updated_at": _utc_now(),
                "metadata": {**take.metadata, "accepted_at": _utc_now()},
            }
        )
        manifest["active_take_id"] = take_id
        manifest["takes"] = [
            updated_take if item.take_id == take_id else item
            for item in manifest["takes"]
        ]
        self._write_manifest(manifest)
        return updated_take, project

    def reject_take(self, take_id: str, *, reason: str | None = None) -> ArrangementTake:
        manifest = self._load_manifest()
        take = self._find_take(manifest, take_id)
        if take_id == manifest.get("active_take_id"):
            raise ValueError(f"Cannot reject active take: {take_id}")
        updated_take = take.model_copy(
            update={
                "status": "rejected",
                "updated_at": _utc_now(),
                "metadata": {
                    **take.metadata,
                    "rejected_at": _utc_now(),
                    "rejection_reason": reason,
                },
            }
        )
        manifest["takes"] = [
            updated_take if item.take_id == take_id else item
            for item in manifest["takes"]
        ]
        self._write_manifest(manifest)
        return updated_take

    def _find_take(self, manifest: dict[str, Any], take_id: str) -> ArrangementTake:
        for take in manifest["takes"]:
            if take.take_id == take_id:
                return take
        raise KeyError(f"Take not found: {take_id}")

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {
                "schema_version": TAKES_MANIFEST_VERSION,
                "project_id": None,
                "active_take_id": None,
                "takes": [],
            }
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {
            "schema_version": payload.get("schema_version", TAKES_MANIFEST_VERSION),
            "project_id": payload.get("project_id"),
            "active_take_id": payload.get("active_take_id"),
            "takes": [
                ArrangementTake.model_validate(item)
                for item in payload.get("takes", [])
            ],
        }

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = {
            "schema_version": TAKES_MANIFEST_VERSION,
            "project_id": manifest.get("project_id"),
            "active_take_id": manifest.get("active_take_id"),
            "takes": [
                take.model_dump(mode="json")
                for take in manifest.get("takes", [])
            ],
        }
        self._manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
