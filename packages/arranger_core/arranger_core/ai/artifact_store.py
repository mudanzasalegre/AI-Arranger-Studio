from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arranger_core.takes.models import ModelArtifactRecord

if TYPE_CHECKING:
    from model_backends import ModelGenerationResult

ARTIFACT_STORE_VERSION = "0.1.0"
ARTIFACT_STATUS_DIRS = ("raw", "imported", "validated", "rejected")


class ArtifactStore:
    def __init__(self, root: str | Path = "outputs/model_artifacts") -> None:
        self.root = Path(root)
        for status_dir in ARTIFACT_STATUS_DIRS:
            (self.root / status_dir).mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "artifact_manifest.json"

    def store_generation_result(
        self,
        result: ModelGenerationResult,
        *,
        project_id: str | None,
    ) -> list[ModelArtifactRecord]:
        records: list[ModelArtifactRecord] = []
        for artifact in result.artifacts:
            artifact_id = f"artifact_{uuid4().hex[:12]}"
            source_path = Path(artifact.path)
            raw_path = self.root / "raw" / f"{artifact_id}{source_path.suffix or '.artifact'}"
            if not source_path.exists():
                raise FileNotFoundError(f"Model artifact does not exist: {source_path}")
            if source_path.resolve() != raw_path.resolve():
                shutil.copy2(source_path, raw_path)
            record = ModelArtifactRecord(
                artifact_id=artifact_id,
                project_id=project_id,
                backend_id=result.backend_id,
                task=result.task,
                artifact_type=artifact.artifact_type,
                raw_path=str(raw_path),
                status="raw",
                created_at=_utc_now(),
                metadata={
                    **artifact.metadata,
                    "result_warnings": result.warnings,
                    "result_confidence": result.confidence,
                    "raw_metadata": result.raw_metadata,
                },
            )
            records.append(record)
            self._upsert_record(record)
        return records

    def list_records(self, *, project_id: str | None = None) -> list[ModelArtifactRecord]:
        records = self._load_manifest()
        if project_id is None:
            return records
        return [record for record in records if record.project_id == project_id]

    def get(self, artifact_id: str) -> ModelArtifactRecord:
        for record in self._load_manifest():
            if record.artifact_id == artifact_id:
                return record
        raise KeyError(f"Artifact not found: {artifact_id}")

    def mark_imported(
        self,
        record: ModelArtifactRecord,
        *,
        imported_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifactRecord:
        return self._update_record(
            record,
            status="imported",
            imported_path=str(imported_path),
            metadata=metadata,
        )

    def mark_validated(
        self,
        record: ModelArtifactRecord,
        *,
        validated_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifactRecord:
        return self._update_record(
            record,
            status="validated",
            validated_path=str(validated_path) if validated_path else record.validated_path,
            metadata=metadata,
        )

    def mark_rejected(
        self,
        record: ModelArtifactRecord,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifactRecord:
        rejected_path = self.root / "rejected" / Path(record.raw_path).name
        raw_path = Path(record.raw_path)
        if raw_path.exists() and raw_path.resolve() != rejected_path.resolve():
            shutil.copy2(raw_path, rejected_path)
        return self._update_record(
            record,
            status="rejected",
            rejected_path=str(rejected_path),
            metadata={"rejection_reason": reason, **(metadata or {})},
        )

    def _update_record(
        self,
        record: ModelArtifactRecord,
        *,
        status: str,
        imported_path: str | None = None,
        validated_path: str | None = None,
        rejected_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifactRecord:
        updated = record.model_copy(
            update={
                "status": status,
                "imported_path": (
                    imported_path if imported_path is not None else record.imported_path
                ),
                "validated_path": (
                    validated_path if validated_path is not None else record.validated_path
                ),
                "rejected_path": (
                    rejected_path if rejected_path is not None else record.rejected_path
                ),
                "metadata": {**record.metadata, **(metadata or {})},
            }
        )
        self._upsert_record(updated)
        return updated

    def _upsert_record(self, record: ModelArtifactRecord) -> None:
        records = [
            item
            for item in self._load_manifest()
            if item.artifact_id != record.artifact_id
        ]
        records.append(record)
        records.sort(key=lambda item: item.created_at)
        self._write_manifest(records)

    def _load_manifest(self) -> list[ModelArtifactRecord]:
        if not self._manifest_path.exists():
            return []
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return [ModelArtifactRecord.model_validate(item) for item in data.get("artifacts", [])]

    def _write_manifest(self, records: list[ModelArtifactRecord]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ARTIFACT_STORE_VERSION,
            "artifacts": [record.model_dump(mode="json") for record in records],
        }
        self._manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
