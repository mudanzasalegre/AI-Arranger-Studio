from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from model_backends.base import ArtifactType, ModelArtifact, ModelTask

DEFAULT_ARTIFACT_OUTPUT_DIR = Path("outputs/model_artifacts/raw")
MAX_ARTIFACT_STEM_LENGTH = 72
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def ensure_artifact_dir(output_dir: str | Path | None = None) -> Path:
    artifact_dir = Path(output_dir or DEFAULT_ARTIFACT_OUTPUT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def safe_artifact_stem(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts if part is not None and str(part) != "")
    cleaned = _SAFE_CHARS_RE.sub("_", raw).strip("._")
    cleaned = cleaned or "artifact"
    if len(cleaned) <= MAX_ARTIFACT_STEM_LENGTH:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
    prefix = cleaned[: MAX_ARTIFACT_STEM_LENGTH - len(digest) - 1].rstrip("._")
    return f"{prefix}_{digest}"


def artifact_path(
    output_dir: str | Path | None,
    *,
    backend_id: str,
    task: ModelTask,
    seed: int | None,
    suffix: str,
    request_id: str = "",
) -> Path:
    artifact_dir = ensure_artifact_dir(output_dir)
    stem = safe_artifact_stem(backend_id, task, request_id or f"seed_{seed or 0}")
    return artifact_dir / f"{stem}{suffix}"


def write_json_artifact(
    path: str | Path,
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> ModelArtifact:
    artifact_path_value = Path(path)
    artifact_path_value.parent.mkdir(parents=True, exist_ok=True)
    artifact_path_value.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ModelArtifact(
        artifact_type="json",
        path=str(artifact_path_value),
        metadata=metadata or {},
    )


def artifact_record(
    artifact_type: ArtifactType,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> ModelArtifact:
    return ModelArtifact(
        artifact_type=artifact_type,
        path=str(path),
        metadata=metadata or {},
    )
