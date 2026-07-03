from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "models"))

import ai_local_smoke  # noqa: E402


def test_should_run_backend_uses_enabled_status_force_and_skip():
    models = {
        "midigpt": {"enabled": True, "status": "available"},
        "text2midi": {"enabled": False, "status": "disabled"},
        "local_llm_planner": {"enabled": True, "status": "unavailable"},
    }

    assert ai_local_smoke._should_run_backend(models, "midigpt") is True
    assert ai_local_smoke._should_run_backend(models, "text2midi") is False
    assert ai_local_smoke._should_run_backend(models, "text2midi", force=True) is True
    assert ai_local_smoke._should_run_backend(models, "midigpt", skip=True) is False
    assert ai_local_smoke._should_run_backend(models, "local_llm_planner") is True


def test_assert_final_artifact_statuses_rejects_raw(tmp_path):
    artifact_root = tmp_path / "model_artifacts"
    artifact_root.mkdir()
    (artifact_root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "artifacts": [
                    {"artifact_id": "a1", "status": "validated"},
                    {"artifact_id": "a2", "status": "rejected"},
                    {"artifact_id": "a3", "status": "raw"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert ai_local_smoke._assert_final_artifact_statuses(
        artifact_root,
        ["a1", "a2"],
    ) == {"a1": "validated", "a2": "rejected"}
    with pytest.raises(RuntimeError, match="without final status"):
        ai_local_smoke._assert_final_artifact_statuses(artifact_root, ["a3"])


def test_export_pending_takes_are_blocked():
    with pytest.raises(RuntimeError, match="Export includes pending takes"):
        ai_local_smoke._assert_export_has_no_pending_takes(
            {
                "takes": [
                    {"take_id": "take_base", "status": "accepted"},
                    {"take_id": "take_pending", "status": "pending"},
                ]
            }
        )

    ai_local_smoke._assert_export_has_no_pending_takes(
        {
            "takes": [
                {"take_id": "take_base", "status": "accepted"},
                {"take_id": "take_rejected", "status": "rejected"},
            ]
        }
    )


def test_find_artifact_root_prefers_manifest_with_all_artifacts(tmp_path):
    api_storage = tmp_path / "api"
    api_artifacts = api_storage / "model_artifacts"
    fallback_artifacts = tmp_path / "fallback"
    api_artifacts.mkdir(parents=True)
    fallback_artifacts.mkdir()
    (api_artifacts / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": [{"artifact_id": "a1", "status": "validated"}]}),
        encoding="utf-8",
    )
    (fallback_artifacts / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"artifact_id": "a1", "status": "validated"},
                    {"artifact_id": "a2", "status": "rejected"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert ai_local_smoke._find_artifact_root(
        str(fallback_artifacts),
        api_storage,
        ["a1", "a2"],
    ) == fallback_artifacts
