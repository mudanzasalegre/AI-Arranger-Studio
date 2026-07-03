from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "models_pro" / "professional_benchmark_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("professional_benchmark_gate", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_summary_passes_pr36_acceptance_criteria(tmp_path):
    module = _load_module()
    cases = [
        {
            "id": f"demo_{index}",
            "status": "ok",
            "rating": "B" if index else "A",
            "score": 0.8,
            "blocking_errors": [],
            "commercial_blockers": [],
        }
        for index in range(5)
    ]

    summary = module._aggregate_summary(
        cases,
        output_root=tmp_path,
        min_demos=5,
        min_average_rating="B",
        export_mode="private",
    )

    assert summary["status"] == "ok"
    assert summary["generated_demos"] == 5
    assert summary["blocking_error_count"] == 0
    assert summary["average_rating"] == "B"


def test_aggregate_summary_fails_on_commercial_blocker(tmp_path):
    module = _load_module()
    cases = [
        {
            "id": f"demo_{index}",
            "status": "ok",
            "rating": "A",
            "score": 0.9,
            "blocking_errors": [],
            "commercial_blockers": ["model_license_incompatible"] if index == 0 else [],
        }
        for index in range(5)
    ]

    summary = module._aggregate_summary(
        cases,
        output_root=tmp_path,
        min_demos=5,
        min_average_rating="B",
        export_mode="commercial",
    )

    assert summary["status"] == "fail"
    assert summary["commercial_blocker_count"] == 1
