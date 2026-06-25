from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_golden_generate_single_preset_writes_metrics(tmp_path):
    output_dir = tmp_path / "golden"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/golden_generate.py"),
            "--output-dir",
            str(output_dir),
            "--preset-id",
            "jazz_hard_bop_minor_blues_sextet",
            "--no-clean",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metrics_path = output_dir / "jazz_hard_bop_minor_blues_sextet/music_metrics.json"
    summary_path = output_dir / "golden_summary.json"

    assert metrics_path.exists()
    assert summary_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["preset_count"] == 1
    assert metrics["project"]["tracks"] == 6
    assert metrics["validation"]["errors"] == 0
    assert metrics["tracks"]
    assert all("notes_per_bar" in track for track in metrics["tracks"])
    assert "estimated_score_1_to_5" in metrics["baseline_rating"]


def test_makefile_exposes_golden_baseline_target():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "golden-baseline:" in makefile
