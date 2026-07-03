from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = [
    "models/hf_cache/hub",
    "models/hf_cache/assets",
    "models/external_repos",
    "models/checkpoints/text2midi",
    "models/checkpoints/custom/melody",
    "models/checkpoints/custom/bass",
    "models/checkpoints/custom/piano_comping",
    "models/checkpoints/custom/horns",
    "models/checkpoints/custom/drums",
    "models/manifests",
    "outputs/model_artifacts/raw",
    "outputs/model_artifacts/imported",
    "outputs/model_artifacts/rejected",
    "outputs/model_artifacts/validated",
    "outputs/model_smoke",
    "outputs/professional_benchmark",
]


def main() -> None:
    created: list[str] = []
    for rel in DIRS:
        path = ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(rel)

    report = {
        "status": "ok",
        "root": str(ROOT),
        "created_or_verified": created,
        "notes": [
            "models/ and outputs/ should remain ignored by git.",
            "Copy configs/model_registry.example.yaml to configs/model_registry.yaml when PR-20 starts.",
        ],
    }
    report_path = ROOT / "models/manifests/local_model_dirs_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
