from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
required = [
    "docs/00_MASTER_RECIPE.md",
    "docs/04_OBJECTIVES.md",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "docker-compose.yml",
    "apps/api/app/main.py",
    "apps/web/package.json",
    "apps/web/app/page.tsx",
    "packages/arranger_core/arranger_core/__init__.py",
    "packages/dataset_tools/dataset_tools/__init__.py",
    "packages/midi_models/midi_models/__init__.py",
    "packages/model_backends/model_backends/__init__.py",
    "packages/training/training/__init__.py",
    "configs/instruments.yaml",
    "configs/ai_models.yaml",
    "configs/chord_dictionary.yaml",
    "configs/jazz_progressions.yaml",
]

for rel in required:
    path = ROOT / rel
    assert path.exists(), f"Missing {rel}"

for path in sorted((ROOT / "configs").rglob("*.yaml")):
    with path.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)

print("Bootstrap check OK")
