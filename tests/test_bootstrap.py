from pathlib import Path


def test_objective_zero_scaffold_exists():
    root = Path(__file__).resolve().parents[1]
    required_paths = [
        "apps/api/app/main.py",
        "apps/web/app/page.tsx",
        "apps/web/package.json",
        "configs/instruments.yaml",
        "docs/00_MASTER_RECIPE.md",
        "docs/04_OBJECTIVES.md",
        "docker-compose.yml",
        "Makefile",
        "packages/arranger_core/arranger_core/__init__.py",
        "packages/dataset_tools/dataset_tools/__init__.py",
        "packages/midi_models/midi_models/__init__.py",
        "packages/model_backends/model_backends/__init__.py",
        "configs/ai_models.yaml",
        "pyproject.toml",
        "requirements.txt",
    ]

    for relative_path in required_paths:
        assert (root / relative_path).exists(), f"Missing {relative_path}"


def test_makefile_exposes_objective_zero_commands():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    for target in ["setup:", "test:", "lint:", "api:", "web:"]:
        assert target in makefile
