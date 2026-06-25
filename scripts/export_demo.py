from __future__ import annotations

from pathlib import Path

from arranger_core import ArrangementProject, export_project

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = ROOT / "examples/projects/arrangement_project.example.json"
OUTPUT_DIR = ROOT / "outputs/demo"


def main() -> None:
    project = ArrangementProject.load_json(PROJECT_PATH)
    manifest = export_project(project, OUTPUT_DIR, include_pdf=True)
    print(f"Exported {manifest['project_id']} to {manifest['output_dir']}")


if __name__ == "__main__":
    main()
