from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app import main
from arranger_core import GenerationSpec, generate_arrangement
from fastapi.testclient import TestClient


def test_professional_generate_endpoint_writes_project_storage(tmp_path, monkeypatch):
    storage = tmp_path / "api-storage"
    captured = {}
    monkeypatch.setenv("AI_ARRANGER_API_STORAGE", str(storage))
    monkeypatch.setenv("AI_MODELS_CONFIG", "configs/ai_models.pro.yaml")

    class FakeProfessionalOrchestrator:
        def __init__(self, *, repo_root: Path) -> None:
            captured["repo_root"] = repo_root

        def generate(self, options):
            captured["options"] = options
            project = generate_arrangement(
                GenerationSpec(
                    prompt=options.prompt,
                    ensemble="jazz_quartet_alto",
                    form="minor_blues_12",
                    seed=options.seed,
                ),
                project_id=options.run_id,
            )
            output_dir = Path(options.output_root) / options.run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            project.save_json(output_dir / "arrangement_project.json")
            if project.generation_spec is not None:
                (output_dir / "generation_spec.json").write_text(
                    project.generation_spec.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
            export_manifest = {"files": []}
            validation = {"status": "pass", "errors": [], "warnings": [], "metrics": {}}
            return SimpleNamespace(
                status="ok",
                project_id=options.run_id,
                export_manifest=export_manifest,
                validation=validation,
                quality={"status": "pass", "rating": "B", "score": 0.8},
                model_trace={"model_artifacts": []},
                takes_manifest={"takes": []},
                steps=[{"step": "llm_planner", "status": "ok"}],
                fallbacks=[],
            )

    monkeypatch.setattr(main, "ProfessionalGenerationOrchestrator", FakeProfessionalOrchestrator)
    client = TestClient(main.app)

    response = client.post(
        "/v1/projects/generate-professional",
        json={
            "prompt": "cool jazz en Fa menor, 133 bpm",
            "seed": 4243,
            "project_id": "api-pro-run",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "generated_professional"
    assert payload["project_id"] == "api-pro-run"
    assert payload["quality"]["rating"] == "B"
    assert captured["options"].output_root == str(storage / "projects")
    assert captured["options"].ai_config_path == "configs/ai_models.pro.yaml"
    assert (storage / "projects" / "api-pro-run" / "arrangement_project.json").exists()
