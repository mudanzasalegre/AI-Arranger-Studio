from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_path in (
    ROOT / "packages" / "arranger_core",
    ROOT / "packages" / "dataset_tools",
    ROOT / "packages" / "midi_models",
    ROOT / "packages" / "model_backends",
):
    sys.path.insert(0, str(package_path))

from arranger_core import (  # noqa: E402
    AIDrumsGenerator,
    AIHornResponseGenerator,
    AIMelodyGenerator,
    AIPianoCompingGenerator,
    AIWalkingBassGenerator,
    DeterministicRoleModelBackend,
    GenerationSpec,
    RuleBasedArranger,
    export_project,
    validate_project,
)

OUTPUT_ROOT = ROOT / "outputs" / "pr19_custom_role_model_smoke"


def main() -> None:
    outputs_root = (ROOT / "outputs").resolve()
    smoke_root = OUTPUT_ROOT.resolve()
    if outputs_root not in smoke_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {smoke_root}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    backend = DeterministicRoleModelBackend()
    arranger = RuleBasedArranger(
        drums_generator=AIDrumsGenerator(backend, model_mode="external_model"),
        bass_generator=AIWalkingBassGenerator(backend, model_mode="custom_model"),
        piano_generator=AIPianoCompingGenerator(backend, model_mode="custom_model"),
        melody_generator=AIMelodyGenerator(backend, model_mode="external_model"),
        horn_response_generator=AIHornResponseGenerator(backend, model_mode="custom_model"),
    )
    project = arranger.generate(
        GenerationSpec(
            ensemble="jazz_sextet",
            form="minor_blues_12",
            style="hard_bop",
            seed=1901,
            constraints={"humanize": False},
        ),
        project_id="pr19-custom-role-model-smoke",
    )
    report = validate_project(project)
    if report["errors"]:
        raise RuntimeError(f"Custom role model smoke failed validation: {report['errors']}")
    export_manifest = export_project(project, smoke_root / "export", include_pdf=False)
    summary = {
        "status": "pass",
        "track_count": len(project.tracks),
        "track_modes": {
            track.id: track.metadata.get("role_model_mode")
            for track in project.tracks
        },
        "model_backend": backend.name,
        "validation_status": report["status"],
        "exported_files": len(export_manifest["files"]),
        "export_dir": str(smoke_root / "export"),
    }
    (smoke_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
