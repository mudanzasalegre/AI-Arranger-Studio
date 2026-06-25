from __future__ import annotations

import json
import shutil
from pathlib import Path

from arranger_core import (
    PresetLibrary,
    export_project,
    find_musescore_cli,
    generate_arrangement,
    validate_project,
)
from music21 import converter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/obj12_preset_pack"


def main() -> None:
    outputs_root = (ROOT / "outputs").resolve()
    smoke_root = OUTPUT_ROOT.resolve()
    if outputs_root not in smoke_root.parents:
        raise RuntimeError(f"Refusing to clean path outside outputs/: {smoke_root}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)

    library = PresetLibrary.load_default()
    musescore = find_musescore_cli()
    summaries = []
    for preset in library.list_presets():
        project = generate_arrangement(preset.spec, project_id=preset.id)
        report = validate_project(project)
        if report["errors"]:
            raise RuntimeError(f"{preset.id} validation errors: {report['errors']}")

        output_dir = smoke_root / preset.id
        manifest = export_project(project, output_dir, include_pdf=True)
        converter.parse(output_dir / "full_score.musicxml")
        if musescore is not None and manifest["pdf_status"] != "created":
            raise RuntimeError(f"{preset.id} did not create PDFs despite MuseScore availability")

        summaries.append(
            {
                "preset_id": preset.id,
                "style": preset.spec.style,
                "form": preset.spec.form,
                "meter": preset.spec.meter,
                "bars": project.bar_count,
                "tracks": len(project.tracks),
                "validation_status": report["status"],
                "warnings": len(report["warnings"]),
                "exported_files": len(manifest["files"]),
                "pdf_status": manifest["pdf_status"],
                "output_dir": str(output_dir),
            }
        )

    payload = {
        "presets": summaries,
        "preset_count": len(summaries),
        "evaluation_prompt_count": len(library.evaluation_pack()),
        "musescore_cli": str(musescore) if musescore else None,
    }
    (smoke_root / "preset_smoke_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (smoke_root / "evaluation_pack.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in library.evaluation_pack()],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
