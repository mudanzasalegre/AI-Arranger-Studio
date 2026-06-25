from __future__ import annotations

from pathlib import Path

from arranger_core import (
    GenerationSpec,
    compile_prompt,
    export_project,
    find_musescore_cli,
    generate_lead_sheet_project,
)
from music21 import converter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/obj6_lead_sheet_demo"
BLUES_PROMPT = (
    "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, "
    "trompeta, trombon, piano, contrabajo y bateria"
)


def main() -> None:
    include_pdf = find_musescore_cli() is not None
    cases = [
        (
            "minor_blues_12",
            compile_prompt(BLUES_PROMPT, seed=2606).model_copy(
                update={"constraints": {"melody_range": {"low": "C4", "high": "Bb5"}}}
            ),
        ),
        (
            "aaba_32",
            GenerationSpec(
                style="ballad",
                key="F major",
                tempo=88,
                form="aaba_32",
                ensemble="jazz_quartet_alto",
                complexity=0.45,
                seed=2607,
                constraints={"melody_range": "D4-A5"},
            ),
        ),
    ]

    for case_id, spec in cases:
        project = generate_lead_sheet_project(spec, project_id=f"obj6-{case_id}")
        manifest = export_project(project, OUTPUT_ROOT / case_id, include_pdf=include_pdf)
        musicxml_path = _manifest_path(manifest["files"], "musicxml_full")
        converter.parse(musicxml_path)
        musicxml_text = musicxml_path.read_text(encoding="utf-8")
        if "<harmony" not in musicxml_text:
            raise RuntimeError(f"{case_id}: MusicXML export has no harmony elements")
        if "<articulations>" not in musicxml_text:
            raise RuntimeError(f"{case_id}: MusicXML export has no articulations")
        pdf_created_without_record = (
            manifest["pdf_status"] == "created"
            and not _manifest_has_kind(manifest["files"], "pdf_full")
        )
        if pdf_created_without_record:
            raise RuntimeError(f"{case_id}: PDF status says created but no full PDF was recorded")
        print(
            f"Generated {case_id}: {project.bar_count} bars, "
            f"{len(project.chord_grid)} chord symbols, pdf_status={manifest['pdf_status']}"
        )


def _manifest_path(files: list[dict[str, object]], kind: str) -> Path:
    for file_record in files:
        if file_record.get("kind") == kind:
            return Path(str(file_record["path"]))
    raise RuntimeError(f"Missing export file kind: {kind}")


def _manifest_has_kind(files: list[dict[str, object]], kind: str) -> bool:
    return any(file_record.get("kind") == kind for file_record in files)


if __name__ == "__main__":
    main()
