from __future__ import annotations

from pathlib import Path

from arranger_core import compile_prompt, export_project, generate_harmony_project

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/obj5_harmony_demo"
PROMPT = (
    "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, "
    "trompeta, trombon, piano, contrabajo y bateria"
)


def main() -> None:
    spec = compile_prompt(PROMPT, seed=1234)
    project = generate_harmony_project(spec, project_id="obj5-harmony-demo")
    manifest = export_project(project, OUTPUT_DIR, include_pdf=False)

    musicxml_path = _manifest_path(manifest["files"], "musicxml_full")
    musicxml_text = musicxml_path.read_text(encoding="utf-8")

    if project.bar_count != 12:
        raise RuntimeError(f"Expected 12 bars, got {project.bar_count}")
    if "<harmony" not in musicxml_text:
        raise RuntimeError("MusicXML export does not contain harmony elements")
    if "minor-seventh" not in musicxml_text:
        raise RuntimeError("MusicXML export does not contain minor seventh harmony")

    print(
        "Generated obj5 harmony smoke: "
        f"{project.bar_count} bars, {len(project.chord_grid)} chord symbols, "
        f"{musicxml_path}"
    )


def _manifest_path(files: list[dict[str, object]], kind: str) -> Path:
    for file_record in files:
        if file_record.get("kind") == kind:
            return Path(str(file_record["path"]))
    raise RuntimeError(f"Missing export file kind: {kind}")


if __name__ == "__main__":
    main()
