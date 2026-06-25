from __future__ import annotations

from pathlib import Path

from arranger_core import GenerationSpec, compile_prompt, export_project, generate_arrangement
from music21 import converter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/obj7_arrangement_demo"
SEXTET_PROMPT = (
    "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, "
    "trompeta, trombon, piano, contrabajo y bateria"
)


def main() -> None:
    cases = [
        (
            "jazz_trio",
            GenerationSpec(ensemble="jazz_trio", form="minor_blues_12", seed=3701),
        ),
        (
            "jazz_quartet",
            GenerationSpec(ensemble="jazz_quartet_alto", form="aaba_32", seed=3702),
        ),
        (
            "jazz_sextet",
            compile_prompt(SEXTET_PROMPT, seed=3703),
        ),
    ]

    for case_id, spec in cases:
        project = generate_arrangement(spec, project_id=f"obj7-{case_id}")
        manifest = export_project(project, OUTPUT_ROOT / case_id, include_pdf=False)
        musicxml_path = _manifest_path(manifest["files"], "musicxml_full")
        converter.parse(musicxml_path)
        midi_track_count = sum(
            1 for file_record in manifest["files"] if file_record.get("kind") == "midi_track"
        )
        if midi_track_count != len(project.tracks):
            raise RuntimeError(
                f"{case_id}: expected {len(project.tracks)} MIDI tracks, got {midi_track_count}"
            )
        if project.validate_bar_durations():
            raise RuntimeError(f"{case_id}: bar duration validation failed")
        print(
            f"Generated {case_id}: {project.bar_count} bars, "
            f"{len(project.tracks)} tracks, {midi_track_count} MIDI track files"
        )


def _manifest_path(files: list[dict[str, object]], kind: str) -> Path:
    for file_record in files:
        if file_record.get("kind") == kind:
            return Path(str(file_record["path"]))
    raise RuntimeError(f"Missing export file kind: {kind}")


if __name__ == "__main__":
    main()
