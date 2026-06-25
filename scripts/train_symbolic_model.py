from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from arranger_core import (
    AIWalkingBassGenerator,
    GenerationSpec,
    RuleBasedArranger,
    export_project,
    validate_project,
)
from dataset_tools import (
    DatasetManifest,
    DatasetManifestEntry,
    ExtractedPattern,
    NormalizedFile,
    PatternIndex,
    build_training_examples,
    extract_patterns,
    sha256_file,
)
from midi_models import SymbolicPatternModelBackend, train_symbolic_pattern_model

ROOT = Path(__file__).resolve().parents[1]
MIDI_DATABASES_ROOT = ROOT / "midi_databases"
DEFAULT_SOURCE_DIRS = [
    MIDI_DATABASES_ROOT / "JAZZVAR_DATASET",
    MIDI_DATABASES_ROOT / "RELEASE2.0_mid_unquant",
]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/models/jazzvar_release2_symbolic"
SUPPORTED_EXTENSIONS = {".mid", ".midi", ".musicxml", ".xml"}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    source_dirs = [Path(path).resolve() for path in args.source_dir]
    _validate_source_dirs(source_dirs)
    if args.clean:
        _clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = train_from_directories(
        source_dirs=source_dirs,
        output_dir=output_dir,
        license_name=args.license,
        style=args.style,
        quality=args.quality,
        max_files=args.max_files,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


def train_from_directories(
    *,
    source_dirs: list[Path],
    output_dir: Path,
    license_name: str,
    style: str,
    quality: int,
    max_files: int | None,
    progress_every: int,
) -> dict[str, Any]:
    files = _selected_files(source_dirs, max_files=max_files)
    manifest_entries: list[DatasetManifestEntry] = []
    failed_files: list[dict[str, str]] = []
    pattern_index = PatternIndex()
    counters: Counter[str] = Counter()

    print(
        f"Training symbolic model from {len(files)} files in "
        f"{', '.join(_display_path(path) for path in source_dirs)}",
        flush=True,
    )
    for file_number, path in enumerate(files, start=1):
        dataset_name = _dataset_name(path, source_dirs)
        try:
            file_hash = sha256_file(path)
            manifest_entry = DatasetManifestEntry(
                path=_relative_to_root(path),
                source=dataset_name,
                license=license_name,
                copyright_notes=(
                    "User-selected local dataset for symbolic model training. "
                    "Only JAZZVAR_DATASET and RELEASE2.0_mid_unquant were included."
                ),
                usable_for_training=True,
                usable_for_pattern_extraction=True,
                style=style,
                quality=quality,
                tags=["jazz", dataset_name, "model_training"],
                hash=file_hash,
            )
            manifest_entries.append(manifest_entry)
            normalized = NormalizedFile(
                file_id=f"{_slug(dataset_name)}_{file_number:06d}_{file_hash[:12]}",
                original_path=str(path),
                normalized_path=str(path),
                source=dataset_name,
                license=license_name,
                hash=file_hash,
                style=style,
                quality=quality,
                tags=manifest_entry.tags,
                usable_for_training=True,
                usable_for_pattern_extraction=True,
                stats={"format": path.suffix.lower().removeprefix(".")},
            )
            extracted = extract_patterns(path, normalized)
            for pattern in extracted:
                _add_or_increment(pattern_index, pattern)
            counters["processed_files"] += 1
            counters["extracted_patterns_raw"] += len(extracted)
        except Exception as exc:
            failed_files.append({"path": str(path), "error": repr(exc)})
            counters["failed_files"] += 1

        if progress_every > 0 and file_number % progress_every == 0:
            print(
                f"Processed {file_number}/{len(files)} files; "
                f"unique_patterns={len(pattern_index.patterns)}; "
                f"failed={len(failed_files)}",
                flush=True,
            )

    manifest = DatasetManifest(entries=manifest_entries)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest.save_json(manifest_path)

    pattern_index_path = output_dir / "pattern_index.json"
    pattern_index.save_json(pattern_index_path)

    failed_path = output_dir / "failed_files.json"
    failed_path.write_text(json.dumps(failed_files, indent=2) + "\n", encoding="utf-8")

    training_summary = build_training_examples(
        pattern_index,
        output_dir / "training",
        seed=240,
        min_quality=3,
    )
    model_path = output_dir / "model/symbolic_pattern_model.json"
    model = train_symbolic_pattern_model(
        pattern_index,
        model_path,
        source_roots=[_relative_to_root(path) for path in source_dirs],
        training_summary=training_summary,
        model_name="jazzvar-release2-symbolic",
        metadata={
            "selected_only": [
                "midi_databases/JAZZVAR_DATASET",
                "midi_databases/RELEASE2.0_mid_unquant",
            ],
            "license": license_name,
            "style": style,
            "quality": quality,
        },
    )
    smoke_summary = _smoke_model(model_path, output_dir / "smoke_export")

    summary = {
        "status": "pass",
        "source_dirs": [_relative_to_root(path) for path in source_dirs],
        "discovered_files": len(files),
        "processed_files": counters["processed_files"],
        "failed_files": counters["failed_files"],
        "failed_files_path": str(failed_path),
        "raw_extracted_patterns": counters["extracted_patterns_raw"],
        "unique_patterns": len(pattern_index.patterns),
        "category_counts": dict(
            sorted(Counter(pattern.category for pattern in pattern_index.patterns).items())
        ),
        "training_examples": training_summary.total_examples,
        "split_counts": training_summary.split_counts,
        "manifest_path": str(manifest_path),
        "pattern_index_path": str(pattern_index_path),
        "training_examples_path": training_summary.training_examples_path,
        "feature_store_path": training_summary.feature_store_path,
        "model_path": str(model_path),
        "stored_pattern_count": model["stored_pattern_count"],
        "smoke": smoke_summary,
    }
    (output_dir / "training_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _smoke_model(model_path: Path, output_dir: Path) -> dict[str, Any]:
    backend = SymbolicPatternModelBackend.load(model_path)
    arranger = RuleBasedArranger(bass_generator=AIWalkingBassGenerator(backend))
    project = arranger.generate(
        GenerationSpec(
            prompt="trained symbolic model smoke",
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=240,
        ),
        project_id="trained-symbolic-model-smoke",
    )
    report = validate_project(project)
    if report["errors"]:
        raise RuntimeError(f"Trained symbolic model smoke failed: {report['errors']}")
    manifest = export_project(project, output_dir, include_pdf=False)
    bass = next(track for track in project.tracks if track.id == "double_bass")
    return {
        "status": "pass",
        "arranger": project.metadata["arranger"],
        "bass_generator": bass.metadata["generator"],
        "bass_backend": bass.metadata["model_backend"],
        "validation_status": report["status"],
        "exported_files": len(manifest["files"]),
        "output_dir": str(output_dir),
    }


def _selected_files(source_dirs: list[Path], *, max_files: int | None) -> list[Path]:
    selected: list[Path] = []
    for source_dir in source_dirs:
        selected.extend(
            path
            for path in sorted(source_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if max_files is not None:
        return selected[:max_files]
    return selected


def _add_or_increment(index: PatternIndex, pattern: ExtractedPattern) -> None:
    for existing in index.patterns:
        if existing.fingerprint == pattern.fingerprint:
            existing.weight += pattern.weight
            existing.tags = sorted({*existing.tags, *pattern.tags})
            return
    index.add(pattern)


def _validate_source_dirs(source_dirs: list[Path]) -> None:
    expected = {path.resolve() for path in DEFAULT_SOURCE_DIRS}
    actual = {path.resolve() for path in source_dirs}
    if actual != expected:
        raise ValueError(
            "This training run is restricted to exactly "
            "midi_databases/JAZZVAR_DATASET and "
            "midi_databases/RELEASE2.0_mid_unquant"
        )
    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")


def _clean_output_dir(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_outputs = (ROOT / "outputs").resolve()
    if resolved_path == resolved_outputs or not resolved_path.is_relative_to(resolved_outputs):
        raise RuntimeError(f"Refusing to clean path outside outputs/: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _dataset_name(path: Path, source_dirs: list[Path]) -> str:
    for source_dir in source_dirs:
        if path.resolve().is_relative_to(source_dir.resolve()):
            return source_dir.name
    return "unknown"


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        return resolved.relative_to(ROOT.resolve()).as_posix()
    return str(resolved)


def _display_path(path: Path) -> str:
    return _relative_to_root(path)


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the symbolic pattern model using only JAZZVAR_DATASET and "
            "RELEASE2.0_mid_unquant."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        default=DEFAULT_SOURCE_DIRS,
        help="Restricted source directory. Defaults are the two approved MIDI folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--license", default="user_supplied_training_only")
    parser.add_argument("--style", default="hard_bop")
    parser.add_argument("--quality", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the output directory before training. Cleaning is limited to outputs/.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
