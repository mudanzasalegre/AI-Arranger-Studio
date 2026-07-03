from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_tools import (
    DatasetFileProfile,
    DatasetManifest,
    DatasetManifestEntry,
    DatasetProfileReport,
    profile_dataset_file,
)

from training.tokenizers.miditok_real import MidiTokSource

_PROFILE_ROLE_TO_MIDITOK_ROLE = {
    "melody": "melody",
    "solo": "melody",
    "bass": "walking_bass",
    "drums": "drums",
    "comping": "piano_comping",
    "horns": "horn_responses",
}


def miditok_sources_from_dataset_manifest(
    manifest: DatasetManifest | str | Path,
    *,
    source_root: str | Path | None = None,
    profile_report: DatasetProfileReport | str | Path | None = None,
    min_role_confidence: float = 0.5,
) -> list[MidiTokSource]:
    manifest_model = _load_manifest(manifest)
    root = _source_root(manifest, source_root)
    profiles_by_path = _profiles_by_path(profile_report)

    sources: list[MidiTokSource] = []
    for entry_index, entry in enumerate(manifest_model.entries, start=1):
        source_path = _resolve_source_path(root, entry.path)
        profile = profiles_by_path.get(str(source_path.resolve()))
        if profile is None:
            profile = profile_dataset_file(
                source_path,
                file_id=_source_file_id(entry_index, entry, source_path),
                metadata=entry,
                file_hash=entry.hash or None,
            )
        sources.append(
            MidiTokSource(
                path=str(source_path),
                source_file_id=profile.file_id,
                style=entry.style,
                license=entry.license,
                chord_context=_chord_context(entry),
                source_dataset=entry.source,
                track_roles=_track_roles(profile, min_role_confidence=min_role_confidence),
                training_allowed=entry.usable_for_training,
                commercial_training=entry.commercial_training,
                tags=entry.tags,
                metadata={
                    "manifest_entry": entry.model_dump(mode="json"),
                    "dataset_profile": profile.model_dump(mode="json"),
                    "quality": entry.quality,
                    "roles": profile.role_coverage,
                    "file_features": profile.file_features,
                    "pattern_sensitivity": profile.pattern_sensitivity,
                },
            )
        )
    return sources


def _load_manifest(manifest: DatasetManifest | str | Path) -> DatasetManifest:
    if isinstance(manifest, DatasetManifest):
        return manifest
    return DatasetManifest.load_json(manifest)


def _source_root(manifest: DatasetManifest | str | Path, source_root: str | Path | None) -> Path:
    if source_root is not None:
        return Path(source_root)
    if isinstance(manifest, str | Path):
        return Path(manifest).parent
    return Path.cwd()


def _profiles_by_path(
    profile_report: DatasetProfileReport | str | Path | None,
) -> dict[str, DatasetFileProfile]:
    if profile_report is None:
        return {}
    report = (
        DatasetProfileReport.load_json(profile_report)
        if isinstance(profile_report, str | Path)
        else profile_report
    )
    output: dict[str, DatasetFileProfile] = {}
    for profile in report.files:
        for path in (profile.original_path, profile.normalized_path):
            if path:
                output[str(Path(path).resolve())] = profile
    return output


def _resolve_source_path(source_root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return source_root / path


def _source_file_id(
    entry_index: int,
    entry: DatasetManifestEntry,
    source_path: Path,
) -> str:
    file_hash = entry.hash or profile_dataset_file(source_path, metadata=entry).hash
    return f"file_{entry_index:04d}_{file_hash[:12]}"


def _track_roles(
    profile: DatasetFileProfile,
    *,
    min_role_confidence: float,
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for track in profile.track_profiles:
        if track.classification.confidence < min_role_confidence:
            continue
        role = _PROFILE_ROLE_TO_MIDITOK_ROLE.get(track.classification.role)
        if role is None:
            continue
        roles[str(track.track_index)] = role
        if track.name:
            roles[track.name] = role
            roles[track.name.lower()] = role
    return roles


def _chord_context(entry: DatasetManifestEntry) -> list[str]:
    context = _metadata_list(entry, "chord_context")
    if context:
        return context
    return [tag for tag in entry.tags if tag.startswith("chord:")]


def _metadata_list(entry: DatasetManifestEntry, key: str) -> list[str]:
    value: Any = getattr(entry, key, None)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
