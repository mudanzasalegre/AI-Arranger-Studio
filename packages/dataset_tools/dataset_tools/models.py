from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_EXTENSIONS = {".mid", ".midi", ".musicxml", ".xml"}
PatternCategory = Literal[
    "progressions",
    "walking_bass_cells",
    "piano_voicings",
    "drum_grooves",
    "melodic_motifs",
    "horn_responses",
]
DatasetSplit = Literal["train", "val", "test"]
LicenseConfidence = Literal["high", "medium", "low"]
CommercialTrainingUse = Literal["allowed", "forbidden", "review_required"]
DatasetRole = Literal[
    "melody",
    "bass",
    "drums",
    "comping",
    "horns",
    "pad",
    "solo",
    "harmony",
    "unknown",
]


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetManifestEntry(DatasetModel):
    path: str
    source: str
    license: str
    license_confidence: LicenseConfidence = "low"
    commercial_training: CommercialTrainingUse = "review_required"
    local_learning_only: bool = False
    copyright_notes: str = ""
    usable_for_training: bool = False
    usable_for_pattern_extraction: bool = False
    style: str = "unknown"
    quality: int = Field(default=3, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    roles: list[DatasetRole] = Field(default_factory=list)
    contains_melody: bool = False
    contains_chords: bool = False
    contains_arrangement: bool = False
    imported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hash: str = ""

    @field_validator("path")
    @classmethod
    def path_extension_is_supported(cls, value: str) -> str:
        suffix = Path(value).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset file extension: {suffix}")
        return value


class DatasetManifest(DatasetModel):
    schema_version: str = "0.1.0"
    entries: list[DatasetManifestEntry] = Field(default_factory=list)

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> DatasetManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class NormalizedFile(DatasetModel):
    file_id: str
    original_path: str
    normalized_path: str
    source: str
    license: str
    license_confidence: LicenseConfidence = "low"
    commercial_training: CommercialTrainingUse = "review_required"
    local_learning_only: bool = False
    hash: str
    style: str
    quality: int
    tags: list[str] = Field(default_factory=list)
    usable_for_training: bool
    usable_for_pattern_extraction: bool
    duplicate_of: str | None = None
    role_hints: list[str] = Field(default_factory=list)
    roles: list[DatasetRole] = Field(default_factory=list)
    contains_melody: bool = False
    contains_chords: bool = False
    contains_arrangement: bool = False
    stats: dict[str, Any] = Field(default_factory=dict)


class ExtractedPattern(DatasetModel):
    id: str
    category: PatternCategory
    role: str
    style: str
    quality: int
    source_file_id: str
    source_path: str
    source_hash: str = ""
    license: str
    usable_for_training: bool
    usable_for_pattern_extraction: bool
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0
    context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str


class PatternIndex(DatasetModel):
    schema_version: str = "0.1.0"
    patterns: list[ExtractedPattern] = Field(default_factory=list)

    def add(self, pattern: ExtractedPattern) -> None:
        if pattern.fingerprint in {existing.fingerprint for existing in self.patterns}:
            return
        self.patterns.append(pattern)

    def search(
        self,
        *,
        category: PatternCategory | str | None = None,
        role: str | None = None,
        style: str | None = None,
        min_quality: int = 1,
        tags: list[str] | None = None,
        usable_for_training: bool | None = None,
        usable_for_pattern_extraction: bool | None = None,
    ) -> list[ExtractedPattern]:
        required_tags = set(tags or [])
        matches: list[ExtractedPattern] = []
        for pattern in self.patterns:
            if category and pattern.category != category:
                continue
            if role and pattern.role != role:
                continue
            if style and pattern.style != style:
                continue
            if pattern.quality < min_quality:
                continue
            training_filter_mismatch = (
                usable_for_training is not None
                and pattern.usable_for_training != usable_for_training
            )
            if training_filter_mismatch:
                continue
            extraction_filter_mismatch = (
                usable_for_pattern_extraction is not None
                and pattern.usable_for_pattern_extraction != usable_for_pattern_extraction
            )
            if extraction_filter_mismatch:
                continue
            if required_tags and not required_tags.issubset(set(pattern.tags)):
                continue
            matches.append(pattern)
        return sorted(matches, key=lambda item: (-item.quality, -item.weight, item.id))

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> PatternIndex:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ImportSummary(DatasetModel):
    imported_files: int = 0
    duplicate_files: int = 0
    skipped_for_license: int = 0
    skipped_for_quality: int = 0
    extracted_patterns: int = 0
    pattern_counts: dict[str, int] = Field(default_factory=dict)
    profiled_files: int = 0
    role_counts: dict[str, int] = Field(default_factory=dict)
    manifest_path: str
    normalized_files_path: str
    pattern_index_path: str
    profile_report_path: str = ""
    role_manifest_path: str = ""
    summary_path: str


class RoleClassification(DatasetModel):
    role: DatasetRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    alternatives: dict[DatasetRole, float] = Field(default_factory=dict)


class TrackProfile(DatasetModel):
    track_index: int
    name: str
    source_kind: Literal["midi_track", "musicxml_part"]
    channels: list[int] = Field(default_factory=list)
    programs: list[int] = Field(default_factory=list)
    instrument_guess: str = "unknown"
    classification: RoleClassification
    features: dict[str, Any] = Field(default_factory=dict)
    no_memorization_fingerprint: str


class DatasetFileProfile(DatasetModel):
    schema_version: str = "0.1.0"
    file_id: str
    original_path: str
    normalized_path: str = ""
    source: str
    license: str
    license_confidence: LicenseConfidence = "low"
    commercial_training: CommercialTrainingUse = "review_required"
    local_learning_only: bool = False
    style: str = "unknown"
    quality: int = Field(default=3, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    usable_for_training: bool = False
    usable_for_pattern_extraction: bool = False
    duplicate_of: str | None = None
    hash: str
    format: str
    track_profiles: list[TrackProfile] = Field(default_factory=list)
    file_features: dict[str, Any] = Field(default_factory=dict)
    role_coverage: list[DatasetRole] = Field(default_factory=list)
    contains_melody: bool = False
    contains_chords: bool = False
    contains_arrangement: bool = False
    pattern_sensitivity: dict[str, Any] = Field(default_factory=dict)
    no_memorization_fingerprint: str


class DatasetProfileReport(DatasetModel):
    schema_version: str = "0.1.0"
    files: list[DatasetFileProfile] = Field(default_factory=list)
    role_counts: dict[str, int] = Field(default_factory=dict)
    file_count: int = 0
    track_count: int = 0
    note_count: int = 0

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> DatasetProfileReport:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class TrainingExample(DatasetModel):
    id: str
    split: DatasetSplit = "train"
    style: str
    role: str
    instrument: str = "unknown"
    key: str = "unknown"
    meter: str = "4/4"
    tempo: int | None = None
    chord_context: list[str] = Field(default_factory=list)
    previous_tokens: list[str] = Field(default_factory=list)
    target_tokens: list[str] = Field(default_factory=list)
    source_file_id: str
    source_path: str
    source_hash: str
    license: str
    usable_for_training: bool
    pattern_id: str
    pattern_fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureRecord(DatasetModel):
    id: str
    example_id: str
    role: str
    style: str
    source_file_id: str
    license: str
    usable_for_training: bool
    values: dict[str, Any] = Field(default_factory=dict)


class FeatureStore(DatasetModel):
    schema_version: str = "0.1.0"
    records: list[FeatureRecord] = Field(default_factory=list)

    def add(self, record: FeatureRecord) -> None:
        if record.id in {existing.id for existing in self.records}:
            return
        self.records.append(record)

    def search(
        self,
        *,
        role: str | None = None,
        style: str | None = None,
        usable_for_training: bool | None = None,
    ) -> list[FeatureRecord]:
        matches = []
        for record in self.records:
            if role and record.role != role:
                continue
            if style and record.style != style:
                continue
            training_filter_mismatch = (
                usable_for_training is not None
                and record.usable_for_training != usable_for_training
            )
            if training_filter_mismatch:
                continue
            matches.append(record)
        return sorted(matches, key=lambda item: item.id)

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> FeatureStore:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class DatasetSplitSummary(DatasetModel):
    total_examples: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    skipped_not_training_allowed: int = 0
    skipped_blocked_license: int = 0
    training_examples_path: str
    feature_store_path: str
    split_manifest_path: str


class MemorizationReport(DatasetModel):
    status: Literal["pass", "fail"]
    threshold: float
    candidate_count: int
    reference_count: int
    flagged_count: int
    max_similarity: float
    matches: list[dict[str, Any]] = Field(default_factory=list)
