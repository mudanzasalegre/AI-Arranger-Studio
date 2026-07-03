from __future__ import annotations

import hashlib
import importlib
import json
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mido
from pydantic import BaseModel, ConfigDict, Field

MIDITOK_TRAINING_ROLES: tuple[str, ...] = (
    "melody",
    "walking_bass",
    "piano_comping",
    "horn_responses",
    "drums",
)
MIDITOK_TRAINING_SPLITS: tuple[str, ...] = ("train", "val", "test")

BLOCKED_TRAINING_LICENSES = {
    "",
    "unknown",
    "proprietary",
    "all rights reserved",
    "all-rights-reserved",
    "private",
    "research_only",
    "research-only",
    "research only",
    "non_commercial",
    "non-commercial",
    "noncommercial",
    "cc-by-nc",
    "cc-by-nc-sa",
}
BLOCKED_COMMERCIAL_TRAINING = {
    "blocked",
    "forbidden",
    "not_allowed",
    "research_only",
    "research-only",
    "non_commercial",
    "non-commercial",
}

RoleName = Literal["melody", "walking_bass", "piano_comping", "horn_responses", "drums"]
DatasetSplit = Literal["train", "val", "test", "rejected"]

_ROLE_ALIASES = {
    "lead": "melody",
    "melody": "melody",
    "solo": "melody",
    "alto": "melody",
    "alto_sax": "melody",
    "sax": "melody",
    "bass": "walking_bass",
    "double_bass": "walking_bass",
    "upright_bass": "walking_bass",
    "walking_bass": "walking_bass",
    "comping": "piano_comping",
    "piano": "piano_comping",
    "piano_comping": "piano_comping",
    "keys": "piano_comping",
    "horn": "horn_responses",
    "horn_response": "horn_responses",
    "horn_responses": "horn_responses",
    "horns": "horn_responses",
    "trumpet": "horn_responses",
    "trombone": "horn_responses",
    "drum": "drums",
    "drums": "drums",
    "drum_kit": "drums",
}


class MidiTokUnavailableError(RuntimeError):
    """Raised when the optional MidiTok stack is not installed."""


class TrainingTokenizerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MidiTokSource(TrainingTokenizerModel):
    path: str
    source_file_id: str
    style: str
    license: str
    chord_context: list[str] = Field(default_factory=list)
    source_dataset: str = "local"
    track_roles: dict[str, str] = Field(default_factory=dict)
    training_allowed: bool = True
    commercial_training: str = "allowed"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MidiTokRoleSegment(TrainingTokenizerModel):
    id: str
    split: DatasetSplit
    role: RoleName
    style: str
    chord_context: list[str] = Field(default_factory=list)
    source_file_id: str
    source_path: str
    source_hash: str
    source_dataset: str
    license: str
    commercial_training: str
    train_eligible: bool
    token_sequences: list[list[str]] = Field(default_factory=list)
    token_id_sequences: list[list[int]] = Field(default_factory=list)
    token_count: int
    note_count_input: int
    note_count_reconstructed: int | None = None
    information_loss_ratio: float | None = None
    midi_path: str
    reconstructed_midi_path: str | None = None
    tokenizer_name: str
    tokenizer_config_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MidiTokDatasetSummary(TrainingTokenizerModel):
    schema_version: str = "0.1.0"
    generated_at: str
    tokenizer_name: str
    roles: list[str]
    source_count: int
    total_segments: int
    train_segments: int
    rejected_segments: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    output_dir: str
    tokenizer_path: str
    tokenizer_config_path: str
    tokenized_segments_path: str
    metadata_path: str
    license_report_path: str
    summary_path: str
    average_information_loss_ratio: float
    max_information_loss_ratio: float
    acceptable_information_loss: bool


class MidiTokRealTokenizer:
    """Lazy wrapper around MidiTok for role-aware local training dataset export."""

    name = "miditok_real_remi"
    version = "0.1.0"

    def __init__(
        self,
        *,
        tokenizer_family: str = "REMI",
        pitch_range: tuple[int, int] = (21, 109),
        beat_res: dict[tuple[int, int], int] | None = None,
        use_programs: bool = True,
        use_tempos: bool = True,
        use_time_signatures: bool = True,
    ) -> None:
        self.tokenizer_family = tokenizer_family
        self.pitch_range = pitch_range
        self.beat_res = beat_res or {(0, 4): 8, (4, 12): 4}
        self.use_programs = use_programs
        self.use_tempos = use_tempos
        self.use_time_signatures = use_time_signatures
        self._tokenizer: Any | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = self._build_tokenizer()
        return self._tokenizer

    def encode_midi(self, midi_path: str | Path) -> tuple[list[list[str]], list[list[int]]]:
        sequences = _sequence_list(self.tokenizer.encode(Path(midi_path), encode_ids=True))
        token_sequences: list[list[str]] = []
        token_id_sequences: list[list[int]] = []
        for sequence in sequences:
            token_sequences.append([str(token) for token in getattr(sequence, "tokens", [])])
            token_id_sequences.append([int(token_id) for token_id in getattr(sequence, "ids", [])])
        return token_sequences, token_id_sequences

    def save(self, output_dir: str | Path, *, roles: tuple[str, ...]) -> tuple[Path, Path]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        config_path = out_path / "miditok_config.json"
        config_path.write_text(
            json.dumps(self.config(roles=roles), indent=2) + "\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.tokenizer.save_params(
                out_path,
                additional_attributes={
                    "ai_arranger_tokenizer": self.name,
                    "ai_arranger_tokenizer_version": self.version,
                    "roles": list(roles),
                },
                filename="tokenizer.json",
            )
        return out_path / "tokenizer.json", config_path

    def reconstruct(
        self,
        token_id_sequences: list[list[int]],
        output_path: str | Path,
    ) -> Path:
        if not token_id_sequences:
            raise ValueError("Cannot reconstruct MIDI from an empty token sequence")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        score = self.tokenizer.decode(token_id_sequences[0])
        score.dump_midi(output)
        return output

    def config(self, *, roles: tuple[str, ...]) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "tokenizer_name": self.name,
            "tokenizer_version": self.version,
            "tokenizer_family": self.tokenizer_family,
            "roles": list(roles),
            "pitch_range": list(self.pitch_range),
            "beat_resolution": {
                f"{start}_{end}": resolution
                for (start, end), resolution in self.beat_res.items()
            },
            "use_programs": self.use_programs,
            "use_tempos": self.use_tempos,
            "use_time_signatures": self.use_time_signatures,
            "lazy_imports": ["miditok", "symusic"],
        }

    def _build_tokenizer(self) -> Any:
        miditok = _import_miditok()
        tokenizer_cls = getattr(miditok, self.tokenizer_family, None)
        if tokenizer_cls is None:
            raise ValueError(f"Unsupported MidiTok tokenizer family: {self.tokenizer_family}")
        config = miditok.TokenizerConfig(
            pitch_range=self.pitch_range,
            beat_res=self.beat_res,
            use_programs=self.use_programs,
            use_tempos=self.use_tempos,
            use_time_signatures=self.use_time_signatures,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return tokenizer_cls(config)


def export_miditok_role_dataset(
    sources: list[MidiTokSource] | list[dict[str, Any]],
    output_dir: str | Path,
    *,
    roles: tuple[str, ...] = MIDITOK_TRAINING_ROLES,
    tokenizer: MidiTokRealTokenizer | None = None,
    max_acceptable_loss_ratio: float = 0.25,
) -> MidiTokDatasetSummary:
    tokenizer = tokenizer or MidiTokRealTokenizer()
    selected_roles = tuple(role for role in roles if role in MIDITOK_TRAINING_ROLES)
    if not selected_roles:
        raise ValueError("At least one supported MidiTok training role is required")

    output_path = Path(output_dir)
    manifest_dir = output_path / "manifests"
    isolated_dir = manifest_dir / "role_midi"
    reconstructed_dir = manifest_dir / "reconstructed"
    for path in (manifest_dir, isolated_dir, reconstructed_dir):
        path.mkdir(parents=True, exist_ok=True)
    for role in selected_roles:
        role_dir = output_path / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("train.jsonl", "val.jsonl", "test.jsonl", "metadata.jsonl"):
            target = role_dir / filename
            if target.exists():
                target.unlink()

    tokenizer_path, tokenizer_config_path = tokenizer.save(manifest_dir, roles=selected_roles)
    source_models = [
        source if isinstance(source, MidiTokSource) else MidiTokSource(**source)
        for source in sources
    ]

    segments: list[MidiTokRoleSegment] = []
    rejected_sources: list[dict[str, Any]] = []
    for source in source_models:
        midi_path = _ensure_midi_path(source.path, manifest_dir / "converted")
        source_hash = _hash_file(midi_path)
        train_eligible = _train_eligible(source)
        if not train_eligible:
            rejected_sources.append(
                {
                    "source_file_id": source.source_file_id,
                    "source_path": source.path,
                    "license": source.license,
                    "commercial_training": source.commercial_training,
                    "reason": _license_rejection_reason(source),
                }
            )

        role_midis = _split_midi_by_role(
            midi_path,
            isolated_dir,
            source=source,
            roles=selected_roles,
        )
        for role, role_midi_path in role_midis.items():
            token_sequences, token_id_sequences = tokenizer.encode_midi(role_midi_path)
            reconstructed_path = (
                reconstructed_dir / f"{source.source_file_id}_{role}_reconstructed.mid"
            )
            note_count_input = _note_count(role_midi_path)
            note_count_reconstructed: int | None = None
            loss_ratio: float | None = None
            if token_id_sequences:
                tokenizer.reconstruct(token_id_sequences, reconstructed_path)
                note_count_reconstructed = _note_count(reconstructed_path)
                loss_ratio = _loss_ratio(note_count_input, note_count_reconstructed)

            segment = MidiTokRoleSegment(
                id=f"miditok_{_stable_hash(f'{source.source_file_id}:{role}')[:16]}",
                split="train" if train_eligible else "rejected",
                role=role,  # type: ignore[arg-type]
                style=source.style,
                chord_context=source.chord_context,
                source_file_id=source.source_file_id,
                source_path=str(Path(source.path)),
                source_hash=source_hash,
                source_dataset=source.source_dataset,
                license=source.license,
                commercial_training=source.commercial_training,
                train_eligible=train_eligible,
                token_sequences=token_sequences,
                token_id_sequences=token_id_sequences,
                token_count=sum(len(sequence) for sequence in token_id_sequences),
                note_count_input=note_count_input,
                note_count_reconstructed=note_count_reconstructed,
                information_loss_ratio=loss_ratio,
                midi_path=str(role_midi_path),
                reconstructed_midi_path=str(reconstructed_path)
                if reconstructed_path.exists()
                else None,
                tokenizer_name=tokenizer.name,
                tokenizer_config_path=str(tokenizer_config_path),
                metadata={
                    "tags": source.tags,
                    "source_metadata": source.metadata,
                    "track_roles": source.track_roles,
                    "role_source_tracks": _role_source_tracks(midi_path, source, role),
                    "tokenizer_family": tokenizer.tokenizer_family,
                },
            )
            segments.append(segment)
            if segment.split != "rejected":
                _append_jsonl(output_path / role / f"{segment.split}.jsonl", segment)
                _append_jsonl(output_path / role / "metadata.jsonl", _segment_metadata(segment))

    tokenized_segments_path = manifest_dir / "segments.jsonl"
    metadata_path = manifest_dir / "metadata.jsonl"
    for path, payloads in (
        (tokenized_segments_path, [segment.model_dump(mode="json") for segment in segments]),
        (metadata_path, [_segment_metadata(segment) for segment in segments]),
    ):
        with path.open("w", encoding="utf-8") as file:
            for payload in payloads:
                file.write(json.dumps(payload, sort_keys=True) + "\n")

    license_report_path = manifest_dir / "license_report.json"
    license_report_path.write_text(
        json.dumps(_license_report(source_models, segments, rejected_sources), indent=2) + "\n",
        encoding="utf-8",
    )

    losses = [
        float(segment.information_loss_ratio)
        for segment in segments
        if segment.information_loss_ratio is not None
    ]
    role_counts = Counter(segment.role for segment in segments if segment.split != "rejected")
    split_counts = Counter(segment.split for segment in segments)
    summary_path = manifest_dir / "tokenization_summary.json"
    summary = MidiTokDatasetSummary(
        generated_at=datetime.now(UTC).isoformat(),
        tokenizer_name=tokenizer.name,
        roles=list(selected_roles),
        source_count=len(source_models),
        total_segments=len(segments),
        train_segments=sum(1 for segment in segments if segment.split == "train"),
        rejected_segments=sum(1 for segment in segments if segment.split == "rejected"),
        role_counts={role: role_counts.get(role, 0) for role in selected_roles},
        split_counts={
            split: split_counts.get(split, 0)
            for split in (*MIDITOK_TRAINING_SPLITS, "rejected")
        },
        output_dir=str(output_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_config_path=str(tokenizer_config_path),
        tokenized_segments_path=str(tokenized_segments_path),
        metadata_path=str(metadata_path),
        license_report_path=str(license_report_path),
        summary_path=str(summary_path),
        average_information_loss_ratio=round(sum(losses) / len(losses), 6) if losses else 1.0,
        max_information_loss_ratio=round(max(losses), 6) if losses else 1.0,
        acceptable_information_loss=bool(losses)
        and max(losses) <= max_acceptable_loss_ratio,
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def load_miditok_segments(path: str | Path) -> list[MidiTokRoleSegment]:
    return [
        MidiTokRoleSegment.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _import_miditok() -> Any:
    try:
        return importlib.import_module("miditok")
    except ImportError as exc:
        raise MidiTokUnavailableError(
            "MidiTok is not installed. Run: python -m pip install -r requirements-training-ai.txt"
        ) from exc


def _ensure_midi_path(path: str | Path, converted_dir: Path) -> Path:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        return input_path
    if suffix not in {".musicxml", ".xml"}:
        raise ValueError(f"Unsupported source format for MidiTok tokenization: {input_path}")
    converted_dir.mkdir(parents=True, exist_ok=True)
    output_path = converted_dir / f"{input_path.stem}.mid"
    music21_converter = importlib.import_module("music21.converter")
    score = music21_converter.parse(str(input_path))
    written = score.write("midi", fp=str(output_path))
    return Path(written)


def _split_midi_by_role(
    midi_path: Path,
    output_dir: Path,
    *,
    source: MidiTokSource,
    roles: tuple[str, ...],
) -> dict[str, Path]:
    midi = mido.MidiFile(midi_path)
    events_by_role: dict[str, list[mido.MidiTrack]] = {role: [] for role in roles}
    for track_index, track in enumerate(midi.tracks):
        role = _role_for_track(track, track_index, source)
        if role not in roles:
            continue
        if _note_count_for_track(track) == 0:
            continue
        events_by_role[role].append(track)

    role_paths: dict[str, Path] = {}
    for role, tracks in events_by_role.items():
        if not tracks:
            continue
        role_midi = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
        role_midi.tracks.append(_global_meta_track(midi))
        for index, track in enumerate(tracks):
            fallback_name = f"{source.source_file_id}:{role}:{index}"
            role_midi.tracks.append(_copy_track(track, fallback_name=fallback_name))
        output_path = output_dir / f"{source.source_file_id}_{role}.mid"
        role_midi.save(output_path)
        role_paths[role] = output_path
    return role_paths


def _global_meta_track(midi: mido.MidiFile) -> mido.MidiTrack:
    output = mido.MidiTrack()
    output.append(mido.MetaMessage("track_name", name="global", time=0))
    for track in midi.tracks[:1]:
        for message in track:
            if message.is_meta and message.type in {"set_tempo", "time_signature", "key_signature"}:
                output.append(message.copy())
    return output


def _copy_track(track: mido.MidiTrack, *, fallback_name: str) -> mido.MidiTrack:
    copied = mido.MidiTrack()
    has_name = False
    for message in track:
        copied.append(message.copy())
        if message.is_meta and message.type == "track_name":
            has_name = True
    if not has_name:
        copied.insert(0, mido.MetaMessage("track_name", name=fallback_name, time=0))
    return copied


def _role_for_track(track: mido.MidiTrack, track_index: int, source: MidiTokSource) -> str:
    track_name = _track_name(track)
    explicit_role = (
        source.track_roles.get(str(track_index))
        or source.track_roles.get(track_name)
        or source.track_roles.get(track_name.lower())
        or source.track_roles.get("*")
    )
    if explicit_role:
        return _canonical_role(explicit_role)
    return _infer_role_from_track(track, track_name)


def _infer_role_from_track(track: mido.MidiTrack, track_name: str) -> str:
    normalized = track_name.lower().replace("-", "_").replace(" ", "_")
    for token, role in _ROLE_ALIASES.items():
        if token in normalized:
            return role
    if _track_uses_drum_channel(track):
        return "drums"
    return ""


def _track_name(track: mido.MidiTrack) -> str:
    for message in track:
        if message.is_meta and message.type == "track_name":
            return str(message.name)
    return ""


def _track_uses_drum_channel(track: mido.MidiTrack) -> bool:
    for message in track:
        if not message.is_meta and getattr(message, "channel", None) == 9:
            return True
    return False


def _note_count(path: str | Path) -> int:
    midi = mido.MidiFile(path)
    return sum(_note_count_for_track(track) for track in midi.tracks)


def _note_count_for_track(track: mido.MidiTrack) -> int:
    return sum(
        1
        for message in track
        if not message.is_meta
        and message.type == "note_on"
        and int(getattr(message, "velocity", 0)) > 0
    )


def _role_source_tracks(midi_path: Path, source: MidiTokSource, role: str) -> list[str]:
    midi = mido.MidiFile(midi_path)
    names: list[str] = []
    for index, track in enumerate(midi.tracks):
        if _role_for_track(track, index, source) == role:
            name = _track_name(track) or f"track_{index}"
            names.append(name)
    return names


def _sequence_list(sequence_or_sequences: Any) -> list[Any]:
    if isinstance(sequence_or_sequences, list):
        return sequence_or_sequences
    return [sequence_or_sequences]


def _train_eligible(source: MidiTokSource) -> bool:
    if not source.training_allowed:
        return False
    if _blocked_license(source.license):
        return False
    return _normalize_flag(source.commercial_training) not in BLOCKED_COMMERCIAL_TRAINING


def _license_rejection_reason(source: MidiTokSource) -> str:
    if not source.training_allowed:
        return "source_training_not_allowed"
    if _blocked_license(source.license):
        return "blocked_or_unknown_license"
    if _normalize_flag(source.commercial_training) in BLOCKED_COMMERCIAL_TRAINING:
        return "commercial_training_not_allowed"
    return "not_rejected"


def _blocked_license(license_name: str) -> bool:
    return _normalize_flag(license_name) in BLOCKED_TRAINING_LICENSES


def _normalize_flag(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _canonical_role(role: str) -> str:
    normalized = role.strip().lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(normalized, normalized)


def _loss_ratio(input_count: int, reconstructed_count: int) -> float:
    if input_count <= 0:
        return 0.0 if reconstructed_count == 0 else 1.0
    return round(abs(reconstructed_count - input_count) / input_count, 6)


def _segment_metadata(segment: MidiTokRoleSegment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "split": segment.split,
        "role": segment.role,
        "style": segment.style,
        "chord_context": segment.chord_context,
        "source_file_id": segment.source_file_id,
        "source_hash": segment.source_hash,
        "source_dataset": segment.source_dataset,
        "license": segment.license,
        "commercial_training": segment.commercial_training,
        "train_eligible": segment.train_eligible,
        "token_count": segment.token_count,
        "note_count_input": segment.note_count_input,
        "note_count_reconstructed": segment.note_count_reconstructed,
        "information_loss_ratio": segment.information_loss_ratio,
        "midi_path": segment.midi_path,
        "reconstructed_midi_path": segment.reconstructed_midi_path,
        "metadata": segment.metadata,
    }


def _license_report(
    sources: list[MidiTokSource],
    segments: list[MidiTokRoleSegment],
    rejected_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": {
            "blocked_licenses": sorted(BLOCKED_TRAINING_LICENSES),
            "blocked_commercial_training_flags": sorted(BLOCKED_COMMERCIAL_TRAINING),
        },
        "sources": [
            {
                "source_file_id": source.source_file_id,
                "source_path": source.path,
                "license": source.license,
                "commercial_training": source.commercial_training,
                "training_allowed": source.training_allowed,
                "train_eligible": _train_eligible(source),
            }
            for source in sources
        ],
        "rejected_sources": rejected_sources,
        "segments": [
            {
                "id": segment.id,
                "role": segment.role,
                "split": segment.split,
                "source_file_id": segment.source_file_id,
                "license": segment.license,
                "train_eligible": segment.train_eligible,
            }
            for segment in segments
        ],
    }


def _append_jsonl(path: Path, model: BaseModel | dict[str, Any]) -> None:
    payload = model.model_dump(mode="json") if isinstance(model, BaseModel) else model
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
