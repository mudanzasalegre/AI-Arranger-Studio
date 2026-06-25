from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arranger_core.config_loader import MusicConfigLoader
from arranger_core.music_theory import (
    interval_to_note_name,
    note_to_midi,
    pitch_class,
    transpose_note,
)


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Instrument(CatalogModel):
    id: str
    display_name: str
    family: str
    midi_program: int | None = None
    midi_channel: int | None = None
    clef: str
    transposition_semitones: int = 0
    sounding_range: tuple[str, str]
    comfortable_range: tuple[str, str]
    polyphonic: bool
    breath_required: bool


class Ensemble(CatalogModel):
    id: str
    instruments: list[str]


class ProgressionTemplate(CatalogModel):
    id: str
    bars: int
    degrees: list[Any]
    variations: list[Any] = Field(default_factory=list)


class StyleProfile(CatalogModel):
    style: str
    family: str
    meter_options: list[str] = Field(default_factory=list)
    tempo_ranges: dict[str, tuple[int, int]] = Field(default_factory=dict)
    feel: dict[str, Any] = Field(default_factory=dict)
    harmony: dict[str, Any] = Field(default_factory=dict)
    rhythm: dict[str, Any] = Field(default_factory=dict)
    melody: dict[str, Any] = Field(default_factory=dict)
    instrumentation: dict[str, Any] = Field(default_factory=dict)
    roles: dict[str, str] = Field(default_factory=dict)


class InstrumentCatalog:
    def __init__(
        self,
        instruments: dict[str, Instrument],
        ensembles: dict[str, Ensemble] | None = None,
    ) -> None:
        self.instruments = instruments
        self.ensembles = ensembles or {}

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> InstrumentCatalog:
        loader = MusicConfigLoader(config_root)
        instrument_data = loader.load_yaml("instruments.yaml").get("instruments", {})
        ensemble_data = loader.load_yaml("ensembles.yaml").get("ensembles", {})

        instruments = {
            instrument_id: Instrument(id=instrument_id, **data)
            for instrument_id, data in instrument_data.items()
        }
        ensembles = {
            ensemble_id: Ensemble(id=ensemble_id, **data)
            for ensemble_id, data in ensemble_data.items()
        }
        return cls(instruments=instruments, ensembles=ensembles)

    def get(self, instrument_id: str) -> Instrument:
        try:
            return self.instruments[instrument_id]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument: {instrument_id}") from exc

    def get_ensemble(self, ensemble_id: str) -> Ensemble:
        try:
            return self.ensembles[ensemble_id]
        except KeyError as exc:
            raise KeyError(f"Unknown ensemble: {ensemble_id}") from exc

    def instruments_for_ensemble(self, ensemble_id: str) -> list[Instrument]:
        return [
            self.get(instrument_id)
            for instrument_id in self.get_ensemble(ensemble_id).instruments
        ]

    def written_to_sounding(self, instrument_id: str, written_note: str) -> str:
        instrument = self.get(instrument_id)
        return transpose_note(written_note, instrument.transposition_semitones)

    def sounding_to_written(self, instrument_id: str, sounding_note: str) -> str:
        instrument = self.get(instrument_id)
        return transpose_note(sounding_note, -instrument.transposition_semitones)

    def written_midi_to_sounding_midi(self, instrument_id: str, written_note: str) -> int:
        instrument = self.get(instrument_id)
        return note_to_midi(written_note) + instrument.transposition_semitones


class ScaleCatalog:
    def __init__(self, scales: dict[str, list[int]]) -> None:
        self.scales = scales

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> ScaleCatalog:
        loader = MusicConfigLoader(config_root)
        scales = loader.load_yaml("scales.yaml").get("scales", {})
        return cls(scales={scale_id: list(intervals) for scale_id, intervals in scales.items()})

    def get(self, scale_id: str) -> list[int]:
        try:
            return self.scales[scale_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scale: {scale_id}") from exc

    def notes(self, root: str, scale_id: str) -> list[str]:
        prefer_sharps = "#" in root and "b" not in root
        return [
            interval_to_note_name(root, interval, prefer_sharps=prefer_sharps)
            for interval in self.get(scale_id)
        ]

    def pitch_classes(self, root: str, scale_id: str) -> list[int]:
        root_pc = pitch_class(root)
        return [(root_pc + interval) % 12 for interval in self.get(scale_id)]


class ProgressionLibrary:
    def __init__(self, progressions: dict[str, ProgressionTemplate]) -> None:
        self.progressions = progressions

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> ProgressionLibrary:
        loader = MusicConfigLoader(config_root)
        data = loader.load_yaml("jazz_progressions.yaml").get("progressions", {})
        progressions = {
            progression_id: ProgressionTemplate(id=progression_id, **progression)
            for progression_id, progression in data.items()
        }
        return cls(progressions)

    def get(self, progression_id: str) -> ProgressionTemplate:
        try:
            return self.progressions[progression_id]
        except KeyError as exc:
            raise KeyError(f"Unknown progression: {progression_id}") from exc


class StyleProfileCatalog:
    def __init__(self, styles: dict[str, StyleProfile]) -> None:
        self.styles = styles

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> StyleProfileCatalog:
        loader = MusicConfigLoader(config_root)
        loaded = loader.load_yaml_files("style_profiles/jazz/*.yaml")
        styles = {
            data["style"]: StyleProfile(**data)
            for data in loaded.values()
        }
        return cls(styles)

    def get(self, style_id: str) -> StyleProfile:
        try:
            return self.styles[style_id]
        except KeyError as exc:
            raise KeyError(f"Unknown style profile: {style_id}") from exc


class PatternLibrary:
    def __init__(self, patterns: dict[str, dict[str, Any]]) -> None:
        self.patterns = patterns

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> PatternLibrary:
        loader = MusicConfigLoader(config_root)
        loaded = loader.load_yaml_files("patterns/*.yaml")
        patterns: dict[str, dict[str, Any]] = {}
        for data in loaded.values():
            patterns.update(data)
        return cls(patterns)

    def category(self, category_id: str) -> dict[str, Any]:
        try:
            return self.patterns[category_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pattern category: {category_id}") from exc

    def get(self, category_id: str, pattern_id: str) -> Any:
        category = self.category(category_id)
        try:
            return category[pattern_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pattern {pattern_id!r} in {category_id!r}") from exc
