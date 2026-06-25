from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arranger_core.schema import ArrangementProject, GenerationSpec, Section, meter_to_quarter_beats

SONG_PLAN_VERSION = "0.1.0"


class SongPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnergyPoint(SongPlanModel):
    bar: int = Field(ge=1)
    energy: float = Field(ge=0.0, le=1.0)


class PhrasePlan(SongPlanModel):
    id: str
    section_id: str
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)
    function: str
    motif_id: str
    variation: str
    energy: float = Field(ge=0.0, le=1.0)
    density: float = Field(ge=0.0, le=1.0)
    cadence_bar: int = Field(ge=1)
    target_role: str
    target_note: str | None = None
    breath_points: list[int] = Field(default_factory=list)

    @field_validator("end_bar")
    @classmethod
    def end_bar_is_positive(cls, value: int) -> int:
        return value


class SectionPlan(SongPlanModel):
    id: str
    name: str
    label: str | None = None
    function: str
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)
    energy: float = Field(ge=0.0, le=1.0)
    role_densities: dict[str, float] = Field(default_factory=dict)
    groove_feel: str
    register_target: str
    articulation: str
    harmonic_rhythm: str
    events: list[str] = Field(default_factory=list)
    phrase_ids: list[str] = Field(default_factory=list)

    @property
    def duration_bars(self) -> int:
        return self.end_bar - self.start_bar + 1


class GrooveMap(SongPlanModel):
    meter: str
    feel: str
    swing_ratio: float = Field(ge=0.5, le=0.75)
    beat_grid: list[float] = Field(default_factory=list)
    fill_bars: list[int] = Field(default_factory=list)
    setup_bars: list[int] = Field(default_factory=list)
    break_bars: list[int] = Field(default_factory=list)
    horn_hit_bars: list[int] = Field(default_factory=list)
    comping_safe_beats: list[float] = Field(default_factory=list)
    kick_lock_beats: list[float] = Field(default_factory=list)


class SongPlan(SongPlanModel):
    schema_version: str = SONG_PLAN_VERSION
    song_id: str
    style: str
    form: str
    seed: int
    tempo_curve: list[dict[str, int]] = Field(default_factory=list)
    global_energy_curve: list[EnergyPoint] = Field(default_factory=list)
    sections: list[SectionPlan] = Field(default_factory=list)
    phrases: list[PhrasePlan] = Field(default_factory=list)
    groove_map: GrooveMap
    main_motif: dict[str, Any] = Field(default_factory=dict)
    ending_strategy: str
    mix_profile: str

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> SongPlan:
        if isinstance(data, bytes | bytearray):
            data = data.decode("utf-8")
        return cls.model_validate_json(data)

    def save_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> SongPlan:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class SongPlanner:
    def plan(self, spec: GenerationSpec, project: ArrangementProject) -> SongPlan:
        rng = random.Random(spec.seed + 16001)
        section_blueprints = _section_blueprints(spec, project)
        sections: list[SectionPlan] = []
        phrases: list[PhrasePlan] = []
        for index, blueprint in enumerate(section_blueprints):
            energy = _section_energy(spec, blueprint["function"], index, len(section_blueprints))
            section_id = f"section_{index + 1:02d}_{_slug(blueprint['name'])}"
            phrase_ids: list[str] = []
            section_phrases = _phrase_boundaries(
                start_bar=blueprint["start_bar"],
                end_bar=blueprint["end_bar"],
            )
            for phrase_index, (start_bar, end_bar) in enumerate(section_phrases, start=1):
                phrase_id = f"{section_id}_phrase_{phrase_index:02d}"
                phrase_ids.append(phrase_id)
                phrases.append(
                    PhrasePlan(
                        id=phrase_id,
                        section_id=section_id,
                        start_bar=start_bar,
                        end_bar=end_bar,
                        function=_phrase_function(phrase_index, len(section_phrases)),
                        motif_id="main_motif",
                        variation=_motif_variation(blueprint["function"], phrase_index),
                        energy=round(min(1.0, energy + (phrase_index - 1) * 0.04), 3),
                        density=round(_density_for_energy(spec, energy), 3),
                        cadence_bar=end_bar,
                        target_role=_target_role_for_section(blueprint["function"]),
                        target_note=_target_note(spec, rng),
                        breath_points=_breath_points(start_bar, end_bar),
                    )
                )

            sections.append(
                SectionPlan(
                    id=section_id,
                    name=blueprint["name"],
                    label=blueprint.get("label"),
                    function=blueprint["function"],
                    start_bar=blueprint["start_bar"],
                    end_bar=blueprint["end_bar"],
                    energy=round(energy, 3),
                    role_densities=_role_densities(spec, energy, blueprint["function"]),
                    groove_feel=_groove_feel(spec),
                    register_target=_register_target(spec, blueprint["function"]),
                    articulation=_articulation(spec),
                    harmonic_rhythm=_harmonic_rhythm(spec, blueprint["function"]),
                    events=_section_events(blueprint["function"], blueprint["end_bar"]),
                    phrase_ids=phrase_ids,
                )
            )

        groove_map = _groove_map(spec, project, sections)
        return SongPlan(
            song_id=project.project_id,
            style=spec.style,
            form=spec.form,
            seed=spec.seed,
            tempo_curve=[{"bar": 1, "bpm": spec.tempo}],
            global_energy_curve=[
                EnergyPoint(bar=section.start_bar, energy=section.energy)
                for section in sections
            ],
            sections=sections,
            phrases=phrases,
            groove_map=groove_map,
            main_motif=_main_motif(spec, rng),
            ending_strategy=_ending_strategy(spec),
            mix_profile=_mix_profile(spec),
        )


def generate_song_plan(spec: GenerationSpec, project: ArrangementProject) -> SongPlan:
    return SongPlanner().plan(spec, project)


def _section_blueprints(spec: GenerationSpec, project: ArrangementProject) -> list[dict[str, Any]]:
    form = spec.form.lower()
    if form in {"minor_blues_12", "jazz_blues_12", "blues_12"} and project.bar_count >= 12:
        return [
            _blueprint("Head Statement", 1, 4, "head_statement", label="A"),
            _blueprint("Response", 5, 8, "response", label="B"),
            _blueprint("Turnaround", 9, 12, "turnaround", label="T"),
        ]
    if "ballad" in form and project.form:
        return [
            _from_section(project.form[0], "intro_head"),
            *[
                _from_section(section, "bridge" if _is_bridge(section) else "head_development")
                for section in project.form[1:-1]
            ],
            _from_section(project.form[-1], "ending"),
        ]
    if project.form:
        return [
            _from_section(
                section,
                "bridge" if _is_bridge(section) else _default_section_function(index),
            )
            for index, section in enumerate(project.form)
        ]
    bar_count = spec.duration_bars or project.bar_count or 8
    return [_blueprint("Whole Song", 1, bar_count, "head_statement")]


def _blueprint(
    name: str,
    start_bar: int,
    end_bar: int,
    function: str,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "start_bar": start_bar,
        "end_bar": end_bar,
        "function": function,
    }


def _from_section(section: Section, function: str) -> dict[str, Any]:
    return _blueprint(
        section.name,
        section.start_bar,
        section.end_bar,
        function,
        label=section.label,
    )


def _section_energy(
    spec: GenerationSpec,
    function: str,
    index: int,
    section_count: int,
) -> float:
    if spec.style == "jazz_ballad" or "ballad" in spec.form:
        profile = {
            "intro_head": 0.32,
            "head_development": 0.44,
            "bridge": 0.52,
            "ending": 0.34,
        }
        return profile.get(function, 0.4)
    if function == "bridge":
        return 0.78 if spec.style in {"swing", "bebop"} else 0.68
    if function == "turnaround":
        return 0.82
    if function == "response":
        return 0.68
    if function == "ending":
        return 0.46
    if spec.style == "funk_jazz":
        return min(0.9, 0.62 + index * 0.08)
    if section_count > 1:
        return min(0.86, 0.54 + index * 0.06)
    return 0.58


def _role_densities(spec: GenerationSpec, energy: float, function: str) -> dict[str, float]:
    scale = 0.78 if spec.style == "jazz_ballad" else 1.0
    if function in {"intro_head", "ending"}:
        scale *= 0.72
    return {
        "drums": round(min(1.0, (0.45 + energy * 0.45) * scale), 3),
        "walking_bass": round(min(1.0, (0.55 + energy * 0.25) * scale), 3),
        "comping": round(min(1.0, (0.35 + energy * 0.55) * scale), 3),
        "melody": round(min(1.0, (0.48 + energy * 0.35) * scale), 3),
        "horn_response": round(min(1.0, (0.2 + energy * 0.5) * scale), 3),
    }


def _groove_map(
    spec: GenerationSpec,
    project: ArrangementProject,
    sections: list[SectionPlan],
) -> GrooveMap:
    bar_count = max(1, project.bar_count)
    fill_bars = sorted({section.end_bar for section in sections})
    setup_bars = sorted({max(1, bar - 1) for bar in fill_bars})
    horn_hit_bars = sorted(
        {
            section.start_bar + 1
            for section in sections
            if section.function in {"response", "bridge", "turnaround"}
            and section.start_bar + 1 <= section.end_bar
        }
    )
    return GrooveMap(
        meter=spec.meter,
        feel=_groove_feel(spec),
        swing_ratio=_swing_ratio(spec),
        beat_grid=_beat_grid(spec.meter),
        fill_bars=[bar for bar in fill_bars if 1 <= bar <= bar_count],
        setup_bars=[bar for bar in setup_bars if 1 <= bar <= bar_count],
        break_bars=[
            section.start_bar
            for section in sections
            if section.function in {"bridge", "ending"}
        ],
        horn_hit_bars=horn_hit_bars,
        comping_safe_beats=_comping_safe_beats(spec),
        kick_lock_beats=_kick_lock_beats(spec),
    )


def _main_motif(spec: GenerationSpec, rng: random.Random) -> dict[str, Any]:
    contour_options = {
        "jazz_ballad": ["up", "hold", "down", "resolve"],
        "bebop": ["up", "chromatic", "up", "drop"],
        "bossa_nova": ["step", "up", "syncopate", "resolve"],
        "funk_jazz": ["repeat", "jump", "repeat", "answer"],
    }
    contour = contour_options.get(spec.style, ["up", "step", "down", "resolve"])
    return {
        "id": "main_motif",
        "seed_variant": rng.randint(1, 9999),
        "contour": contour,
        "rhythm_cell": _motif_rhythm(spec),
        "development_strategy": ["repeat", "sequence", "fragment", "answer"],
    }


def _phrase_boundaries(start_bar: int, end_bar: int) -> list[tuple[int, int]]:
    length = end_bar - start_bar + 1
    phrase_size = 4 if length >= 4 else max(1, length)
    boundaries = []
    current = start_bar
    while current <= end_bar:
        phrase_end = min(end_bar, current + phrase_size - 1)
        boundaries.append((current, phrase_end))
        current = phrase_end + 1
    return boundaries


def _phrase_function(index: int, count: int) -> str:
    if count == 1:
        return "complete_phrase"
    if index == 1:
        return "question"
    if index == count:
        return "cadence"
    return "answer"


def _motif_variation(section_function: str, phrase_index: int) -> str:
    if section_function == "bridge":
        return "sequence"
    if section_function in {"turnaround", "ending"}:
        return "cadential_fragment"
    return ("repeat" if phrase_index == 1 else "answer")


def _density_for_energy(spec: GenerationSpec, energy: float) -> float:
    scale = 0.72 if spec.style == "jazz_ballad" else 1.0
    return min(1.0, (0.35 + energy * 0.55) * scale)


def _target_role_for_section(function: str) -> str:
    if function in {"response", "turnaround", "bridge"}:
        return "horn_response"
    return "melody"


def _target_note(spec: GenerationSpec, rng: random.Random) -> str:
    root = spec.key.split()[0] if spec.key else "C"
    options = [root, f"{root}3", f"{root}4"]
    return rng.choice(options)


def _breath_points(start_bar: int, end_bar: int) -> list[int]:
    return [bar for bar in range(start_bar + 1, end_bar + 1, 2)]


def _section_events(function: str, end_bar: int) -> list[str]:
    events = []
    if function in {"bridge", "response", "turnaround"}:
        events.append("horn_hit")
    if function in {"turnaround", "ending"}:
        events.append("section_cadence")
    events.append(f"drum_fill_before_bar_{end_bar + 1}")
    return events


def _groove_feel(spec: GenerationSpec) -> str:
    if spec.meter == "3/4" or spec.style == "jazz_waltz":
        return "waltz"
    if spec.style == "bossa_nova":
        return "bossa"
    if spec.style == "funk_jazz":
        return "straight_eighth"
    if spec.style == "jazz_ballad":
        return "slow_swing"
    return "swing"


def _swing_ratio(spec: GenerationSpec) -> float:
    if spec.style in {"bossa_nova", "funk_jazz"}:
        return 0.5
    if spec.tempo < 90:
        return 0.66
    if spec.tempo > 180:
        return 0.57
    return 0.61


def _beat_grid(meter: str) -> list[float]:
    beats = meter_to_quarter_beats(meter)
    step_count = max(1, round(beats / 0.5))
    return [round(index * 0.5, 3) for index in range(step_count)]


def _comping_safe_beats(spec: GenerationSpec) -> list[float]:
    if spec.style == "bossa_nova":
        return [0.5, 1.5, 2.5, 3.0]
    if spec.meter == "3/4":
        return [0.5, 1.5, 2.0]
    return [0.5, 1.5, 2.5, 3.5]


def _kick_lock_beats(spec: GenerationSpec) -> list[float]:
    if spec.style == "funk_jazz":
        return [0.0, 1.5, 2.5]
    if spec.style == "bossa_nova":
        return [0.0, 2.0]
    return [0.0, 2.0]


def _register_target(spec: GenerationSpec, function: str) -> str:
    if spec.style == "jazz_ballad" or function in {"intro_head", "ending"}:
        return "low_mid"
    if function == "bridge":
        return "mid_high"
    return "mid"


def _articulation(spec: GenerationSpec) -> str:
    if spec.style == "funk_jazz":
        return "short_accented"
    if spec.style == "jazz_ballad":
        return "legato_warm"
    if spec.style == "bossa_nova":
        return "light_detached"
    return "swing_accented"


def _harmonic_rhythm(spec: GenerationSpec, function: str) -> str:
    if function == "bridge":
        return "active"
    if spec.style == "modal_jazz":
        return "static_vamp"
    if spec.style == "jazz_ballad":
        return "slow"
    return "medium"


def _ending_strategy(spec: GenerationSpec) -> str:
    if spec.style == "jazz_ballad":
        return "soft_tag"
    if spec.style == "funk_jazz":
        return "short_hit"
    if spec.style == "bossa_nova":
        return "light_tag"
    return "turnaround_tag"


def _mix_profile(spec: GenerationSpec) -> str:
    if spec.style == "jazz_ballad":
        return "ballad_warm"
    if spec.style == "funk_jazz":
        return "funk_tight"
    if spec.style == "bossa_nova":
        return "bossa_clean"
    if spec.style == "modal_jazz":
        return "modal_spacious"
    return "jazz_small_combo"


def _motif_rhythm(spec: GenerationSpec) -> list[float]:
    if spec.style == "jazz_waltz" or spec.meter == "3/4":
        return [1.0, 0.5, 1.5]
    if spec.style == "bossa_nova":
        return [0.5, 1.0, 0.5, 2.0]
    return [1.0, 0.5, 0.5, 2.0]


def _is_bridge(section: Section) -> bool:
    text = f"{section.name} {section.label or ''}".lower()
    return "bridge" in text or section.label == "B"


def _default_section_function(index: int) -> str:
    return "head_statement" if index == 0 else "head_development"


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)
