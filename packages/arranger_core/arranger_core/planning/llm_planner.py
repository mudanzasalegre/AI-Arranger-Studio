from __future__ import annotations

import json
from typing import Any, Protocol
from uuid import uuid4

from arranger_core.planning.plan_schema import (
    GenerationStrategy,
    LlmPlannerResult,
    LlmSectionPatch,
    LlmSongPlanPatch,
    PlanAttempt,
    RoleIntent,
)
from arranger_core.planning.plan_validator import PlanValidator
from arranger_core.prompt_compiler import PromptCompiler
from arranger_core.schema import ArrangementProject, GenerationSpec, meter_to_quarter_beats
from arranger_core.song_planner import (
    EnergyPoint,
    GrooveMap,
    PhrasePlan,
    SectionPlan,
    SongPlan,
    SongPlanner,
)

SYSTEM_PROMPT = """You are a symbolic music planner for a text-to-MIDI application.
Return only valid JSON matching LlmSongPlanPatch.
Do not generate notes, MIDI, MusicXML, lyrics, audio, exports, or score files.
Only plan SongPlan, SectionPlan, PhrasePlan, GrooveMap, RoleIntent, and GenerationStrategy.
"""


class PlannerJsonProvider(Protocol):
    def generate_plan_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        previous_error: str | None = None,
    ) -> str:
        """Return a JSON string for LlmSongPlanPatch."""


class LlmPlanner:
    def __init__(
        self,
        *,
        provider: PlannerJsonProvider | None = None,
        prompt_compiler: PromptCompiler | None = None,
        song_planner: SongPlanner | None = None,
        validator: PlanValidator | None = None,
    ) -> None:
        self.provider = provider
        self.prompt_compiler = prompt_compiler or PromptCompiler()
        self.song_planner = song_planner or SongPlanner()
        self.validator = validator or PlanValidator()

    def plan(
        self,
        *,
        prompt: str,
        project: ArrangementProject,
        mode: str = "create_or_patch_plan",
        locked_tracks: list[str] | None = None,
        locked_sections: list[str] | None = None,
        seed: int | None = None,
    ) -> LlmPlannerResult:
        attempts: list[PlanAttempt] = []
        seed = _resolve_seed(project, seed)

        if self.provider is not None:
            previous_error: str | None = None
            for attempt_number in (1, 2):
                try:
                    raw_json = self.provider.generate_plan_json(
                        prompt=prompt,
                        system_prompt=_system_prompt(
                            mode=mode,
                            project=project,
                            locked_tracks=locked_tracks or [],
                            locked_sections=locked_sections or [],
                            previous_error=previous_error,
                        ),
                        previous_error=previous_error,
                    )
                except Exception as exc:
                    previous_error = f"provider_error: {exc}"
                    attempts.append(
                        PlanAttempt(
                            attempt=attempt_number,
                            source="llm",
                            status="fail",
                            error=previous_error,
                        )
                    )
                    continue
                patch, parse_report = self.validator.parse_patch_json(raw_json)
                if patch is not None:
                    validation = self.validator.validate_patch(
                        patch,
                        project=project,
                        locked_tracks=locked_tracks,
                        locked_sections=locked_sections,
                    )
                else:
                    validation = parse_report

                if patch is not None and validation["status"] != "fail":
                    song_plan = song_plan_from_patch(patch, project=project, seed=seed)
                    plan_report = self.validator.validate_song_plan(song_plan)
                    if plan_report["status"] != "fail":
                        attempts.append(
                            PlanAttempt(
                                attempt=attempt_number,
                                source="llm",
                                status="pass",
                            )
                        )
                        return LlmPlannerResult(
                            status="ok",
                            planner="llm",
                            plan_version=_plan_version(),
                            song_plan_patch=patch,
                            song_plan=song_plan,
                            validation=_merge_reports(validation, plan_report),
                            attempts=attempts,
                        )
                    validation = _merge_reports(validation, plan_report)

                previous_error = _summarize_validation(validation)
                attempts.append(
                    PlanAttempt(
                        attempt=attempt_number,
                        source="llm",
                        status="fail",
                        error=previous_error,
                    )
                )

        fallback_result = self._fallback(
            prompt=prompt,
            project=project,
            locked_tracks=locked_tracks or [],
            locked_sections=locked_sections or [],
            seed=seed,
            attempts=attempts,
        )
        return fallback_result

    def _fallback(
        self,
        *,
        prompt: str,
        project: ArrangementProject,
        locked_tracks: list[str],
        locked_sections: list[str],
        seed: int,
        attempts: list[PlanAttempt],
    ) -> LlmPlannerResult:
        spec = self.prompt_compiler.compile(prompt, seed=seed)
        song_plan = self.song_planner.plan(spec, project)
        patch = patch_from_song_plan(
            song_plan,
            spec=spec,
            instruments=_project_or_spec_instruments(project, spec),
        )
        validation = self.validator.validate_patch(
            patch,
            project=project,
            locked_tracks=locked_tracks,
            locked_sections=locked_sections,
        )
        plan_report = self.validator.validate_song_plan(song_plan)
        merged_validation = _merge_reports(validation, plan_report)
        attempts.append(
            PlanAttempt(
                attempt=len(attempts) + 1,
                source="fallback_rule_based",
                status="pass" if merged_validation["status"] != "fail" else "fail",
                error=(
                    None
                    if merged_validation["status"] != "fail"
                    else _summarize_validation(merged_validation)
                ),
            )
        )
        return LlmPlannerResult(
            status="ok" if merged_validation["status"] != "fail" else "failed",
            planner="fallback_rule_based",
            plan_version=_plan_version(),
            song_plan_patch=patch,
            song_plan=song_plan,
            validation=merged_validation,
            attempts=attempts,
            fallback_used=True,
        )


def song_plan_from_patch(
    patch: LlmSongPlanPatch,
    *,
    project: ArrangementProject,
    seed: int,
) -> SongPlan:
    sections: list[SectionPlan] = []
    phrases: list[PhrasePlan] = []
    for index, section_patch in enumerate(patch.sections, start=1):
        section_id = f"section_{index:02d}_{_slug(section_patch.name)}"
        phrase_ids: list[str] = []
        for phrase_index, (start_bar, end_bar) in enumerate(
            _phrase_boundaries(section_patch.start_bar, section_patch.end_bar),
            start=1,
        ):
            phrase_id = f"{section_id}_phrase_{phrase_index:02d}"
            phrase_ids.append(phrase_id)
            phrases.append(
                PhrasePlan(
                    id=phrase_id,
                    section_id=section_id,
                    start_bar=start_bar,
                    end_bar=end_bar,
                    function=_phrase_function(phrase_index, start_bar, end_bar),
                    motif_id="main_motif",
                    variation=_variation_for_section(section_patch, phrase_index),
                    energy=section_patch.energy,
                    density=_average_density(section_patch),
                    cadence_bar=end_bar,
                    target_role=_target_role(section_patch),
                    target_note=None,
                    breath_points=_breath_points(start_bar, end_bar),
                )
            )
        sections.append(
            SectionPlan(
                id=section_id,
                name=section_patch.name,
                label=None,
                function=_section_function(section_patch, index),
                start_bar=section_patch.start_bar,
                end_bar=section_patch.end_bar,
                energy=section_patch.energy,
                role_densities=dict(section_patch.density_by_role),
                groove_feel=section_patch.groove_feel or _feel_for_style(patch.style, patch.meter),
                register_target=_register_target(section_patch),
                articulation=_articulation_for_style(patch.style),
                harmonic_rhythm=_harmonic_rhythm(section_patch),
                events=_events_for_section(section_patch),
                phrase_ids=phrase_ids,
            )
        )

    groove_map = _groove_map_from_patch(patch, project=project, sections=sections)
    return SongPlan(
        song_id=project.project_id,
        style=patch.style,
        form=patch.form,
        seed=seed,
        tempo_curve=[{"bar": 1, "bpm": patch.tempo}],
        global_energy_curve=[
            EnergyPoint(bar=section.start_bar, energy=section.energy)
            for section in sections
        ],
        sections=sections,
        phrases=phrases,
        groove_map=groove_map,
        main_motif={
            "id": "main_motif",
            "source": "llm_planner_json",
            "role_intents": [
                intent.model_dump(mode="json")
                for intent in [*patch.role_intents, *patch.generation_strategy.role_intents]
            ],
        },
        ending_strategy=_ending_strategy(patch),
        mix_profile=f"{patch.style}_planned",
    )


def patch_from_song_plan(
    song_plan: SongPlan,
    *,
    spec: GenerationSpec,
    instruments: list[str],
) -> LlmSongPlanPatch:
    role_intents = [
        RoleIntent(role=role, instruments=[], density=density, strategy="rule_based")
        for role, density in _aggregate_role_densities(song_plan).items()
    ]
    return LlmSongPlanPatch(
        style=spec.style,
        substyle=spec.substyle,
        tempo=spec.tempo,
        meter=spec.meter,
        key=spec.key,
        form=spec.form,
        ensemble=spec.ensemble,
        instruments=instruments,
        sections=[
            LlmSectionPatch(
                name=section.name,
                start_bar=section.start_bar,
                end_bar=section.end_bar,
                energy=section.energy,
                density_by_role=dict(section.role_densities),
                groove_feel=section.groove_feel,
                role_focus=[_primary_role(section.role_densities)],
            )
            for section in song_plan.sections
        ],
        generation_strategy=GenerationStrategy(
            mode="rule_based",
            priority_roles=[intent.role for intent in role_intents],
            role_intents=role_intents,
        ),
        role_intents=role_intents,
    )


def _system_prompt(
    *,
    mode: str,
    project: ArrangementProject,
    locked_tracks: list[str],
    locked_sections: list[str],
    previous_error: str | None,
) -> str:
    context = {
        "mode": mode,
        "project_id": project.project_id,
        "bar_count": project.bar_count,
        "track_ids": [track.id for track in project.tracks],
        "locked_tracks": locked_tracks,
        "locked_sections": locked_sections,
    }
    if project.generation_spec is not None:
        context["generation_spec"] = {
            "style": project.generation_spec.style,
            "substyle": project.generation_spec.substyle,
            "tempo": project.generation_spec.tempo,
            "meter": project.generation_spec.meter,
            "key": project.generation_spec.key,
            "form": project.generation_spec.form,
            "ensemble": project.generation_spec.ensemble,
            "instruments": project.generation_spec.instruments,
        }
    repair = f"\nPrevious validation error: {previous_error}" if previous_error else ""
    return f"{SYSTEM_PROMPT}\nProject context:\n{json.dumps(context, sort_keys=True)}{repair}"


def _resolve_seed(project: ArrangementProject, seed: int | None) -> int:
    if seed is not None:
        return seed
    if project.generation_spec is not None:
        return project.generation_spec.seed
    return 0


def _project_or_spec_instruments(project: ArrangementProject, spec: GenerationSpec) -> list[str]:
    if spec.instruments:
        return list(spec.instruments)
    return [track.instrument for track in project.tracks]


def _phrase_boundaries(start_bar: int, end_bar: int) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    current = start_bar
    while current <= end_bar:
        phrase_end = min(end_bar, current + 3)
        boundaries.append((current, phrase_end))
        current = phrase_end + 1
    return boundaries


def _phrase_function(phrase_index: int, start_bar: int, end_bar: int) -> str:
    if start_bar == end_bar:
        return "short_gesture"
    if phrase_index == 1:
        return "question"
    return "cadence"


def _variation_for_section(section: LlmSectionPatch, phrase_index: int) -> str:
    if phrase_index == 1:
        return "statement"
    if section.energy >= 0.75:
        return "intensify"
    return "answer"


def _average_density(section: LlmSectionPatch) -> float:
    if not section.density_by_role:
        return round(section.energy, 3)
    return round(sum(section.density_by_role.values()) / len(section.density_by_role), 3)


def _target_role(section: LlmSectionPatch) -> str:
    if section.role_focus:
        return section.role_focus[0]
    if section.density_by_role:
        return max(section.density_by_role.items(), key=lambda item: item[1])[0]
    return "melody"


def _breath_points(start_bar: int, end_bar: int) -> list[int]:
    return [bar for bar in range(start_bar + 1, end_bar + 1, 2)]


def _section_function(section: LlmSectionPatch, index: int) -> str:
    text = f"{section.name} {' '.join(section.role_focus)}".lower()
    if "bridge" in text:
        return "bridge"
    if "turnaround" in text or "ending" in text:
        return "turnaround" if "turnaround" in text else "ending"
    if "response" in text or "horn" in text:
        return "response"
    return "head_statement" if index == 1 else "head_development"


def _register_target(section: LlmSectionPatch) -> str:
    if section.energy >= 0.75:
        return "mid_high"
    if section.energy <= 0.35:
        return "low_mid"
    return "mid"


def _articulation_for_style(style: str) -> str:
    if style == "jazz_ballad":
        return "legato_warm"
    if style == "bossa_nova":
        return "light_detached"
    if style == "funk_jazz":
        return "short_accented"
    return "swing_accented"


def _harmonic_rhythm(section: LlmSectionPatch) -> str:
    if section.end_bar - section.start_bar + 1 <= 2:
        return "active"
    if section.energy < 0.4:
        return "slow"
    return "medium"


def _events_for_section(section: LlmSectionPatch) -> list[str]:
    events: list[str] = []
    horn_roles = {"horns", "horn_response", "trumpet_bflat", "trombone"}
    if any(role in horn_roles for role in section.role_focus):
        events.append("horn_hit")
    if section.energy >= 0.75:
        events.append("drum_setup")
    events.append(f"section_cadence_bar_{section.end_bar}")
    return events


def _groove_map_from_patch(
    patch: LlmSongPlanPatch,
    *,
    project: ArrangementProject,
    sections: list[SectionPlan],
) -> GrooveMap:
    bar_count = max(project.bar_count, max((section.end_bar for section in sections), default=1))
    fill_bars = sorted({section.end_bar for section in sections if section.end_bar <= bar_count})
    setup_bars = sorted({bar - 1 for bar in fill_bars if bar > 1})
    horn_hit_bars = sorted(
        {
            section.start_bar + 1
            for section in sections
            if "horn_hit" in section.events and section.start_bar + 1 <= section.end_bar
        }
    )
    return GrooveMap(
        meter=patch.meter,
        feel=_feel_for_style(patch.style, patch.meter),
        swing_ratio=_swing_ratio(patch.style, patch.tempo),
        beat_grid=_beat_grid(patch.meter),
        fill_bars=fill_bars,
        setup_bars=setup_bars,
        break_bars=[section.start_bar for section in sections if section.energy <= 0.35],
        horn_hit_bars=horn_hit_bars,
        comping_safe_beats=_comping_safe_beats(patch.style, patch.meter),
        kick_lock_beats=_kick_lock_beats(patch.style),
    )


def _feel_for_style(style: str, meter: str) -> str:
    if meter == "3/4" or style == "jazz_waltz":
        return "waltz"
    if style == "bossa_nova":
        return "bossa"
    if style == "funk_jazz":
        return "straight_eighth"
    if style == "jazz_ballad":
        return "slow_swing"
    return "swing"


def _swing_ratio(style: str, tempo: int) -> float:
    if style in {"bossa_nova", "funk_jazz"}:
        return 0.5
    if tempo < 90:
        return 0.66
    if tempo > 180:
        return 0.57
    return 0.61


def _beat_grid(meter: str) -> list[float]:
    beats = meter_to_quarter_beats(meter)
    step_count = max(1, round(beats / 0.5))
    return [round(index * 0.5, 3) for index in range(step_count)]


def _comping_safe_beats(style: str, meter: str) -> list[float]:
    if style == "bossa_nova":
        return [0.5, 1.5, 2.5, 3.0]
    if meter == "3/4":
        return [0.5, 1.5, 2.0]
    return [0.5, 1.5, 2.5, 3.5]


def _kick_lock_beats(style: str) -> list[float]:
    if style == "funk_jazz":
        return [0.0, 1.5, 2.5]
    if style == "bossa_nova":
        return [0.0, 2.0]
    return [0.0, 2.0]


def _ending_strategy(patch: LlmSongPlanPatch) -> str:
    if patch.style == "jazz_ballad":
        return "soft_tag"
    if patch.style == "funk_jazz":
        return "short_hit"
    return "planned_cadence"


def _aggregate_role_densities(song_plan: SongPlan) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for section in song_plan.sections:
        for role, density in section.role_densities.items():
            totals.setdefault(role, []).append(density)
    return {
        role: round(sum(values) / len(values), 3)
        for role, values in totals.items()
        if values
    }


def _primary_role(role_densities: dict[str, float]) -> str:
    if not role_densities:
        return "melody"
    return max(role_densities.items(), key=lambda item: item[1])[0]


def _merge_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    errors = [issue for report in reports for issue in report.get("errors", [])]
    warnings = [issue for report in reports for issue in report.get("warnings", [])]
    return {
        "status": "fail" if errors else "pass_with_warnings" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }


def _summarize_validation(report: dict[str, Any]) -> str:
    issues = report.get("errors") or report.get("warnings") or []
    if not issues:
        return "Unknown planner validation failure"
    first = issues[0]
    return f"{first.get('code', 'validation_error')}: {first.get('message', '')}"


def _plan_version() -> str:
    return f"plan_{uuid4().hex[:12]}"


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part) or "section"
