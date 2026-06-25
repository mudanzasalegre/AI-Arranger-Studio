from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arranger_core.song_planner import SongPlan

LLM_PLANNER_SCHEMA_VERSION = "0.1.0"

ALLOWED_MODEL_OPERATIONS = {
    "plan_song",
    "patch_plan",
    "build_role_intent",
    "choose_generation_strategy",
}


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleIntent(PlanModel):
    role: str
    instruments: list[str] = Field(default_factory=list)
    target_sections: list[str] = Field(default_factory=list)
    density: float = Field(default=0.5, ge=0.0, le=1.0)
    strategy: str | None = None
    allowed_operations: list[str] = Field(default_factory=lambda: ["plan_song"])
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_operations")
    @classmethod
    def operations_must_stay_in_planning_layer(cls, value: list[str]) -> list[str]:
        forbidden = sorted(set(value) - ALLOWED_MODEL_OPERATIONS)
        if forbidden:
            raise ValueError(f"Unsupported planner operations: {forbidden}")
        return value


class GenerationStrategy(PlanModel):
    mode: Literal["rule_based", "llm_plan", "hybrid_symbolic", "retrieval_ready"] = "llm_plan"
    priority_roles: list[str] = Field(default_factory=list)
    role_intents: list[RoleIntent] = Field(default_factory=list)
    forbid_audio_models: bool = True
    allow_note_generation: bool = False
    allow_midi_export: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def forbid_non_planning_outputs(self) -> GenerationStrategy:
        if not self.forbid_audio_models:
            raise ValueError("Audio model usage is forbidden for the LLM planner")
        if self.allow_note_generation:
            raise ValueError("The LLM planner cannot generate final notes")
        if self.allow_midi_export:
            raise ValueError("The LLM planner cannot export MIDI")
        return self


class LlmSectionPatch(PlanModel):
    name: str
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)
    energy: float = Field(ge=0.0, le=1.0)
    density_by_role: dict[str, float] = Field(default_factory=dict)
    groove_feel: str | None = None
    role_focus: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("density_by_role")
    @classmethod
    def densities_must_be_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        invalid = {
            role: density
            for role, density in value.items()
            if density < 0.0 or density > 1.0
        }
        if invalid:
            raise ValueError(f"Role densities must be between 0 and 1: {invalid}")
        return value

    @model_validator(mode="after")
    def section_range_must_be_ordered(self) -> LlmSectionPatch:
        if self.end_bar < self.start_bar:
            raise ValueError("section end_bar must be greater than or equal to start_bar")
        return self


class LlmSongPlanPatch(PlanModel):
    schema_version: str = LLM_PLANNER_SCHEMA_VERSION
    style: str
    substyle: str | None = None
    tempo: int = Field(ge=40, le=260)
    meter: str
    key: str
    form: str
    ensemble: str
    instruments: list[str]
    sections: list[LlmSectionPatch] = Field(min_length=1)
    generation_strategy: GenerationStrategy = Field(default_factory=GenerationStrategy)
    role_intents: list[RoleIntent] = Field(default_factory=list)


class PlanAttempt(PlanModel):
    attempt: int
    source: Literal["llm", "fallback_rule_based"]
    status: Literal["pass", "fail"]
    error: str | None = None


class LlmPlannerResult(PlanModel):
    status: Literal["ok", "failed"]
    planner: Literal["llm", "fallback_rule_based"]
    plan_version: str
    song_plan_patch: LlmSongPlanPatch
    song_plan: SongPlan
    validation: dict[str, Any]
    attempts: list[PlanAttempt] = Field(default_factory=list)
    fallback_used: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
