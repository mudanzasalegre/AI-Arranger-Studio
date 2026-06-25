from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from arranger_core.catalogs import InstrumentCatalog, StyleProfileCatalog
from arranger_core.harmony_engine import FORM_ALIASES, parse_key
from arranger_core.planning.plan_schema import LlmSongPlanPatch
from arranger_core.schema import ArrangementProject, GenerationSpec, meter_to_quarter_beats
from arranger_core.song_planner import SongPlan

FORBIDDEN_OUTPUT_KEYS = {
    "audio",
    "audio_path",
    "audio_url",
    "exports",
    "lyrics",
    "midi",
    "midi_bytes",
    "midi_file",
    "midi_path",
    "musicxml",
    "musicxml_path",
    "note_event",
    "note_events",
    "notes_events",
    "pitch",
    "pitches",
    "score_pdf",
    "velocity",
}

SUPPORTED_FORMS = set(FORM_ALIASES) | set(FORM_ALIASES.values())


class PlanValidator:
    def __init__(
        self,
        *,
        instrument_catalog: InstrumentCatalog | None = None,
        style_catalog: StyleProfileCatalog | None = None,
    ) -> None:
        self.instrument_catalog = instrument_catalog or InstrumentCatalog.load_default()
        self.style_catalog = style_catalog or StyleProfileCatalog.load_default()

    def parse_patch_json(self, raw_json: str) -> tuple[LlmSongPlanPatch | None, dict[str, Any]]:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return None, _report(
                errors=[
                    _issue(
                        "invalid_json",
                        f"LLM response is not valid JSON: {exc.msg}",
                        details={"line": exc.lineno, "column": exc.colno},
                    )
                ]
            )
        if not isinstance(payload, dict):
            return None, _report(
                errors=[_issue("invalid_json_root", "Planner JSON must be an object")]
            )

        forbidden_paths = _forbidden_key_paths(payload)
        if forbidden_paths:
            return None, _report(
                errors=[
                    _issue(
                        "forbidden_output_key",
                        "Planner JSON contains output fields outside the planning layer.",
                        details={"paths": forbidden_paths},
                    )
                ]
            )

        try:
            patch = LlmSongPlanPatch.model_validate(payload)
        except ValidationError as exc:
            return None, _report(
                errors=[
                    _issue(
                        "schema_validation_failed",
                        "Planner JSON does not match LlmSongPlanPatch.",
                        details={"errors": exc.errors()},
                    )
                ]
            )
        return patch, self.validate_patch(patch)

    def validate_patch(
        self,
        patch: LlmSongPlanPatch,
        *,
        project: ArrangementProject | None = None,
        locked_tracks: list[str] | None = None,
        locked_sections: list[str] | None = None,
    ) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if patch.style not in self.style_catalog.styles:
            errors.append(_issue("unknown_style", f"Unknown style: {patch.style}"))

        if patch.form not in SUPPORTED_FORMS:
            errors.append(_issue("unknown_form", f"Unsupported form: {patch.form}"))

        if patch.ensemble not in self.instrument_catalog.ensembles:
            errors.append(_issue("unknown_ensemble", f"Unknown ensemble: {patch.ensemble}"))

        unknown_instruments = [
            instrument_id
            for instrument_id in patch.instruments
            if instrument_id not in self.instrument_catalog.instruments
        ]
        if unknown_instruments:
            errors.append(
                _issue(
                    "unknown_instruments",
                    "Planner requested instruments that are not in the catalog.",
                    details={"instruments": unknown_instruments},
                )
            )

        try:
            meter_to_quarter_beats(patch.meter)
        except ValueError as exc:
            errors.append(_issue("invalid_meter", str(exc)))

        try:
            parse_key(patch.key)
        except ValueError as exc:
            errors.append(_issue("invalid_key", str(exc)))

        errors.extend(_section_range_errors(patch, project=project))
        errors.extend(_locked_target_errors(patch, locked_tracks or [], locked_sections or []))

        try:
            GenerationSpec(
                style=patch.style,
                substyle=patch.substyle,
                tempo=patch.tempo,
                key=patch.key,
                meter=patch.meter,
                form=patch.form,
                ensemble=patch.ensemble,
                instruments=patch.instruments,
            )
        except ValidationError as exc:
            errors.append(
                _issue(
                    "generation_spec_validation_failed",
                    "Patch cannot be represented as GenerationSpec.",
                    details={"errors": exc.errors()},
                )
            )

        if patch.generation_strategy.metadata.get("forbid_audio_models") is True:
            warnings.append(
                _warning(
                    "audio_request_ignored",
                    "Audio generation request was ignored; planner can only produce structure.",
                )
            )

        return _report(errors=errors, warnings=warnings)

    def validate_song_plan(self, song_plan: SongPlan) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        phrase_ids = {phrase.id for phrase in song_plan.phrases}
        missing_phrase_ids = [
            phrase_id
            for section in song_plan.sections
            for phrase_id in section.phrase_ids
            if phrase_id not in phrase_ids
        ]
        if missing_phrase_ids:
            errors.append(
                _issue(
                    "missing_phrase_ids",
                    "SectionPlan references phrase ids that do not exist.",
                    details={"phrase_ids": missing_phrase_ids},
                )
            )
        return _report(errors=errors)


def _section_range_errors(
    patch: LlmSongPlanPatch,
    *,
    project: ArrangementProject | None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    sections = sorted(patch.sections, key=lambda item: (item.start_bar, item.end_bar))
    previous_end = 0
    for section in sections:
        if section.start_bar <= previous_end:
            errors.append(
                _issue(
                    "sections_overlap",
                    "Planner sections cannot overlap.",
                    details={
                        "section": section.name,
                        "start_bar": section.start_bar,
                        "previous_end_bar": previous_end,
                    },
                )
            )
        previous_end = max(previous_end, section.end_bar)

        if project is not None and project.bar_count and section.end_bar > project.bar_count:
            errors.append(
                _issue(
                    "section_outside_project",
                    "Planner section exceeds current project bar count.",
                    details={
                        "section": section.name,
                        "end_bar": section.end_bar,
                        "project_bar_count": project.bar_count,
                    },
                )
            )
    return errors


def _locked_target_errors(
    patch: LlmSongPlanPatch,
    locked_tracks: list[str],
    locked_sections: list[str],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    locked_track_set = {value.lower() for value in locked_tracks}
    locked_section_set = {value.lower() for value in locked_sections}

    touched_tracks = {instrument.lower() for instrument in patch.instruments}
    touched_tracks.update(
        instrument.lower()
        for intent in [*patch.role_intents, *patch.generation_strategy.role_intents]
        for instrument in intent.instruments
    )
    touched_tracks.update(
        role.lower()
        for section in patch.sections
        for role in section.role_focus
    )
    blocked_tracks = sorted(touched_tracks & locked_track_set)
    if blocked_tracks:
        errors.append(
            _issue(
                "locked_track_modified",
                "Planner attempted to target locked tracks.",
                details={"tracks": blocked_tracks},
            )
        )

    section_identities = {
        identity
        for index, section in enumerate(patch.sections, start=1)
        for identity in {
            section.name.lower(),
            _slug(section.name),
            f"section_{index:02d}_{_slug(section.name)}",
        }
    }
    blocked_sections = sorted(section_identities & locked_section_set)
    if blocked_sections:
        errors.append(
            _issue(
                "locked_section_modified",
                "Planner attempted to target locked sections.",
                details={"sections": blocked_sections},
            )
        )
    return errors


def _forbidden_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in FORBIDDEN_OUTPUT_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_forbidden_key_paths(child, prefix=path))
    return paths


def _report(
    *,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warnings = warnings or []
    return {
        "status": "fail" if errors else "pass_with_warnings" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }


def _issue(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "severity": "error",
        "validator": "PlanValidator",
        "code": code,
        "message": message,
        "details": details or {},
    }


def _warning(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "severity": "warning",
        "validator": "PlanValidator",
        "code": code,
        "message": message,
        "details": details or {},
    }


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)
