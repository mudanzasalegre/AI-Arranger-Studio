from __future__ import annotations

import json

from arranger_core import GenerationSpec, LlmPlanner, PlanValidator, generate_arrangement


def test_llm_planner_accepts_valid_json_without_generating_notes():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=401),
        project_id="planner-valid",
    )
    result = LlmPlanner(provider=_SequenceProvider([_valid_patch_json()])).plan(
        prompt="hard bop nocturno con respuestas de metales",
        project=project,
        locked_tracks=[],
        locked_sections=[],
    )

    assert result.status == "ok"
    assert result.planner == "llm"
    assert result.validation["status"] == "pass"
    assert result.song_plan.song_id == project.project_id
    assert len(result.song_plan.sections) == 3
    serialized = json.dumps(result.model_dump(mode="json")).lower()
    assert "note_events" not in serialized
    assert "midi_path" not in serialized
    assert project.metadata["song_plan"]["song_id"] == project.project_id


def test_llm_planner_retries_invalid_json_once_then_accepts_repair():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=402),
        project_id="planner-retry",
    )
    provider = _SequenceProvider(["{not-json", _valid_patch_json()])

    result = LlmPlanner(provider=provider).plan(
        prompt="minor blues con respuesta de trombones",
        project=project,
    )

    assert result.status == "ok"
    assert result.planner == "llm"
    assert [attempt.status for attempt in result.attempts] == ["fail", "pass"]
    assert provider.previous_errors[1] is not None


def test_llm_planner_falls_back_after_two_invalid_llm_outputs():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=403),
        project_id="planner-fallback",
    )

    result = LlmPlanner(provider=_SequenceProvider(["{bad", "{\"midi_path\":\"x.mid\"}"])).plan(
        prompt="hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto",
        project=project,
    )

    assert result.status == "ok"
    assert result.planner == "fallback_rule_based"
    assert result.fallback_used is True
    assert result.validation["status"] == "pass"
    assert [attempt.status for attempt in result.attempts] == ["fail", "fail", "pass"]
    assert result.song_plan.sections


def test_plan_validator_rejects_locked_tracks_and_overlapping_sections():
    validator = PlanValidator()
    patch = json.loads(_valid_patch_json())
    patch["sections"][1]["start_bar"] = 4
    patch["instruments"].append("piano")
    parsed, parse_report = validator.parse_patch_json(json.dumps(patch))

    assert parsed is not None
    assert parse_report["status"] == "fail"
    report = validator.validate_patch(parsed, locked_tracks=["piano"])
    codes = {issue["code"] for issue in report["errors"]}
    assert "sections_overlap" in codes
    assert "locked_track_modified" in codes


class _SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.previous_errors: list[str | None] = []

    def generate_plan_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        previous_error: str | None = None,
    ) -> str:
        self.previous_errors.append(previous_error)
        return self.responses.pop(0)


def _valid_patch_json() -> str:
    return json.dumps(
        {
            "style": "hard_bop",
            "substyle": "minor_blues",
            "tempo": 132,
            "meter": "4/4",
            "key": "C minor",
            "form": "minor_blues_12",
            "ensemble": "jazz_sextet",
            "instruments": [
                "drum_kit",
                "double_bass",
                "piano",
                "alto_sax",
                "trumpet_bflat",
                "trombone",
            ],
            "sections": [
                {
                    "name": "Head",
                    "start_bar": 1,
                    "end_bar": 4,
                    "energy": 0.55,
                    "density_by_role": {
                        "melody": 0.68,
                        "walking_bass": 0.75,
                        "comping": 0.55,
                    },
                    "groove_feel": "swing",
                    "role_focus": ["melody"],
                },
                {
                    "name": "Horn Response",
                    "start_bar": 5,
                    "end_bar": 8,
                    "energy": 0.72,
                    "density_by_role": {
                        "horn_response": 0.78,
                        "drums": 0.7,
                        "comping": 0.62,
                    },
                    "groove_feel": "swing",
                    "role_focus": ["horn_response"],
                },
                {
                    "name": "Turnaround",
                    "start_bar": 9,
                    "end_bar": 12,
                    "energy": 0.82,
                    "density_by_role": {
                        "melody": 0.7,
                        "walking_bass": 0.82,
                        "drums": 0.8,
                    },
                    "groove_feel": "swing",
                    "role_focus": ["melody", "horn_response"],
                },
            ],
            "generation_strategy": {
                "mode": "llm_plan",
                "priority_roles": ["melody", "horn_response", "walking_bass"],
                "role_intents": [
                    {
                        "role": "melody",
                        "instruments": ["alto_sax"],
                        "target_sections": ["Head", "Turnaround"],
                        "density": 0.7,
                        "strategy": "rule_based",
                    }
                ],
                "forbid_audio_models": True,
                "allow_note_generation": False,
                "allow_midi_export": False,
            },
        }
    )
